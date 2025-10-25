"""Lightning DataModule for GRPO rollout collection."""

import asyncio
import os
import tempfile
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import lightning as pl
from openai import AsyncOpenAI

from train_agent.model_schemas import GRPOConfig, SamplingConfig, VLLMConfig
from train_agent.training.trajectory import (
    McpScenario,
    TrajectoryGroup,
    gather_trajectory_groups,
    check_with_groundtruth,
)
from train_agent.training.grpo import (
    calculate_group_advantages,
    create_group_rollout_from_trajectories,
)
from train_agent.training.batch_preparation import (
    prepare_batches_from_trajectory_groups,
    validate_trajectory_alignment,
)
from train_agent.inference.vllm_engine import VLLMEngine
from train_agent.inference.sglang_engine import SGLangEngine
from train_agent.utils.settings import settings


class GRPORolloutDataset(Dataset):
    """Dataset that wraps MCP scenarios for rollout collection."""

    def __init__(self, scenarios: List[McpScenario]):
        self.scenarios = scenarios

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> McpScenario:
        return self.scenarios[idx]


class GRPORolloutDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for GRPO training with vLLM lifecycle and batch generation.

    This DataModule handles the complete online GRPO data pipeline:

    ## Training Flow (per step):
    1. **Save Checkpoint**: Save current model parameters to disk (rank 0 only)
    2. **Start vLLM Server**: Launch vLLM server with fresh checkpoint
       - Rank 0 starts server on port 8011
       - All ranks connect inference clients to rank 0's server
       - Synchronization via barriers ensures all ranks wait for server startup
    3. **Collect Rollouts**: Generate trajectories from current policy
       - Select scenarios for this step (cycling through available scenarios)
       - Run async rollout collection (all ranks participate)
       - Gather trajectory groups with per-token logprobs
    4. **Stop vLLM Server**: Free GPU memory by stopping vLLM
       - Critical for making GPU memory available for training
       - All ranks wait at barrier after shutdown
    5. **Score Trajectories**: Assign rewards based on task completion and verification
       - Two-tier reward system:
         * 1.0: Task completed AND verified correct with ground truth
         * 0.2: Task completed (without verification or wrong answer)
         * 0.0: Task not completed
    6. **Calculate Advantages**: Compute GRPO advantages per trajectory
       - Group rollouts by scenario
       - Calculate relative advantages within each group
    7. **Prepare Batches**: Convert trajectories to training batches
       - Tokenize conversations with position tracking
       - Create loss masks (1 for assistant tokens, 0 for others)
       - Align old logprobs with tokenized positions
       - Calculate step-level statistics (mean/std advantages, num batches)
    8. **Yield Batches**: Return batches one-by-one to Lightning
       - Each batch includes step-level stats as metadata
       - Last batch marked with flag for logging

    ## Multi-GPU Coordination:
    - Only rank 0 manages vLLM server (start/stop)
    - All ranks connect clients and collect rollouts
    - Barriers ensure synchronization at critical points
    - FSDP handles model sharding during training

    ## Key Features:
    - **True Online RL**: Fresh checkpoint and rollouts every step
    - **Memory Management**: vLLM start/stop cycle prevents OOM
    - **Position Tracking**: Accurate token-level loss masking
    - **Step-Level Metrics**: Aggregated statistics for monitoring
    """

    def __init__(
        self,
        grpo_config: GRPOConfig,
        train_scenarios: List[McpScenario],
        val_scenarios: Optional[List[McpScenario]] = None,
        vllm_config: VLLMConfig = VLLMConfig.from_config(),
        sampling_config: SamplingConfig = SamplingConfig(),
        use_sglang: bool = True,  # Use SGLang by default for better performance
    ):
        super().__init__()
        self.grpo_config = grpo_config
        self.train_scenarios = train_scenarios
        self.val_scenarios = val_scenarios

        # Inference backend selection
        self.use_sglang = use_sglang
        self.vllm_config = vllm_config
        self.sampling_config = sampling_config

        # Inference engine and client
        self.inference_engine: Optional[SGLangEngine | VLLMEngine] = None
        self.inference_client: Optional[AsyncOpenAI] = None
        self.checkpoint_dir = "grpo_checkpoint_dir"
        self.server_started = False  # Track if server is already running

        # Scenario tracking
        self.scenario_idx = 0

        # Trainer reference (set in setup)
        self.trainer = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets and start inference server.

        Starting the server here (instead of in train_dataloader) means:
        - Server is ready before training begins
        - Clearer lifecycle management
        - With SGLang: server stays on for entire training session
        - With vLLM: server restarts are handled in _update_model_weights()
        """
        if stage == "fit" or stage is None:
            self.train_dataset = GRPORolloutDataset(self.train_scenarios)
            if self.val_scenarios:
                self.val_dataset = GRPORolloutDataset(self.val_scenarios)

            # Start inference server once at setup time
            print(f"\n{'='*80}")
            print(f"[DataModule Setup] Starting inference server...")
            print(f"Backend: {'SGLang (with hot weight updates)' if self.use_sglang else 'vLLM (with server restarts)'}")
            print(f"{'='*80}")
            self._start_inference_server()
            print(f"✓ Inference server ready and waiting for training to begin")

    def _save_model_checkpoint_for_vllm(self) -> str:
        """Save current model state to a temporary checkpoint for vLLM server."""
        lightning_module = self.trainer.lightning_module
        global_step = lightning_module.global_step

        checkpoint_path = os.path.join(self.checkpoint_dir, f"step_{global_step}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model and tokenizer
        lightning_module.model.save_pretrained(checkpoint_path)
        lightning_module.tokenizer.save_pretrained(checkpoint_path)

        print(f"Saved checkpoint for vLLM at: {checkpoint_path}")
        return checkpoint_path

    def _start_inference_server(self):
        """Start inference server (SGLang or vLLM) if not already running."""
        if self.server_started:
            print("Inference server already running, skipping startup")
            return

        global_rank = self.trainer.global_rank

        # Only rank 0 starts the inference server
        if global_rank == 0:
            if self.use_sglang:
                print("Starting SGLang server (with weight update support)...")
                self.inference_engine = SGLangEngine(self.vllm_config)
                self.inference_engine.start_server(port=30000, host="0.0.0.0")
                print("✓ SGLang server started for rollout collection on rank 0")
            else:
                print("Starting vLLM server (traditional mode)...")
                self.inference_engine = VLLMEngine(self.vllm_config)
                self.inference_engine.start_server(port=8011, host="0.0.0.0")
                print("✓ vLLM server started for rollout collection on rank 0")
        else:
            print(f"Rank {global_rank} waiting for inference server from rank 0...")

        # All ranks wait for server to be ready
        if hasattr(self.trainer.strategy, 'barrier'):
            self.trainer.strategy.barrier()

        # All ranks initialize inference client
        if self.use_sglang:
            self.inference_client = AsyncOpenAI(
                base_url="http://localhost:30000/v1",
                api_key="EMPTY"
            )
        else:
            self.inference_client = AsyncOpenAI(
                base_url="http://localhost:8011/v1",
                api_key="EMPTY"
            )

        if global_rank != 0:
            print(f"Rank {global_rank} connected to inference server")

        self.server_started = True

    def _update_model_weights(self, checkpoint_path: str):
        """Update model weights in running inference server.

        For SGLang: Hot-swap weights without restart (fast!)
        For vLLM: Stop and restart server (slow, legacy mode)
        """
        global_rank = self.trainer.global_rank

        if self.use_sglang:
            # SGLang: Just update weights, no restart needed!
            print(f"[SGLang] Updating weights from checkpoint: {checkpoint_path}")
            if global_rank == 0:
                success = self.inference_engine.update_weights_from_disk(checkpoint_path)
                if success:
                    print("✓ Weights updated successfully (zero downtime!)")
                else:
                    print("✗ Weight update failed, consider restarting server")

            # All ranks wait for weight update to complete
            if hasattr(self.trainer.strategy, 'barrier'):
                self.trainer.strategy.barrier()
        else:
            # vLLM: Legacy mode - restart server with new checkpoint
            print(f"[vLLM] Restarting server with checkpoint: {checkpoint_path}")
            self._stop_inference_server()
            self.vllm_config.model_name = checkpoint_path
            self._start_inference_server()

    def _stop_inference_server(self):
        """Stop inference server to free GPU memory."""
        global_rank = self.trainer.global_rank

        # Only rank 0 stops the inference server
        if global_rank == 0:
            if self.inference_engine is not None:
                self.inference_engine.stop_server()
                self.inference_engine = None
                engine_type = "SGLang" if self.use_sglang else "vLLM"
                print(f"{engine_type} server stopped on rank 0")
        else:
            print(f"Rank {global_rank} disconnecting from inference server...")

        # All ranks clear their inference client
        self.inference_client = None
        self.server_started = False

        # All ranks wait for cleanup to complete
        if hasattr(self.trainer.strategy, 'barrier'):
            self.trainer.strategy.barrier()

    def _get_scenarios_for_step(self) -> List[McpScenario]:
        """Get scenarios for current training step."""
        scenarios = []
        for _ in range(self.grpo_config.groups_per_step):
            scenario = self.train_scenarios[self.scenario_idx % len(self.train_scenarios)]
            scenarios.append(scenario)
            self.scenario_idx += 1
        return scenarios

    async def _collect_rollouts_async(self) -> List[TrajectoryGroup]:
        """Collect rollouts from vLLM server asynchronously."""
        if self.inference_client is None:
            raise RuntimeError("Inference client not initialized")

        scenarios = self._get_scenarios_for_step()
        lightning_module = self.trainer.lightning_module

        print(f"\nCollecting {len(scenarios)} trajectory groups...")
        print(f"  - Rollouts per group: {self.grpo_config.rollouts_per_group}")
        print(f"  - Max turns: {self.grpo_config.max_turns}")

        trajectory_groups = await gather_trajectory_groups(
            inference_client=self.inference_client,
            model_name=self.vllm_config.model_name,
            scenarios=scenarios,
            rollouts_per_group=self.grpo_config.rollouts_per_group,
            sampling_config=self.sampling_config,
            debug=False,
            mcp_url=settings.mcp_url,
            tokenizer=lightning_module.tokenizer,
        )

        print(f"Collected {len(trajectory_groups)} trajectory groups")
        return trajectory_groups

    def _score_and_prepare_batches(
        self, trajectory_groups: List[TrajectoryGroup]
    ) -> List[Dict[str, torch.Tensor]]:
        """Score trajectories, calculate advantages, and prepare training batches."""
        lightning_module = self.trainer.lightning_module

        # Score trajectories with two-tier reward system:
        # - 1.0: Task completed AND verified correct with ground truth
        # - 0.2: Task completed (without verification or wrong answer)
        # - 0.0: Task not completed
        print("Scoring trajectories...")
        for group in trajectory_groups:
            for traj in group.trajectories:
                task_completed = traj.metrics.get("task_completed", False)
                expected_answer = traj.expected_answer

                if not task_completed:
                    traj.reward = 0.0
                    print(f"  ✗ Task not completed - Reward: {traj.reward}")
                elif expected_answer:
                    answer_correct = check_with_groundtruth(traj, expected_answer)
                    traj.reward = 1.0 if answer_correct else 0.2
                    print(f"  {'✓' if answer_correct else '~'} Expected: '{expected_answer}' - "
                          f"{'Verified correct' if answer_correct else 'Completed but wrong'} - Reward: {traj.reward}")
                else:
                    traj.reward = 0.2
                    print(f"  ~ Task completed (no ground truth) - Reward: {traj.reward}")

        # Calculate advantages
        print("Calculating advantages...")
        advantages_list = []
        for group in trajectory_groups:
            group_rollout = create_group_rollout_from_trajectories(
                scenario_id=group.scenario_id,
                trajectories=group.trajectories,
                rewards=group.rewards,
            )

            advantages = calculate_group_advantages(
                group_rollout, advantage_type=self.grpo_config.advantage_type
            )
            advantages_list.append(advantages)

            print(f"  Group {group.scenario_id[:30]}...: "
                  f"rewards={group.rewards}, advantages={advantages}")

        # Validate alignment for first trajectory
        if trajectory_groups and trajectory_groups[0].trajectories:
            validate_trajectory_alignment(
                trajectory_groups[0].trajectories[0],
                lightning_module.tokenizer,
                verbose=True,
            )

        # Prepare training batches
        print("Preparing training batches...")
        train_dataloader = prepare_batches_from_trajectory_groups(
            trajectory_groups=trajectory_groups,
            advantages_list=advantages_list,
            tokenizer=lightning_module.tokenizer,
            batch_size=self.grpo_config.training_config.batch_size,
            max_length=self.vllm_config.max_seq_length,
            crop_to_actual_length=True,
        )

        batches = list(train_dataloader)
        print(f"Prepared {len(batches)} training batches")

        return batches

    def create_mock_batches(self, num_batches: int = 3, seq_len: int = 512) -> List[Dict[str, torch.Tensor]]:
        """
        Create mock batches for quick testing.

        Args:
            num_batches: Number of batches to create
            seq_len: Sequence length for each sample

        Returns:
            List of mock batches with input_ids, attention_mask, loss_mask, old_logprobs, advantages
        """
        lightning_module = self.trainer.lightning_module
        batch_size = self.grpo_config.training_config.batch_size
        vocab_size = lightning_module.tokenizer.vocab_size

        print(f"\nCreating {num_batches} mock batches...")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Sequence length: {seq_len}")

        batches = []
        for i in range(num_batches):
            batch = {
                'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long),
                'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long),
                'loss_mask': (torch.rand(batch_size, seq_len) > 0.5).float(),
                'old_logprobs': torch.randn(batch_size, seq_len) - 2.0,
                'advantages': torch.randn(batch_size) * 0.5,
            }
            batches.append(batch)
            print(f"  Mock batch {i+1}/{num_batches} created")

        return batches

    def train_dataloader(self) -> DataLoader:
        """Generate batches through full rollout collection pipeline.

        Note: Inference server is already started in setup(), so we just use it here.
        """
        def batch_generator():
            # Server is already running from setup()
            for step_idx in range(self.grpo_config.training_config.max_training_steps):
                print(f"\n{'='*80}")
                print(f"DataModule: Generating batches for step {step_idx}")
                print(f"{'='*80}")

                # Save checkpoint and update weights (SGLang) or restart server (vLLM)
                if step_idx > 0:  # Skip on first step (using base model)
                    checkpoint_path = self._save_model_checkpoint_for_vllm()
                    self._update_model_weights(checkpoint_path)
                else:
                    print(f"Step 0: Using base model directly (no checkpoint needed)")

                # Collect rollouts
                import nest_asyncio
                nest_asyncio.apply()
                trajectory_groups = asyncio.run(self._collect_rollouts_async())

                # Score and prepare batches
                batches = self._score_and_prepare_batches(trajectory_groups)

                # Calculate step-level statistics
                all_advantages = []
                for batch in batches:
                    all_advantages.extend(batch['advantages'].cpu().tolist())

                step_stats = {
                    'mean_advantage': torch.tensor(all_advantages).mean().item() if all_advantages else 0.0,
                    'std_advantage': torch.tensor(all_advantages).std().item() if all_advantages else 0.0,
                    'num_batches': len(batches),
                }

                # Yield all batches from this step with metadata
                for i, batch in enumerate(batches):
                    # Add step-level stats and batch index to each batch
                    batch['step_stats'] = step_stats
                    batch['is_last_batch_in_step'] = (i == len(batches) - 1)
                    yield batch

        # Return iterable directly (Lightning handles this)
        class BatchIterableDataset(IterableDataset):
            def __init__(self, generator_fn):
                self.generator_fn = generator_fn

            def __iter__(self):
                return self.generator_fn()

        dataset = BatchIterableDataset(batch_generator)
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        if hasattr(self, 'val_dataset'):
            return DataLoader(
                self.val_dataset,
                batch_size=1,
                num_workers=0,
                shuffle=False,
            )
        # Return an empty Dataset-backed DataLoader so Lightning's sanity/validation can run without errors
        class EmptyDataset(Dataset):
            def __len__(self) -> int:
                return 0

            def __getitem__(self, idx: int):
                raise IndexError("Empty dataset")

        return DataLoader(EmptyDataset(), batch_size=1, num_workers=0)

    def teardown(self, stage: Optional[str] = None):
        """Clean up resources when training/testing is complete.

        This stops the inference server to free GPU memory.
        """
        if stage == "fit" or stage is None:
            if self.server_started:
                print(f"\n{'='*80}")
                print(f"[DataModule Teardown] Stopping inference server...")
                print(f"{'='*80}")
                self._stop_inference_server()
                print(f"✓ Inference server stopped and GPU memory freed")


"""Lightning DataModule for GRPO rollout collection."""

import asyncio
import os
import tempfile
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from openai import AsyncOpenAI

from train_agent.model_schemas import GRPOConfig, SamplingConfig, VLLMConfig
from train_agent.training.trajectory import (
    McpScenario,
    TrajectoryGroup,
    gather_trajectory_groups,
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
    5. **Score Trajectories**: Assign rewards based on task completion
       - Currently uses placeholder rewards (1.0 for completed, 0.0 otherwise)
       - TODO: Implement proper trajectory judging/scoring
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
    ):
        super().__init__()
        self.grpo_config = grpo_config
        self.train_scenarios = train_scenarios
        self.val_scenarios = val_scenarios

        # vLLM and sampling configs
        self.vllm_config = vllm_config
        self.sampling_config = sampling_config

        # vLLM engine and client
        self.vllm_engine: Optional[VLLMEngine] = None
        self.inference_client: Optional[AsyncOpenAI] = None
        self.checkpoint_dir = tempfile.mkdtemp(prefix="grpo_checkpoint_")
        self.first_checkpoint_path: Optional[str] = None

        # Scenario tracking
        self.scenario_idx = 0

        # Trainer reference (set in setup)
        self.trainer = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = GRPORolloutDataset(self.train_scenarios)
            if self.val_scenarios:
                self.val_dataset = GRPORolloutDataset(self.val_scenarios)

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

    def _start_vllm_server(self, checkpoint_path: str):
        """Start vLLM server with the current model checkpoint."""
        global_rank = self.trainer.global_rank

        # Update vLLM config to use the checkpoint
        self.vllm_config.model_name = checkpoint_path

        # Only rank 0 starts the vLLM server
        if global_rank == 0:
            self.vllm_engine = VLLMEngine(self.vllm_config)
            self.vllm_engine.start_server(port=8011, host="0.0.0.0")
            print("vLLM server started for rollout collection on rank 0")
        else:
            print(f"Rank {global_rank} waiting for vLLM server from rank 0...")

        # All ranks wait for server to be ready
        if hasattr(self.trainer.strategy, 'barrier'):
            self.trainer.strategy.barrier()

        # All ranks initialize inference client
        self.inference_client = AsyncOpenAI(
            base_url="http://localhost:8011/v1",
            api_key="EMPTY"
        )

        if global_rank != 0:
            print(f"Rank {global_rank} connected to vLLM server")

    def _stop_vllm_server(self):
        """Stop vLLM server to free GPU memory."""
        global_rank = self.trainer.global_rank

        # Only rank 0 stops the vLLM server
        if global_rank == 0:
            if self.vllm_engine is not None:
                self.vllm_engine.stop_server()
                self.vllm_engine = None
                print("vLLM server stopped on rank 0")
        else:
            print(f"Rank {global_rank} disconnecting from vLLM server...")

        # All ranks clear their inference client
        self.inference_client = None

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
            debug=True,
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

        # Score trajectories
        print("Scoring trajectories...")
        for group in trajectory_groups:
            for traj in group.trajectories:
                traj.reward = 1.0 if traj.metrics.get("task_completed", False) else 0.0

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
        """Generate batches through full rollout collection pipeline."""
        def batch_generator():
            for step_idx in range(self.grpo_config.training_config.max_training_steps):
                print(f"\n{'='*80}")
                print(f"DataModule: Generating batches for step {step_idx}")
                print(f"{'='*80}")

                # Save fresh checkpoint every step and start vLLM server
                global_rank = self.trainer.global_rank
                if global_rank == 0:
                    # Save fresh checkpoint with current model parameters
                    checkpoint = self._save_model_checkpoint_for_vllm()
                    # Store first checkpoint for potential reuse in other contexts
                    if self.first_checkpoint_path is None:
                        self.first_checkpoint_path = checkpoint
                else:
                    checkpoint = None

                # Broadcast checkpoint path to all ranks
                # All ranks know the checkpoint path pattern
                if checkpoint is None:
                    lightning_module = self.trainer.lightning_module
                    checkpoint = os.path.join(self.checkpoint_dir, f"step_{lightning_module.global_step}")

                print(f"Starting vLLM server with checkpoint: {checkpoint}")
                self._start_vllm_server(checkpoint)

                # Collect rollouts
                import nest_asyncio
                nest_asyncio.apply()
                trajectory_groups = asyncio.run(self._collect_rollouts_async())

                # Stop vLLM server
                self._stop_vllm_server()

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
        class BatchIterableDataset:
            def __init__(self, generator_fn):
                self.generator_fn = generator_fn

            def __iter__(self):
                return self.generator_fn()

        dataset = BatchIterableDataset(batch_generator)
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def val_dataloader(self) -> Optional[DataLoader]:
        if hasattr(self, 'val_dataset'):
            return DataLoader(
                self.val_dataset,
                batch_size=1,
                num_workers=0,
                shuffle=False,
            )
        return None


"""
PyTorch Lightning module for GRPO training with online rollout collection.
"""

import asyncio
import os
import tempfile
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import AsyncOpenAI

from train_agent.model_schemas import GRPOConfig, SamplingConfig, VLLMConfig
from train_agent.training.grpo import (
    compute_drgrpo_loss,
    calculate_group_advantages,
    create_group_rollout_from_trajectories,
)
from train_agent.training.trajectory import (
    McpScenario,
    TrajectoryGroup,
    gather_trajectory_groups,
)
from train_agent.training.batch_preparation import (
    prepare_batches_from_trajectory_groups,
    validate_trajectory_alignment,
)
from train_agent.inference.vllm_engine import VLLMEngine
from train_agent.utils.settings import settings


class GRPOLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for online GRPO training with rollout collection.

    This module handles:
    - Online rollout collection from vLLM server
    - Dynamic vLLM server start/stop for GPU memory management
    - GRPO loss computation with advantages
    - Model checkpointing and optimization
    """

    def __init__(
        self,
        grpo_config: GRPOConfig,
        train_scenarios: List[McpScenario],
        vllm_config: Optional[VLLMConfig] = None,
        sampling_config: Optional[SamplingConfig] = None,
    ):
        """
        Initialize GRPO Lightning module with online rollout capability.

        Args:
            grpo_config: GRPO configuration
            train_scenarios: List of training scenarios for rollout collection
            vllm_config: vLLM configuration (optional, will use defaults if None)
            sampling_config: Sampling configuration for rollouts (optional)
        """
        super().__init__()
        self.grpo_config = grpo_config
        self.save_hyperparameters(grpo_config.model_dump())

        # Store training scenarios for rollout collection
        self.train_scenarios = train_scenarios
        self.scenario_idx = 0  # Track which scenarios to use per step

        # Store training config attributes as instance attributes
        self.learning_rate = grpo_config.training_config.learning_rate
        self.weight_decay = grpo_config.training_config.weight_decay
        self.warmup_steps = grpo_config.training_config.warmup_steps
        self.max_steps = grpo_config.training_config.max_training_steps
        self.gradient_clip_val = grpo_config.training_config.gradient_clip_val

        # Store GRPO-specific attributes
        self.advantage_type = grpo_config.advantage_type
        self.rollouts_per_group = grpo_config.rollouts_per_group
        self.groups_per_step = grpo_config.groups_per_step
        self.clip_epsilon = grpo_config.clip_epsilon
        self.max_turns = grpo_config.max_turns

        # vLLM and sampling configs
        self.vllm_config = vllm_config or VLLMConfig.from_config()
        self.sampling_config = sampling_config or SamplingConfig(
            temperature=0.7, top_p=0.9, max_tokens=8000
        )

        # vLLM engine (will be initialized when needed)
        self.vllm_engine: Optional[VLLMEngine] = None
        self.inference_client: Optional[AsyncOpenAI] = None
        self.checkpoint_dir = tempfile.mkdtemp(prefix="grpo_checkpoint_")
        self.first_checkpoint_path: Optional[str] = None  # Store first checkpoint path

        # Model will be initialized in configure_sharded_model() after FSDP is set up
        self.model = None
        
        # Load tokenizer (lightweight, can be done in __init__)
        self.tokenizer = AutoTokenizer.from_pretrained(
            grpo_config.model_name,
            trust_remote_code=grpo_config.trust_remote_code
        )

        # Batch preparation parameters
        self.batch_size = grpo_config.training_config.batch_size
        self.max_length = 8192

        # Cache for current step's batches
        self.current_step_batches: List[Dict[str, torch.Tensor]] = []
        self.batch_iterator_idx = 0

    def configure_sharded_model(self):
        """
        Initialize model AFTER FSDP is set up.
        
        This is called by Lightning when using FSDPStrategy, ensuring the model
        is properly sharded during initialization, preventing each GPU from 
        loading the full model into memory.
        
        Official PyTorch Lightning pattern for efficient FSDP initialization.
        """
        if self.model is None:
            print(f"\n{'='*80}")
            print(f"Loading model {self.grpo_config.model_name} with FSDP sharding...")
            print(f"{'='*80}")
            
            # Convert dtype string to torch dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.grpo_config.torch_dtype, torch.bfloat16)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.grpo_config.model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=self.grpo_config.trust_remote_code,
            )
            print(f"Model loaded and sharded successfully! (dtype={torch_dtype})")
            print(f"{'='*80}\n")

    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the model."""
        if self.model is None:
            raise RuntimeError(
                "Model not initialized. configure_sharded_model() should have been called by Lightning."
            )

        # Ensure tensors are on the same device as the sharded model
        model_device = next(self.model.parameters()).device
        if input_ids.device != model_device:
            input_ids = input_ids.to(model_device)
        if attention_mask is not None and attention_mask.device != model_device:
            attention_mask = attention_mask.to(model_device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.logits

    def compute_token_logprobs(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute token-level log probabilities for each token in the batch.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] (optional)

        Returns:
            Token-level log probabilities [batch_size, seq_len-1]
            (seq_len-1 because we predict tokens 1:N from positions 0:N-1)
        """
        # Forward pass to get logits
        logits = self(input_ids, attention_mask)

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Extract log probs for actual tokens (shift to align predictions with targets)
        # Predictions at positions 0:N-1 predict tokens at positions 1:N
        token_logprobs = torch.gather(
            log_probs[:, :-1, :],  # predictions: exclude last position
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1)  # targets: exclude first position
        ).squeeze(-1)  # [batch_size, seq_len-1]

        return token_logprobs

    def _save_model_checkpoint_for_vllm(self) -> str:
        """Save current model state to a temporary checkpoint for vLLM server."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"step_{self.global_step}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        print(f"Saved checkpoint for vLLM at: {checkpoint_path}")
        return checkpoint_path

    def _start_vllm_server(self, checkpoint_path: str):
        """Start vLLM server with the current model checkpoint.

        Only rank 0 starts the server to avoid port collisions in DDP.
        All other ranks wait for the server to be ready.
        """
        # Update vLLM config to use the checkpoint
        self.vllm_config.model_name = checkpoint_path

        # Only rank 0 starts the vLLM server
        if self.global_rank == 0:
            # Initialize vLLM engine
            self.vllm_engine = VLLMEngine(self.vllm_config)
            self.vllm_engine.start_server(port=8011, host="0.0.0.0")
            print("vLLM server started for rollout collection on rank 0")
        else:
            print(f"Rank {self.global_rank} waiting for vLLM server from rank 0...")

        # All ranks wait for server to be ready (barrier synchronization)
        if self.trainer is not None and hasattr(self.trainer.strategy, 'barrier'):
            self.trainer.strategy.barrier()

        # All ranks initialize inference client (pointing to rank 0's server)
        self.inference_client = AsyncOpenAI(
            base_url="http://localhost:8011/v1",
            api_key="EMPTY"
        )

        if self.global_rank != 0:
            print(f"Rank {self.global_rank} connected to vLLM server")

    def _stop_vllm_server(self):
        """Stop vLLM server to free GPU memory.

        Only rank 0 stops the server since it's the only one that started it.
        """
        # Only rank 0 stops the vLLM server
        if self.global_rank == 0:
            if self.vllm_engine is not None:
                self.vllm_engine.stop_server()
                self.vllm_engine = None
                print("vLLM server stopped on rank 0")
        else:
            print(f"Rank {self.global_rank} disconnecting from vLLM server...")

        # All ranks clear their inference client
        self.inference_client = None

        # All ranks wait for cleanup to complete (barrier synchronization)
        if self.trainer is not None and hasattr(self.trainer.strategy, 'barrier'):
            self.trainer.strategy.barrier()

    def _get_scenarios_for_step(self) -> List[McpScenario]:
        """Get scenarios for current training step (cycling through available scenarios)."""
        scenarios = []
        for _ in range(self.groups_per_step):
            scenario = self.train_scenarios[self.scenario_idx % len(self.train_scenarios)]
            scenarios.append(scenario)
            self.scenario_idx += 1
        return scenarios

    async def _collect_rollouts_async(self) -> List[TrajectoryGroup]:
        """Collect rollouts from vLLM server asynchronously."""
        if self.inference_client is None:
            raise RuntimeError("Inference client not initialized. Call _start_vllm_server first.")

        scenarios = self._get_scenarios_for_step()

        print(f"\nCollecting {len(scenarios)} trajectory groups...")
        print(f"  - Rollouts per group: {self.rollouts_per_group}")
        print(f"  - Max turns: {self.max_turns}")

        # Gather trajectories
        trajectory_groups = await gather_trajectory_groups(
            inference_client=self.inference_client,
            model_name=self.vllm_config.model_name,
            scenarios=scenarios,
            rollouts_per_group=self.rollouts_per_group,
            sampling_config=self.sampling_config,
            debug=True,
            mcp_url=settings.mcp_url,
            tokenizer=self.tokenizer,
        )

        print(f"Collected {len(trajectory_groups)} trajectory groups")
        return trajectory_groups

    def _score_and_prepare_batches(
        self, trajectory_groups: List[TrajectoryGroup]
    ) -> List[Dict[str, torch.Tensor]]:
        """Score trajectories, calculate advantages, and prepare training batches."""

        # TODO: Implement proper trajectory scoring/judging
        # For now, using placeholder rewards
        print("Scoring trajectories...")
        for group in trajectory_groups:
            for traj in group.trajectories:
                traj.reward = 1.0 if traj.metrics.get("task_completed", False) else 0.0

        # Calculate advantages for each group
        print("Calculating advantages...")
        advantages_list = []
        for group in trajectory_groups:
            group_rollout = create_group_rollout_from_trajectories(
                scenario_id=group.scenario_id,
                trajectories=group.trajectories,
                rewards=group.rewards,
            )

            advantages = calculate_group_advantages(
                group_rollout, advantage_type=self.advantage_type
            )
            advantages_list.append(advantages)

            print(f"  Group {group.scenario_id[:30]}...: "
                  f"rewards={group.rewards}, advantages={advantages}")

        # Validate alignment for first trajectory (sanity check)
        if trajectory_groups and trajectory_groups[0].trajectories:
            is_valid = validate_trajectory_alignment(
                trajectory_groups[0].trajectories[0],
                self.tokenizer,
                verbose=True,
            )
            # if not is_valid:
            #     raise ValueError("Trajectory alignment validation failed")

        # Prepare training batches
        print("Preparing training batches...")
        train_dataloader = prepare_batches_from_trajectory_groups(
            trajectory_groups=trajectory_groups,
            advantages_list=advantages_list,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

        batches = list(train_dataloader)
        print(f"Prepared {len(batches)} training batches")

        return batches

    def _create_mock_batches(self, num_batches: int = 3, seq_len: int = 512) -> List[Dict[str, torch.Tensor]]:
        """
        Create mock batches for testing batch distribution in DDP/FSDP.

        Args:
            num_batches: Number of batches to create (default: 3)
            seq_len: Sequence length for each sample (default: 512)

        Returns:
            List of mock batches matching the expected format:
            {
                'input_ids': [batch_size, seq_len],
                'attention_mask': [batch_size, seq_len],
                'loss_mask': [batch_size, seq_len],
                'old_logprobs': [batch_size, seq_len],
                'advantages': [batch_size],
            }
        """
        print(f"\nCreating {num_batches} mock batches...")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Sequence length: {seq_len}")
        print(f"  - Total samples: {num_batches * self.batch_size}")

        vocab_size = self.tokenizer.vocab_size
        batches = []

        for i in range(num_batches):
            # Create random input_ids (simulating tokenized sequences)
            # NOTE: Create on CPU first, will be moved to device in training_step
            input_ids = torch.randint(
                low=0, high=vocab_size, size=(self.batch_size, seq_len), dtype=torch.long
            )

            # Create attention mask (all ones - no padding for simplicity)
            attention_mask = torch.ones(self.batch_size, seq_len, dtype=torch.long)

            # Create loss mask (randomly mask ~50% of tokens as assistant tokens)
            # In real scenario, this would be 1 for assistant tokens, 0 for system/user/tool
            loss_mask = torch.rand(self.batch_size, seq_len) > 0.5
            loss_mask = loss_mask.float()

            # Create old logprobs (random log probabilities)
            # Typically these are negative values (log of probabilities)
            old_logprobs = torch.randn(self.batch_size, seq_len) - 2.0  # roughly in range [-5, 1]

            # Create advantages (one per trajectory/sample in batch)
            # Typically these are centered around 0 with some positive and negative values
            advantages = torch.randn(self.batch_size) * 0.5  # roughly in range [-1.5, 1.5]

            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'loss_mask': loss_mask,
                'old_logprobs': old_logprobs,
                'advantages': advantages,
            }

            batches.append(batch)

            print(f"  Mock batch {i+1}/{num_batches} created:")
            print(f"    - input_ids: {input_ids.shape}")
            print(f"    - loss_mask sum: {loss_mask.sum().item():.0f} tokens (out of {self.batch_size * seq_len})")
            print(f"    - advantages mean: {advantages.mean().item():.3f}, std: {advantages.std().item():.3f}")

        return batches

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Online GRPO training step - collects fresh rollouts EVERY step.

        Flow (happens every single training step):
        1. Start vLLM server with latest checkpoint or a default checkpoint
        2. Collect rollouts from current policy
        3. Stop vLLM server
        4. Score trajectories & calculate advantages
        5. Prepare batches and compute loss on ALL batches
        6. Save current model checkpoint for vLLM
        7. Return aggregated loss
        8. Save current model checkpoint for next step

        This is TRUE online policy RL - fresh data every step!
        """
        print(f"\n{'='*80}")
        print(f"ONLINE TRAINING STEP {self.global_step}")
        print(f"{'='*80}")


        print(f"Starting vLLM server with checkpoint: {self.first_checkpoint_path or self.vllm_config.model_name}")
        # Step 2: Start vLLM server with first checkpoint (always use the first one)
        self._start_vllm_server(self.first_checkpoint_path or self.vllm_config.model_name)

        # Step 3: Collect rollouts from current policy
        # Handle async call in sync context
        import nest_asyncio
        nest_asyncio.apply()
        trajectory_groups = asyncio.run(self._collect_rollouts_async())

        # Step 4: Stop vLLM server to free GPU memory for training
        self._stop_vllm_server()

        # Step 5 & 6: Score trajectories and prepare batches
        batches = self._score_and_prepare_batches(trajectory_groups)

        print(f"\nTraining on {len(batches)} batches from fresh rollouts...")

        # Step 7: Compute loss across all batches from this rollout collection
        total_loss = 0.0
        total_advantages = []

        for i, training_batch in enumerate(batches):
            # Extract batch data
            # Get device from model parameters (works with FSDP)
            model_device = next(self.model.parameters()).device
            
            input_ids = training_batch['input_ids'].to(model_device)
            attention_mask = training_batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_device)
            loss_mask = training_batch['loss_mask'].to(model_device)
            old_logprobs = training_batch['old_logprobs'].to(model_device)
            advantages = training_batch['advantages'].to(model_device)

            # Compute token-level log probabilities for current policy
            new_logprobs = self.compute_token_logprobs(input_ids, attention_mask)

            # Align old_logprobs and loss_mask to match new_logprobs shape (seq_len-1)
            old_logprobs_aligned = old_logprobs[:, 1:]
            loss_mask_aligned = loss_mask[:, 1:]

            # Compute GRPO loss with masking
            loss = compute_drgrpo_loss(
                new_logprobs=new_logprobs,
                old_logprobs=old_logprobs_aligned,
                advantages=advantages,
                loss_mask=loss_mask_aligned,
                clip_epsilon=self.clip_epsilon,
            )

            total_loss += loss
            total_advantages.extend(advantages.cpu().tolist())

            print(f"  Batch {i+1}/{len(batches)}: loss={loss.item():.4f}")

        # Average loss across all batches
        avg_loss = total_loss / len(batches)

        # Logging
        self.log('train/loss', avg_loss, prog_bar=True)
        self.log('train/mean_advantage', torch.tensor(total_advantages).mean(), prog_bar=False)
        self.log('train/std_advantage', torch.tensor(total_advantages).std(), prog_bar=False)
        self.log('train/num_batches', len(batches), prog_bar=False)

        print(f"\nStep {self.global_step} complete: avg_loss={avg_loss.item():.4f}")
        print(f"{'='*80}\n")

        # Step 8: Save current model checkpoint for vLLM
        if self.global_rank == 0 and self.first_checkpoint_path is None:
            self.first_checkpoint_path = self._save_model_checkpoint_for_vllm()
            print(f"checkpoint saved: {self.first_checkpoint_path}")

        return avg_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step (optional)."""
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        loss_mask = batch['loss_mask']
        old_logprobs = batch['old_logprobs']
        advantages = batch['advantages']

        # Compute token-level log probabilities for current policy
        new_logprobs = self.compute_token_logprobs(input_ids, attention_mask)

        # Align old_logprobs and loss_mask
        old_logprobs_aligned = old_logprobs[:, 1:]
        loss_mask_aligned = loss_mask[:, 1:]

        # Compute GRPO loss with masking
        loss = compute_drgrpo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs_aligned,
            advantages=advantages,
            loss_mask=loss_mask_aligned,
            clip_epsilon=self.clip_epsilon,
        )

        # Logging
        self.log('val/loss', loss)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Cosine annealing scheduler with warmup
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_steps - self.warmup_steps,
            eta_min=self.learning_rate * 0.1,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

    def train_dataloader(self):
        """
        Dummy dataloader for Lightning - we collect data dynamically in training_step.

        Returns a simple iterable that yields a placeholder dict for each training step.
        Lightning requires a dataloader, but we handle data collection internally.
        """
        from torch.utils.data import Dataset
        
        class DummyDataset(Dataset):
            def __init__(self, num_steps):
                self.num_steps = num_steps

            def __len__(self):
                return self.num_steps

            def __getitem__(self, idx):
                # Return a placeholder dict instead of None to satisfy collate_fn
                return {"idx": idx}

        def dummy_collate_fn(batch):
            """Custom collate function that just returns the batch as-is."""
            # We don't actually use this data - real data is collected in training_step
            return batch[0] if batch else {"idx": 0}

        from torch.utils.data import DataLoader
        dataset = DummyDataset(self.max_steps)
        return DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=dummy_collate_fn)

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms before optimizer step."""
        if self.global_step % 10 == 0:  # Log every 10 steps
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_val
            )
            self.log('train/grad_norm', grad_norm)

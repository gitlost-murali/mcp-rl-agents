"""
PyTorch Lightning module for GRPO training.
"""

from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_agent.model_schemas import GRPOConfig
from train_agent.training.grpo import compute_drgrpo_loss


class GRPOLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for GRPO training.

    This module handles:
    - Model forward pass and log probability computation
    - GRPO loss computation with clipping and advantages
    - Optimizer and learning rate scheduler configuration
    - Gradient clipping and metric logging

    Data handling (vLLM server, rollout collection, scoring, batch preparation)
    is delegated to GRPORolloutDataModule in data/lightning_data_module.py.
    """

    def __init__(self, grpo_config: GRPOConfig):
        """
        Initialize GRPO Lightning module.

        Args:
            grpo_config: GRPO configuration
        """
        super().__init__()
        self.grpo_config = grpo_config
        self.save_hyperparameters(grpo_config.model_dump())

        # Store training config attributes
        self.learning_rate = grpo_config.training_config.learning_rate
        self.weight_decay = grpo_config.training_config.weight_decay
        self.warmup_steps = grpo_config.training_config.warmup_steps
        self.max_steps = grpo_config.training_config.max_training_steps
        self.gradient_clip_val = grpo_config.training_config.gradient_clip_val

        # Store GRPO-specific attributes
        self.clip_epsilon = grpo_config.clip_epsilon

        # Model will be initialized in configure_sharded_model()
        self.model = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            grpo_config.model_name,
            trust_remote_code=grpo_config.trust_remote_code
        )

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
            ).train()  # Set to training mode (HuggingFace models default to eval)
            print(f"Model loaded and sharded successfully! (dtype={torch_dtype})")
            print(f"{'='*80}\n")
            self.model.gradient_checkpointing_enable()


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


    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        GRPO training step - computes loss on batches from DataModule.

        Args:
            batch: Dictionary containing:
                - input_ids: [batch_size, seq_len]
                - attention_mask: [batch_size, seq_len]
                - loss_mask: [batch_size, seq_len] (1 for assistant tokens, 0 for others)
                - old_logprobs: [batch_size, seq_len] (from rollout collection)
                - advantages: [batch_size] (from GRPO advantage calculation)
            batch_idx: Batch index

        Returns:
            GRPO loss tensor
        """
        # Get device from model parameters (works with FSDP)
        model_device = next(self.model.parameters()).device

        # Move batch to device (already cropped by collate_fn in DataLoader)
        input_ids = batch['input_ids'].to(model_device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)
        loss_mask = batch['loss_mask'].to(model_device)
        old_logprobs = batch['old_logprobs'].to(model_device)
        advantages = batch['advantages'].to(model_device)

        print("="*80)
        print(f"Batch shape: {input_ids.shape[0]} seqs x {input_ids.shape[1]} tokens (cropped tokens)")
        print("="*80)

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

        # Logging
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/advantage', advantages.mean(), prog_bar=False)

        # Log step-level stats if this is the last batch in the step
        if batch.get('is_last_batch_in_step', False):
            step_stats = batch.get('step_stats', {})
            if step_stats:
                self.log('train/mean_advantage', step_stats['mean_advantage'], prog_bar=False)
                self.log('train/std_advantage', step_stats['std_advantage'], prog_bar=False)
                self.log('train/num_batches', step_stats['num_batches'], prog_bar=False)

        return loss

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


    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms before optimizer step."""
        if self.global_step % 10 == 0:  # Log every 10 steps
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_val
            )
            self.log('train/grad_norm', grad_norm)

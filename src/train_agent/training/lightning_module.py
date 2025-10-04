"""
PyTorch Lightning module for GRPO training.
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional

from train_agent.model_schemas import GRPOConfig
from train_agent.training.grpo import compute_drgrpo_loss


class GRPOLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training language models with Group Relative Policy Optimization.

    GRPO computes policy gradients using advantages calculated relative to other rollouts
    in the same group, making it suitable for RL from AI feedback scenarios.
    """

    def __init__(self, grpo_config: GRPOConfig):
        """
        Initialize GRPO Lightning module.

        Args:
            grpo_config: GRPOConfig object containing all training parameters
        """
        super().__init__()
        self.grpo_config = grpo_config
        self.save_hyperparameters(grpo_config.model_dump())

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

        # Convert dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(grpo_config.torch_dtype, torch.bfloat16)

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            grpo_config.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=grpo_config.trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            grpo_config.model_name,
            trust_remote_code=grpo_config.trust_remote_code
        )

    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.logits

    def compute_sequence_logprobs(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute total log probability for each sequence in the batch.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] (optional)

        Returns:
            Total log probability per sequence [batch_size]
        """
        # Forward pass to get logits
        logits = self(input_ids, attention_mask)

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Extract log probs for actual tokens (shift to align predictions with targets)
        # Predictions at positions 0:N-1 predict tokens at positions 1:N
        new_logprobs = torch.gather(
            log_probs[:, :-1, :],  # predictions: exclude last position
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1)  # targets: exclude first position
        ).squeeze(-1).sum(dim=-1)  # sum over sequence to get total log prob

        return new_logprobs

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for GRPO.

        Expected batch format:
        {
            'input_ids': [batch_size, seq_len],
            'attention_mask': [batch_size, seq_len],
            'advantages': [batch_size],
            'old_logprobs': [batch_size],
        }
        """
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        advantages = batch['advantages']
        old_logprobs = batch['old_logprobs']

        #  Compute log probabilities for current policy
        new_logprobs = self.compute_sequence_logprobs(input_ids, attention_mask)

        # Compute GRPO loss
        loss = compute_drgrpo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
            clip_epsilon=self.clip_epsilon,
        )

        # Logging
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/mean_advantage', advantages.mean(), prog_bar=False)
        self.log('train/std_advantage', advantages.std(), prog_bar=False)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step (optional)."""
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        advantages = batch['advantages']
        old_logprobs = batch['old_logprobs']

        # Compute log probabilities for current policy
        new_logprobs = self.compute_sequence_logprobs(input_ids, attention_mask)

        # Compute GRPO loss
        loss = compute_drgrpo_loss(
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,
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

"""
PyTorch Lightning module for GRPO training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional


class GRPOLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training language models with Group Relative Policy Optimization.

    GRPO computes policy gradients using advantages calculated relative to other rollouts
    in the same group, making it suitable for RL from AI feedback scenarios.
    """

    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: int = 1000,
        gradient_clip_val: float = 1.0,
        advantage_type: str = "mean_normalized",
    ):
        """
        Initialize GRPO Lightning module.

        Args:
            model_name: HuggingFace model name or path
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for AdamW
            warmup_steps: Number of warmup steps for lr scheduler
            max_steps: Maximum training steps for lr scheduler
            gradient_clip_val: Maximum gradient norm for clipping
            advantage_type: Type of advantage calculation (from grpo.py)
        """
        super().__init__()
        self.save_hyperparameters()

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_clip_val = gradient_clip_val
        self.advantage_type = advantage_type

    def forward(self, input_ids, attention_mask=None):
        """Forward pass through the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.logits

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for GRPO.

        Expected batch format:
        {
            'input_ids': [batch_size, seq_len],
            'attention_mask': [batch_size, seq_len],
            'advantages': [batch_size],
        }
        """
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        advantages = batch['advantages']

        # Forward pass
        logits = self(input_ids, attention_mask)

        # Placeholder loss - to be implemented
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

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

        # Forward pass
        logits = self(input_ids, attention_mask)

        # Placeholder loss
        loss = torch.tensor(0.0, device=self.device)

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

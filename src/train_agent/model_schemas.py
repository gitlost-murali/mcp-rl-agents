"""
Pydantic BaseModel schemas for type-safe configuration objects.

This module contains all config classes with validation and type checking.
Use these classes when you need structured config objects with validation.
"""

from pydantic import BaseModel, Field
from typing import Optional

# Import base model name for defaults
from train_agent.config import BASE_MODEL


class TrainingConfig(BaseModel):
    """General training configuration (RL-agnostic)."""
    learning_rate: float = Field(default=1e-5, description="Learning rate for optimizer")
    weight_decay: float = Field(default=0.01, description="Weight decay for AdamW")
    warmup_steps: int = Field(default=100, description="Number of warmup steps for lr scheduler")
    num_epochs: int = Field(default=1, description="Number of epochs through the dataset")
    max_training_steps: int = Field(default=10, description="Maximum training steps (None for unlimited)")
    gradient_clip_val: float = Field(default=1.0, description="Maximum gradient norm for clipping")
    batch_size: int = Field(default=2, description="Batch size for training")

    @classmethod
    def from_config(cls) -> "TrainingConfig":
        """Create TrainingConfig from config.py TRAINING_CONFIG dict."""
        from train_agent.config import TRAINING_CONFIG
        return cls(**TRAINING_CONFIG)


class GRPOConfig(BaseModel):
    """Configuration for GRPO (Group Relative Policy Optimization) - RL-specific parameters."""
    model_name: str = Field(default=BASE_MODEL, description="HuggingFace model name or path")
    training_config: TrainingConfig = Field(default_factory=TrainingConfig, description="General training parameters")

    # GRPO-specific: rollout and group parameters
    num_training_inputs: int = Field(default=18, description="Number of training inputs to generate")
    rollouts_per_group: int = Field(default=4, description="Number of rollouts per input group for comparison")
    groups_per_step: int = Field(default=2, description="Number of input groups to process per training step")
    max_turns: int = Field(default=6, description="Maximum conversation turns per rollout")

    # GRPO-specific: advantage calculation
    advantage_type: str = Field(default="mean_normalized", description="Type of advantage calculation (mean_normalized, etc.)")
    clip_epsilon: float = Field(default=0.2, description="PPO clipping epsilon for policy ratio")

    # Model loading
    torch_dtype: str = Field(default="bfloat16", description="Model dtype: float16, bfloat16, float32")
    trust_remote_code: bool = Field(default=True, description="Trust remote code when loading model")

    @classmethod
    def from_config(cls) -> "GRPOConfig":
        """Create GRPOConfig from config.py GRPO_CONFIG and TRAINING_CONFIG dicts."""
        from train_agent.config import GRPO_CONFIG, BASE_MODEL
        return cls(
            model_name=BASE_MODEL,
            training_config=TrainingConfig.from_config(),
            num_training_inputs=GRPO_CONFIG["num_training_inputs"],
            rollouts_per_group=GRPO_CONFIG["rollouts_per_group"],
            groups_per_step=GRPO_CONFIG["groups_per_step"],
            max_turns=GRPO_CONFIG["max_turns"],
            advantage_type=GRPO_CONFIG["advantage_type"],
        )


class LoRAConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) configuration."""
    rank: int = Field(default=16, description="LoRA rank (r in the paper)")
    alpha: int = Field(default=32, description="LoRA alpha (scaling factor)")
    dropout: float = Field(default=0.05, description="Dropout probability")
    target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Which modules to apply LoRA to"
    )
    adapter_path: Optional[str] = Field(default=None, description="Path to saved LoRA adapter")
    merge_on_load: bool = Field(default=True, description="Whether to merge LoRA into base model before inference")

    @classmethod
    def from_config(cls) -> "LoRAConfig":
        """Create LoRAConfig from config.py LORA_CONFIG dict."""
        from train_agent.config import LORA_CONFIG
        return cls(**LORA_CONFIG)


class VLLMConfig(BaseModel):
    """vLLM inference engine configuration."""
    model_name: str = Field(default=BASE_MODEL, description="Model name or path")
    max_seq_length: int = Field(default=65536, description="Maximum sequence length")
    gpu_memory_utilization: float = Field(default=0.60, description="GPU memory usage (0.0-1.0)")
    tensor_parallel_size: int = Field(default=1, description="Number of GPUs for tensor parallelism")
    pipeline_parallel_size: int = Field(default=1, description="Number of GPUs for pipeline parallelism")
    dtype: str = Field(default="auto", description="Model dtype: 'auto', 'float16', 'bfloat16'")
    trust_remote_code: bool = Field(default=True, description="Trust remote code when loading model")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    @classmethod
    def from_config(cls) -> "VLLMConfig":
        """Create VLLMConfig from config.py INFERENCE_CONFIG dict."""
        from train_agent.config import INFERENCE_CONFIG, BASE_MODEL
        return cls(
            model_name=BASE_MODEL,
            max_seq_length=INFERENCE_CONFIG["max_seq_length"],
            gpu_memory_utilization=INFERENCE_CONFIG["gpu_memory_utilization"],
            tensor_parallel_size=INFERENCE_CONFIG["tensor_parallel_size"],
            pipeline_parallel_size=INFERENCE_CONFIG["pipeline_parallel_size"],
            dtype=INFERENCE_CONFIG["dtype"],
        )


class SamplingConfig(BaseModel):
    """Sampling configuration for generation."""
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Nucleus sampling top-p")
    top_k: int = Field(default=-1, description="Top-k sampling (-1 for disabled)")
    max_tokens: int = Field(default=8000, description="Maximum tokens to generate")
    stop: Optional[list[str]] = Field(default=None, description="Stop sequences")

    @classmethod
    def from_config(cls) -> "SamplingConfig":
        """Create SamplingConfig from config.py ROLLOUT_CONFIG dict."""
        from train_agent.config import ROLLOUT_CONFIG
        return cls(
            temperature=ROLLOUT_CONFIG["sampling_temperature"],
            top_p=ROLLOUT_CONFIG["sampling_top_p"],
            top_k=ROLLOUT_CONFIG["sampling_top_k"],
            max_tokens=ROLLOUT_CONFIG["sampling_max_tokens"],
            stop=ROLLOUT_CONFIG["sampling_stop"],
        )

class RolloutConfig(BaseModel):
    """Configuration for rollout generation during inference/evaluation."""
    max_turns: int = Field(default=10, description="Maximum conversation turns per rollout")
    sampling_temperature: float = Field(default=0.7, description="Temperature for rollout generation")

    @classmethod
    def from_config(cls) -> "RolloutConfig":
        """Create RolloutConfig from config.py ROLLOUT_CONFIG dict."""
        from train_agent.config import ROLLOUT_CONFIG
        return cls(**ROLLOUT_CONFIG)

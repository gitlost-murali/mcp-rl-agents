# @title Advanced Settings
"""
Simple configuration values and dictionaries for the training pipeline.

For type-safe config objects with validation, see model_schemas.py
"""

# Project configuration
MODEL_NAME = "mcprl-3b-exa"  # Name for your trained model
PROJECT_NAME = "mcp-rl"  # Project name for tracking
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # Base model for both training and inference

# General training configuration (RL-agnostic)
TRAINING_CONFIG = {
    "learning_rate": 1e-5,  # Learning rate for optimizer
    "weight_decay": 0.01,  # Weight decay for AdamW
    "warmup_steps": 100,  # Number of warmup steps for lr scheduler
    "num_epochs": 1,  # Number of epochs through the dataset
    "max_training_steps": 4,  # Maximum training steps
    "gradient_clip_val": 1.0,  # Maximum gradient norm for clipping
}

# GRPO-specific configuration (RL training)
GRPO_CONFIG = {
    "num_training_inputs": 18,  # Number of training inputs to generate
    "rollouts_per_group": 4,  # Number of rollouts per input group for comparison
    "groups_per_step": 2,  # Number of input groups to process per training step
    "max_turns": 6,  # Maximum conversation turns per rollout (T4 optimized)
    "advantage_type": "mean_normalized",  # Type of advantage calculation
}

# Rollout generation configuration (for inference/evaluation)
ROLLOUT_CONFIG = {
    "max_turns": 5,  # Maximum number of turns for rollout generation
    "sampling_temperature": 0.7,  # Temperature for rollout generation
}

# Dataset configuration
NUM_TEST_INPUTS = 8  # Number of test inputs to generate
RULER_MODEL = "openrouter/openai/o4-mini"  # Model for RULER evaluation
INPUT_GENERATION_MODEL = "openai/o3"  # Model for generating training inputs
DATASET_FILENAME = "data/dataset.json"  # Path to dataset file

# LoRA configuration
LORA_CONFIG = {
    "rank": 16,  # LoRA rank (r in the paper)
    "alpha": 32,  # LoRA alpha (scaling factor)
    "dropout": 0.05,  # Dropout probability
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Which modules to apply LoRA to
    "adapter_path": None,  # Path to saved LoRA adapter (set after training)
    "merge_on_load": True,  # Whether to merge LoRA into base model before inference
}

# Inference configuration (vLLM)
INFERENCE_CONFIG = {
    "max_seq_length": 65536,  # Maximum sequence length for inference
    "gpu_memory_utilization": 0.6,  # GPU memory usage (0.0-1.0)
    "tensor_parallel_size": 1,  # Number of GPUs for tensor parallelism
    "pipeline_parallel_size": 1,  # Number of GPUs for pipeline parallelism
    "dtype": "auto",  # Model dtype: "auto", "float16", "bfloat16"
}

# Convenience constants (for backward compatibility)
MAX_SEQ_LENGTH = INFERENCE_CONFIG["max_seq_length"]
GPU_MEMORY_UTILIZATION = INFERENCE_CONFIG["gpu_memory_utilization"]
MAX_TURNS = ROLLOUT_CONFIG["max_turns"]

# Paths for model checkpoints
CHECKPOINT_DIR = "./checkpoints"  # Directory for saving LoRA checkpoints
MERGED_MODEL_DIR = "./merged_models"  # Directory for merged models

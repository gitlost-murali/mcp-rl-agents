# @title Advanced Settings

# Model configuration
MODEL_NAME = "mcprl-3b-exa"  # Name for your trained model
PROJECT_NAME = "mcp-rl"  # Project name for tracking

# Training configuration
TRAINING_CONFIG = {
    "num_training_inputs": 18,  # Number of training inputs to generate
    "groups_per_step": 2,  # Inputs to process per training step
    "num_epochs": 1,  # Number of times through all data
    "rollouts_per_group": 4,  # Different responses per input (for RULER comparison)
    "learning_rate": 1e-5,  # Learning rate
    "max_training_steps": None,  # Maximum training steps (set to None for no limit)
}

MAX_TURNS = 10  # Maximum number of turns for the model to generate during one rollout

NUM_TEST_INPUTS = 8  # Number of test inputs to generate
RULER_MODEL = "openrouter/openai/o4-mini"  # Model for RULER evaluation
INPUT_GENERATION_MODEL = "openai/o3"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Colab/T4 specific config to avoid OOM errors
MAX_TURNS = 6  # Decrease the number of turns to avoid OOM errors on a T4
MAX_SEQ_LENGTH = 65536  # Maximum sequence length
GPU_MEMORY_UTILIZATION = 0.85  # GPU memory usage (0.0-1.0)

DATASET_FILENAME = "data/dataset.json"

# LoRA Configuration
LORA_CONFIG = {
    "rank": 16,  # LoRA rank (r in the paper)
    "alpha": 32,  # LoRA alpha (scaling factor)
    "dropout": 0.05,  # Dropout probability
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],  # Which modules to apply LoRA to
    "adapter_path": None,  # Path to saved LoRA adapter (set after training)
    "merge_on_load": True,  # Whether to merge LoRA into base model before inference
}

# Inference Configuration (vLLM)
INFERENCE_CONFIG = {
    "tensor_parallel_size": 1,  # Number of GPUs for tensor parallelism
    "pipeline_parallel_size": 1,  # Number of GPUs for pipeline parallelism
    "dtype": "auto",  # Model dtype: "auto", "float16", "bfloat16"
}

# Paths for model checkpoints
CHECKPOINT_DIR = "./checkpoints"  # Directory for saving LoRA checkpoints
MERGED_MODEL_DIR = "./merged_models"  # Directory for merged models

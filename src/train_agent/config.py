# @title Advanced Settings

# Model configuration
MODEL_NAME = "mcprl-3b-exa"  # Name for your trained model
PROJECT_NAME = "mcp-rl"  # Project name for tracking

# Training configuration
TRAINING_CONFIG = {
    "num_training_inputs": 16,  # Number of training inputs to generate
    "groups_per_step": 2,  # Inputs to process per training step
    "num_epochs": 1,  # Number of times through all data
    "rollouts_per_group": 4,  # Different responses per input (for RULER comparison)
    "learning_rate": 1e-5,  # Learning rate
    "max_training_steps": None,  # Maximum training steps (set to None for no limit)
}

MAX_TURNS = 10  # Maximum number of turns for the model to generate during one rollout

NUM_TEST_INPUTS = 8  # Number of test inputs to generate
RULER_MODEL = "openrouter/openai/o4-mini"  # Model for RULER evaluation
INPUT_GENERATION_MODEL = "openai/o4-mini"

# Colab/T4 specific config to avoid OOM errors
MAX_TURNS = 3  # Decrease the number of turns to avoid OOM errors on a T4
MAX_SEQ_LENGTH = 16384  # Maximum sequence length
GPU_MEMORY_UTILIZATION = 0.7  # GPU memory usage (0.0-1.0)
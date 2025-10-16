# @title Run this cell to train your model!

import asyncio
import json
import os
import random
from typing import Optional

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import FSDPStrategy
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from train_agent.config import (
    DATASET_FILENAME,
    MODEL_NAME,
    PROJECT_NAME,
    BASE_MODEL,
)
from train_agent.data.dataset_generator import load_train_and_val_scenarios
from train_agent.data.lightning_data_module import GRPORolloutDataModule
from train_agent.model_schemas import GRPOConfig, SamplingConfig, VLLMConfig
from train_agent.training.lightning_module import GRPOLightningModule
from train_agent.training.trajectory import McpScenario, rollout
from train_agent.inference.vllm_engine import VLLMEngine
from train_agent.utils.settings import settings

# Optional
if settings.wandb_api_key:
    os.environ["WANDB_API_KEY"] = settings.wandb_api_key
else:
    print("WANDB_API_KEY is not set. We'll skip logging metrics to Weights & Biases.")

if settings.openrouter_key:
    os.environ["OPENROUTER_API_KEY"] = settings.openrouter_key
else:
    raise ValueError("OPENROUTER_KEY is not set. Please set it in the .env file.")

random.seed(42)


class ModelTrainer:
    def __init__(self, grpo_config: Optional[GRPOConfig]):
        """Initialize ModelTrainer with merged Lightning module.

        Args:
            grpo_config: GRPO configuration. If None, creates default from TRAINING_CONFIG.
        """
        # Create GRPO config from config.py if not provided
        if grpo_config is None:
            grpo_config = GRPOConfig.from_config()

        self.grpo_config = grpo_config

        print("ModelTrainer initialized!")
        print(f"Base model: {BASE_MODEL}")
        print(f"Model name: {MODEL_NAME}")
        print(f"Project: {PROJECT_NAME}")
        print(f"Rollouts per group: {grpo_config.rollouts_per_group}")
        print(f"Groups per step: {grpo_config.groups_per_step}")

    def create_scenarios(self, raw_train_scenarios: list[dict]) -> list[McpScenario]:
        """Convert raw scenario dicts to McpScenario objects."""
        return [
            McpScenario(
                task_description=scenario["task"],
                max_turns=self.grpo_config.max_turns,
                scenario_id=scenario.get("id", scenario["task"][:50])
            )
            for scenario in raw_train_scenarios
        ]

    def train(self, raw_train_scenarios: list[dict]):
        """Train the model using online GRPO with Lightning.

        The Lightning module now handles:
        - Online rollout collection (fresh data every step)
        - vLLM server start/stop for GPU memory management
        - GRPO loss computation with advantages
        - All logging and checkpointing
        """
        print(
            f"\nStarting online GRPO training with config:\n"
            f"  - Max turns: {self.grpo_config.max_turns}\n"
            f"  - Rollouts per group: {self.grpo_config.rollouts_per_group}\n"
            f"  - Groups per step: {self.grpo_config.groups_per_step}\n"
            f"  - Max training steps: {self.grpo_config.training_config.max_training_steps}\n"
            f"  - Learning rate: {self.grpo_config.training_config.learning_rate}\n"
        )

        # Convert raw scenarios to McpScenario objects
        train_scenarios = self.create_scenarios(raw_train_scenarios)
        print(f"Loaded {len(train_scenarios)} training scenarios\n")

        # Initialize Lightning module
        lightning_module = GRPOLightningModule(
            grpo_config=self.grpo_config,
        )

        # Configure FSDP: model loaded once, weights sharded across GPUs
        strategy = FSDPStrategy(
            sharding_strategy="FULL_SHARD",
            state_dict_type="full",
        )

        # Set CUDA_VISIBLE_DEVICES for PyTorch Lightning to use GPUs 1, 2, 3
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        print(f"Lightning CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

        # Set up Lightning Trainer
        trainer = pl.Trainer(
            max_steps=self.grpo_config.training_config.max_training_steps,
            callbacks=[
                ModelCheckpoint(
                    dirpath="./checkpoints",
                    filename="grpo-{step:06d}",
                    save_top_k=3,
                    monitor="train/loss"
                ),
                LearningRateMonitor(logging_interval="step"),
            ],
            gradient_clip_val=self.grpo_config.training_config.gradient_clip_val,
            precision="bf16-mixed" if self.grpo_config.torch_dtype == "bfloat16" else "16-mixed",
            accelerator="gpu",
            devices=[0, 1, 2],  # Use 3 GPUs (devices 0, 1, 2 from CUDA_VISIBLE_DEVICES)
            strategy=strategy,
            enable_progress_bar=True,
            log_every_n_steps=1,
        )

        # Start training - Lightning module handles everything!
        print(f"{'='*80}")
        print("Starting online GRPO training...")
        print("Each step collects fresh rollouts from current policy")
        print(f"{'='*80}\n")

        data_module = GRPORolloutDataModule(
            grpo_config=self.grpo_config,
            train_scenarios=train_scenarios,
            vllm_config=VLLMConfig.from_config(),
            sampling_config=SamplingConfig.from_config(),
        )

        trainer.fit(lightning_module, data_module)

        print("\n=== Training completed ===")

    async def test(self, raw_val_scenarios: list[dict], checkpoint_path: Optional[str] = None):
        """Test the trained model on validation scenarios.

        Args:
            raw_val_scenarios: List of raw scenario dictionaries
            checkpoint_path: Path to model checkpoint to test (optional)
        """
        print("Generating test inputs...")
        val_scenarios = self.create_scenarios(raw_val_scenarios)

        print(f"\n🧪 Testing the trained model on {len(val_scenarios)} new inputs:\n")
        print("=" * 80)

        # Determine which model to use
        model_name = checkpoint_path or BASE_MODEL

        # Start vLLM server for testing
        print(f"\nStarting vLLM server with model: {model_name}")
        vllm_config = VLLMConfig.from_config()
        vllm_config.model_name = model_name

        vllm_engine = VLLMEngine(vllm_config)
        vllm_engine.start_server(port=8011, host="0.0.0.0")

        try:
            # Initialize inference client
            inference_client = AsyncOpenAI(
                base_url="http://localhost:8011/v1",
                api_key="EMPTY"
            )

            sampling_config = SamplingConfig.from_config()

            # Create temporary tokenizer for testing
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.grpo_config.model_name,
                trust_remote_code=self.grpo_config.trust_remote_code
            )

            for i, scenario in enumerate(val_scenarios[:3]):
                print(f"\nTest {i + 1}:")
                print(f"Input: {scenario.task_description}")

                # Run the model
                result_trajectory = await rollout(
                    inference_client=inference_client,
                    model_name=model_name,
                    scenario=scenario,
                    sampling_config=sampling_config,
                    debug=True,
                    mcp_url=settings.mcp_url,
                    tokenizer=tokenizer,
                )

                # Extract the model's response
                messages = result_trajectory.messages
                model_response = messages[-1]["content"] if messages else "No response"

                with open(f"debugging_outputs/model_response_{i}.json", "w") as f:
                    json.dump(messages, f, indent=4)

                print(f"Model output: {model_response}")
                print(f"Task completed: {result_trajectory.metrics.get('task_completed', False)}")
                print(f"Number of turns: {result_trajectory.metrics.get('num_turns', 0)}")
                print("-" * 80)

            print("\n🎉 Testing completed!")
            print(f"\nYour model '{MODEL_NAME}' has been trained to use the MCP server at:")
            print(settings.mcp_url)
            print("\nTo use this model in production:")
            print("1. The model checkpoint is saved in ./checkpoints/")
            print("2. You can load it using the Lightning or transformers library")
            print("3. Or continue training with more examples by adjusting the configuration")

        finally:
            # Always stop the vLLM server, even if testing fails
            print("\nStopping vLLM server...")
            vllm_engine.stop_server()
    

if __name__ == "__main__":
    trainer = ModelTrainer(GRPOConfig.from_config())

    # Load training and validation scenarios
    raw_train_scenarios, raw_val_scenarios = load_train_and_val_scenarios(DATASET_FILENAME)

    raw_train_scenarios = [
        {
            "task": "What is the 3rd movie of Jr. NTR?",
            "difficulty": 3
        }
    ]
    # Train the model
    trainer.train(raw_train_scenarios)

    # Optionally test after training
    # asyncio.run(trainer.test(raw_val_scenarios))
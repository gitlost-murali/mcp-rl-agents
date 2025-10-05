# @title Run this cell to train your model!

import asyncio
import json
import os
import random
from dataclasses import dataclass
import traceback
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from train_agent.data.dataset_generator import load_train_and_val_scenarios
from openai import AsyncOpenAI

from train_agent.config import (
    DATASET_FILENAME,
    MODEL_NAME,
    PROJECT_NAME,
    MAX_SEQ_LENGTH,
    GPU_MEMORY_UTILIZATION,
    MAX_TURNS,
    TRAINING_CONFIG,
    RULER_MODEL,
    BASE_MODEL,
)
from train_agent.inference.vllm_engine import VLLMEngine
from train_agent.model_schemas import GRPOConfig, SamplingConfig, VLLMConfig
from train_agent.training.lightning_module import GRPOLightningModule
from train_agent.training.trajectory import (
    Trajectory,
    TrajectoryGroup,
    McpScenario,
    rollout,
    gather_trajectory_groups,
)
from train_agent.utils.mcp_utils import (
    call_mcp_tool,
    get_content_text,
    get_tool_schemas_from_mcp_with_complete_task_tool,
)
from train_agent.utils.settings import settings
from train_agent.utils.debug_utils import log, log_json

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from train_agent.training.grpo import calculate_group_advantages, create_group_rollout_from_trajectories

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
        """Initialize ModelTrainer with Lightning module and inference client.

        Args:
            grpo_config: GRPO configuration. If None, creates default from TRAINING_CONFIG.
        """
        # Create GRPO config from legacy TRAINING_CONFIG if not provided
        if grpo_config is None:
            grpo_config = GRPOConfig(
                model_name=BASE_MODEL,
                rollouts_per_group=TRAINING_CONFIG.get("rollouts_per_group", 4),
                groups_per_step=TRAINING_CONFIG.get("groups_per_step", 2),
                max_turns=MAX_TURNS,
            )
            grpo_config.training_config.learning_rate = TRAINING_CONFIG.get("learning_rate", 1e-5)
            grpo_config.training_config.num_epochs = TRAINING_CONFIG.get("num_epochs", 1)

        self.grpo_config = grpo_config

        # Initialize Lightning module
        self.lightning_module = GRPOLightningModule(grpo_config)

        # Initialize inference client for rollouts (vLLM server via OpenAI client)
        # This expects a vLLM server running on localhost:8000 by default
        self.inference_client = AsyncOpenAI(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY"  # vLLM doesn't need a real API key
        )

        # Sampling config for rollouts
        self.sampling_config = SamplingConfig(temperature=0.7, top_p=0.9, max_tokens=8000)

        print("ModelTrainer initialized!")
        print(f"Base model: {BASE_MODEL}")
        print(f"Model name: {MODEL_NAME}")
        print(f"Project: {PROJECT_NAME}")
        print(f"Rollouts per group: {grpo_config.rollouts_per_group}")
        print(f"Groups per step: {grpo_config.groups_per_step}")

    def create_scenarios(self, raw_train_scenarios: list[dict]):
        """Convert raw scenario dicts to McpScenario objects."""
        return [
            McpScenario(
                task_description=scenario["task"],
                max_turns=MAX_TURNS,
                scenario_id=scenario.get("id", scenario["task"][:50])
            )
            for scenario in raw_train_scenarios
        ]

    async def train(self, raw_train_scenarios: list[dict]):
        """Train the model using GRPO with Lightning."""
        print(
            f"Using config: max_turns={MAX_TURNS}, "
            f"rollouts_per_group={self.grpo_config.rollouts_per_group}, "
            f"groups_per_step={self.grpo_config.groups_per_step}, "
            f"num_epochs={self.grpo_config.training_config.num_epochs}, "
            f"learning_rate={self.grpo_config.training_config.learning_rate}"
        )

        train_scenarios = self.create_scenarios(raw_train_scenarios)

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
            accelerator="auto",
            devices="auto",
        )

        # Main training loop
        for epoch in range(self.grpo_config.training_config.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.grpo_config.training_config.num_epochs} ===")

            # Process scenarios in batches (groups_per_step scenarios at a time)
            for batch_idx in range(0, len(train_scenarios), self.grpo_config.groups_per_step):
                batch_scenarios = train_scenarios[batch_idx:batch_idx + self.grpo_config.groups_per_step]

                print(f"\nGathering trajectory groups for batch {batch_idx // self.grpo_config.groups_per_step + 1}...")

                # Gather trajectories for this batch
                trajectory_groups = await gather_trajectory_groups(
                    inference_client=self.inference_client,
                    model_name=BASE_MODEL,  # TODO: Use current checkpoint model name
                    scenarios=batch_scenarios,
                    rollouts_per_group=self.grpo_config.rollouts_per_group,
                    sampling_config=self.sampling_config,
                    debug=True,
                    mcp_url=settings.mcp_url,
                )

                # TODO: Implement trajectory scoring/judging using RULER or other judge model
                # For now, using a placeholder that assigns rewards based on task completion
                print("Scoring trajectories...")
                for group in trajectory_groups:
                    for traj in group.trajectories:
                        # Simple placeholder: reward 1.0 if completed, 0.0 otherwise
                        traj.reward = 1.0 if traj.metrics.get("task_completed", False) else 0.0

                # Calculate advantages for each group
                print("Calculating advantages...")

                training_batches = []
                for group in trajectory_groups:
                    # Create GroupRollout for advantage calculation
                    group_rollout = create_group_rollout_from_trajectories(
                        scenario_id=group.scenario_id,
                        trajectories=group.trajectories,
                        rewards=group.rewards,
                    )

                    # Calculate advantages
                    advantages = calculate_group_advantages(
                        group_rollout,
                        advantage_type=self.grpo_config.advantage_type
                    )

                    # TODO: Prepare batches for Lightning training
                    # Need to tokenize trajectories and compute old_logprobs
                    # This requires running inference to get logprobs for each trajectory
                    print(f"  Group {group.scenario_id}: rewards={group.rewards}, advantages={advantages}")

                # TODO: Create DataLoader and run trainer.fit()
                # This requires implementing a custom Dataset/DataLoader that yields batches
                # with the format expected by GRPOLightningModule.training_step()
                print("Training step not yet fully implemented - need to tokenize and compute old_logprobs")

        print("\n=== Training completed ===")

    async def test(self, raw_val_scenarios: list[dict]):
        """Test the trained model on validation scenarios."""
        print("Generating test inputs...")
        val_scenarios = [
            McpScenario(
                task_description=scenario["task"],
                max_turns=MAX_TURNS,
                scenario_id=scenario.get("id", scenario["task"][:50])
            )
            for scenario in raw_val_scenarios
        ]

        print(f"\n🧪 Testing the trained model on {len(val_scenarios)} new inputs:\n")
        print("=" * 80)

        for i, scenario in enumerate(val_scenarios[:3]):
            print(f"\nTest {i + 1}:")
            print(f"Input: {scenario.task_description}")

            # Run the model
            result_trajectory = await rollout(
                inference_client=self.inference_client,
                model_name=BASE_MODEL,  # TODO: Use checkpoint model name
                scenario=scenario,
                sampling_config=self.sampling_config,
                debug=True,
                mcp_url=settings.mcp_url,
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
    

if __name__ == "__main__":
    # engine = VLLMEngine(VLLMConfig())
    # engine.start_server()
    # engine._init_client()
    trainer = ModelTrainer(GRPOConfig())

    # asyncio.run(trainer.train(raw_train_scenarios))
    raw_val_scenarios = load_train_and_val_scenarios(DATASET_FILENAME)[1]
    # raw_val_scenarios = [{"task": "what is taylor swift's most streamed song on spotify?", "difficulty": 3}]
    asyncio.run(trainer.test(raw_val_scenarios))
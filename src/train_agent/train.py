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
from train_agent.training.batch_preparation import (
    prepare_batches_from_trajectory_groups,
    validate_trajectory_alignment,
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
        # Create GRPO config from config.py if not provided
        if grpo_config is None:
            grpo_config = GRPOConfig.from_config()

        self.grpo_config = grpo_config

        # Initialize Lightning module
        self.lightning_module = GRPOLightningModule(grpo_config)

        # Initialize inference client for rollouts (vLLM server via OpenAI client)
        # This expects a vLLM server running on localhost:8000 by default

        # To start vLLM server with config.py values:
        # engine = VLLMEngine(VLLMConfig.from_config())
        # engine.start_server()
        # engine._init_client()
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
                max_turns=self.grpo_config.max_turns,
                scenario_id=scenario.get("id", scenario["task"][:50])
            )
            for scenario in raw_train_scenarios
        ]

    async def train(self, raw_train_scenarios: list[dict]):
        """Train the model using GRPO with Lightning."""
        print(
            f"Using config: max_turns={self.grpo_config.max_turns}, "
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
                    tokenizer=self.lightning_module.tokenizer,  # Pass tokenizer for position tracking
                )

                print("=" * 80)
                print(f"Trajectory groups collected: {trajectory_groups[0].trajectories[0].messages[-1]}")
                print("=" * 80)

                # Validate alignment for first trajectory in each group (sanity check)
                print("Validating trajectory alignment...")
                for group_idx, group in enumerate(trajectory_groups):
                    if group.trajectories:
                        is_valid = validate_trajectory_alignment(
                            group.trajectories[0],
                            self.lightning_module.tokenizer,
                            verbose=(group_idx == 0),  # Verbose for first group only
                        )
                        if not is_valid:
                            raise ValueError(f"Trajectory alignment validation failed for group {group_idx}")

                # TODO: Implement trajectory scoring/judging using RULER or other judge model
                # For now, using a placeholder that assigns rewards based on task completion
                print("Scoring trajectories...")
                for group in trajectory_groups:
                    for traj in group.trajectories:
                        # Simple placeholder: reward 1.0 if completed, 0.0 otherwise
                        traj.reward = 1.0 if traj.metrics.get("task_completed", False) else 0.0

                # Calculate advantages for each group
                print("Calculating advantages...")

                advantages_list = []
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
                    advantages_list.append(advantages)

                    print(f"  Group {group.scenario_id}: rewards={group.rewards}, advantages={advantages}")

                # Prepare training batches
                print("Preparing training batches...")
                train_dataloader = prepare_batches_from_trajectory_groups(
                    trajectory_groups=trajectory_groups,
                    advantages_list=advantages_list,
                    tokenizer=self.lightning_module.tokenizer,
                    batch_size=2,  # Small batch size for GPU memory
                    max_length=8192,
                )

                # Run training step
                print(f"Running training on {len(train_dataloader)} batches...")
                trainer.fit(
                    self.lightning_module,
                    train_dataloaders=train_dataloader,
                )

        print("\n=== Training completed ===")

    async def test(self, raw_val_scenarios: list[dict]):
        """Test the trained model on validation scenarios."""
        print("Generating test inputs...")
        val_scenarios = [
            McpScenario(
                task_description=scenario["task"],
                max_turns=self.grpo_config.max_turns,
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
                tokenizer=self.lightning_module.tokenizer,
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
    # engine = VLLMEngine(VLLMConfig.from_config())
    # engine.start_server()
    # engine._init_client()
    trainer = ModelTrainer(GRPOConfig.from_config())

    # asyncio.run(trainer.train(raw_train_scenarios))
    raw_val_scenarios = load_train_and_val_scenarios(DATASET_FILENAME)[1]
    # raw_val_scenarios = [{"task": "what is taylor swift's most streamed song on spotify?", "difficulty": 3}]
    asyncio.run(trainer.test(raw_val_scenarios))
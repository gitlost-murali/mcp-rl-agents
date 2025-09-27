# @title Run this cell to train your model!

import asyncio
import json
import os
import random
from dataclasses import dataclass
import traceback

import weave
from openai import AsyncOpenAI

import art
from art.local import LocalBackend
from art.rewards import ruler_score_group
from art.utils import iterate_dataset


from train_agent.config import (
    MODEL_NAME,
    PROJECT_NAME,
    MAX_SEQ_LENGTH,
    GPU_MEMORY_UTILIZATION,
    MAX_TURNS,
    TRAINING_CONFIG,
    RULER_MODEL,
    BASE_MODEL,
)
from train_agent.utils.mcp_utils import (
    call_mcp_tool,
    get_content_text,
    get_tool_schemas_from_mcp_with_complete_task_tool,
)
from train_agent.utils.settings import settings
from train_agent.utils.debug_utils import log, log_json

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

# Optional
if settings.wandb_api_key:
    os.environ["WANDB_API_KEY"] = settings.wandb_api_key
    weave.init(PROJECT_NAME)
else:
    print("WANDB_API_KEY is not set. We'll skip logging metrics to Weights & Biases.")

random.seed(42)


@dataclass
class McpScenario:
    """A scenario for MCP agent evaluation against a remote MCP server."""

    task_description: str
    max_turns: int = MAX_TURNS


class ModelTrainer:
    def __init__(self):
        self.model = art.TrainableModel(
            name=MODEL_NAME,
            project=PROJECT_NAME,
            base_model=BASE_MODEL,
        )

        # To run on a T4, we need to override some config defaults.
        self.model._internal_config = art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(
                max_seq_length=MAX_SEQ_LENGTH,
            ),
            engine_args=art.dev.EngineArgs(
                enforce_eager=True,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            ),
        )

        # Initialize the server
        self.backend = LocalBackend(
            in_process=True,
            path="./.art",
        )

        asyncio.run(self.register_model())
        print("Model created!")
        print("Base model:", BASE_MODEL)
        print("Model name:", MODEL_NAME)
        print("Project name:", PROJECT_NAME)

    async def register_model(self):
        # Register the model with the local Backend
        await self.model.register(self.backend)

    def create_scenarios(self, raw_train_scenarios: list[dict]):
        return [
            McpScenario(
                task_description=scenario["task"],
                max_turns=MAX_TURNS,
            )
            for scenario in raw_train_scenarios
        ]

    async def create_iterator(self, train_scenarios: list[McpScenario]):
        return iterate_dataset(
            train_scenarios,
            groups_per_step=TRAINING_CONFIG["groups_per_step"],
            num_epochs=TRAINING_CONFIG["num_epochs"],
            initial_step=await self.model.get_step(),  # Resume from checkpoint
        )

    async def train(self, raw_train_scenarios: list[dict]):
        print(
            f"Using config: max_turns={MAX_TURNS}, rollouts_per_group={TRAINING_CONFIG['rollouts_per_group']}, "
            f"groups_per_step={TRAINING_CONFIG['groups_per_step']}, num_epochs={TRAINING_CONFIG['num_epochs']}, "
            f"learning_rate={TRAINING_CONFIG['learning_rate']}"
        )

        await self.model.register(self.backend)

        train_scenarios = self.create_scenarios(raw_train_scenarios)
        train_iterator = await self.create_iterator(train_scenarios)

        # Main training loop using iterate_dataset
        for batch in train_iterator:
            print("Gathering trajectory groups with RULER scoring...")

            # Use gather_trajectory_groups with ruler_score_group
            groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        rollout(self.model, scenario, False)
                        for _ in range(TRAINING_CONFIG["rollouts_per_group"])
                    )
                    for scenario in batch.items
                ),
                pbar_desc=f"train gather step {batch.step}",
            )

            scored_groups = []
            for group in groups:
                # Use RULER to assign relative scores to each trajectory
                judged_group = await ruler_score_group(
                    group, judge_model=RULER_MODEL, debug=True, swallow_exceptions=True
                )
                scored_groups.append(judged_group)

            print("starting train")
            await self.model.train(
                scored_groups,
                config=art.TrainConfig(learning_rate=TRAINING_CONFIG["learning_rate"]),
            )


@weave.op()
async def rollout(
    model: art.Model,
    scenario: McpScenario,
    debug: bool = False,
    mcp_url: str = settings.mcp_url,
) -> art.Trajectory:
    """Run an MCP agent rollout against the remote MCP server."""
    traj = art.Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"task": scenario.task_description},
        metrics={
            "task_completed": False,
            "success": False,
            "ran_out_of_turns": False,
        },
        scenario=scenario,
    )

    tool_schemas: list[
        ChatCompletionToolParam
    ] = await get_tool_schemas_from_mcp_with_complete_task_tool(mcp_url)
    traj.tools = tool_schemas

    # Initialize conversation
    system_prompt = (
        f"You are an MCP (Model Context Protocol) agent.\n\n"
        f"Use MCP tools through the server to complete your task.\n\n"
        f"When you believe you have completed the task, call the 'complete_task' function with a summary of what you accomplished. "
        f"You have a total of {scenario.max_turns} turns."
        # NOTE: removing 'Only use tool calls, do not write any content.' — some models
        # will freeze if they think plain text is disallowed. Let them output thoughts but
        # we only process tool calls below.
    )

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Please complete this task: {scenario.task_description}",
        },
    ]

    num_turns = 0
    task_completed = False

    # Main interaction loop
    while num_turns < scenario.max_turns and not task_completed:
        num_turns += 1

        try:
            # === Log request ===
            last_user = next(
                (m for m in reversed(traj.messages()) if m["role"] == "user"), None
            )
            log(
                "LLM request",
                step=num_turns,
                model=(model.inference_model_name or model.name),
                tools=len(tool_schemas),
                last_user=(last_user["content"][:160] + "..." if last_user else None),
            )

            # Get LLM response
            async with traj.track_duration("llm_completion"):
                openai_client = AsyncOpenAI(
                    api_key=model.inference_api_key,
                    base_url=model.inference_base_url,
                )

                # We also log the request body (without huge params)
                req_preview = {
                    "model": model.inference_model_name
                    if model.inference_model_name
                    else model.name,
                    "messages_len": len(traj.messages()),
                    "tools_len": len(tool_schemas),
                }
                log_json("LLM request (preview)", req_preview)

                response = await openai_client.chat.completions.create(
                    model=model.inference_model_name
                    if model.inference_model_name
                    else model.name,
                    messages=traj.messages(),
                    tools=tool_schemas,
                    max_completion_tokens=8000,
                )

            # === Log response ===
            choice = response.choices[0]

            finish_reason = getattr(choice, "finish_reason", None)
            msg = choice.message
            has_tools = bool(getattr(msg, "tool_calls", None))
            content_preview = (
                (msg.content[:200] + "...")
                if isinstance(msg.content, str) and msg.content
                else str(msg.content)[:200]
            )
            log(
                "LLM response parsed",
                finish_reason=finish_reason,
                has_tool_calls=has_tools,
                content_preview=content_preview,
            )

            traj.messages_and_choices.append(choice)

            # Handle tool calls
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    try:
                        log(
                            "Tool call received",
                            name=tool_call.function.name,
                            raw_args=tool_call.function.arguments,
                        )
                        tool_args = json.loads(tool_call.function.arguments or "{}")

                        if tool_call.function.name == "complete_task":
                            traj.metrics["task_completed"] = True
                            task_completed = True
                            traj.logs.append(
                                f"Task completion attempted with summary: {tool_args.get('summary', '')}"
                            )
                            # We still append a tool message for completeness
                            traj.messages_and_choices.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": "Task marked complete.",
                                }
                            )
                        else:
                            # 🔧 Call MCP tool through remote Smithery session
                            result = await call_mcp_tool(
                                tool_call.function.name, tool_args, mcp_url
                            )

                            content_text = get_content_text(result)
                            log(
                                "Tool result",
                                name=tool_call.function.name,
                                len=len(content_text),
                            )

                            if len(content_text) > 20000:
                                # print(
                                #     f"Tool call result for {tool_call.function.name} is too long: {len(content_text)}"
                                # )
                                # print(f"Args: {tool_args}")
                                # print(content_text[:1000])
                                # print(content_text[-1000:])
                                raise Exception(
                                    f"Tool call result for {tool_call.function.name} is too long: {len(content_text)}"
                                )

                            # Add tool response
                            traj.messages_and_choices.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": content_text,
                                }
                            )

                    except Exception as e:
                        traceback.print_exc()
                        traj.logs.append(f"Tool call error: {e}")

                        # Add error response
                        traj.messages_and_choices.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error: {str(e)}",
                            }
                        )
            else:
                # No tool calls — log and continue (RULER will likely give 0)
                log(
                    "LLM returned no tool_calls; skipping tool execution",
                    turn=num_turns,
                )
                # You can consider breaking here or letting it try another turn
                # break

        except Exception as e:
            traceback.print_exc()
            traj.logs.append(f"Error in turn {num_turns}: {e}")
            break

    if not task_completed and num_turns == scenario.max_turns:
        traj.metrics["ran_out_of_turns"] = True

    traj.metrics["num_turns"] = num_turns

    return traj.finish()

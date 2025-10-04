"""Trajectory data structures and rollout function for MCP agent training."""

import asyncio
import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from train_agent.model_schemas import SamplingConfig
from train_agent.utils.mcp_utils import (
    call_mcp_tool,
    get_content_text,
    get_tool_schemas_from_mcp_with_complete_task_tool,
)
from train_agent.utils.settings import settings
from train_agent.utils.debug_utils import log


@dataclass
class Trajectory:
    messages: List[Dict[str, Any]]
    reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    tools: List[ChatCompletionToolParam] = field(default_factory=list)


@dataclass
class TrajectoryGroup:
    scenario_id: str
    trajectories: List[Trajectory]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def rewards(self) -> List[float]:
        return [t.reward for t in self.trajectories]


@dataclass
class McpScenario:
    task_description: str
    max_turns: int = 10
    scenario_id: Optional[str] = None


async def rollout(
    inference_client: AsyncOpenAI,
    model_name: str,
    scenario: McpScenario,
    sampling_config: Optional[SamplingConfig] = None,
    debug: bool = False,
    mcp_url: Optional[str] = None,
) -> Trajectory:
    """Run MCP agent rollout. Expects OpenAI-compatible inference client (vLLM server, etc)."""

    mcp_url = mcp_url or settings.mcp_url
    sampling_config = sampling_config or SamplingConfig()

    traj = Trajectory(
        messages=[],
        metadata={"task": scenario.task_description},
        metrics={"task_completed": False, "ran_out_of_turns": False, "num_turns": 0},
    )

    tool_schemas = await get_tool_schemas_from_mcp_with_complete_task_tool(mcp_url)
    traj.tools = tool_schemas

    system_prompt = (
        f"You are an MCP (Model Context Protocol) agent.\n\n"
        f"Use MCP tools through the server to complete your task.\n\n"
        f"When you believe you have completed the task, call the 'complete_task' function "
        f"with a summary of what you accomplished. You have a total of {scenario.max_turns} turns."
    )

    traj.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please complete this task: {scenario.task_description}"},
    ]

    num_turns = 0
    task_completed = False

    while num_turns < scenario.max_turns and not task_completed:
        num_turns += 1

        try:
            if debug:
                log("LLM request", step=num_turns, model=model_name)

            response = await inference_client.chat.completions.create(
                model=model_name,
                messages=traj.messages, #type: ignore
                tools=tool_schemas,
                max_completion_tokens=sampling_config.max_tokens,
                temperature=sampling_config.temperature,
                top_p=sampling_config.top_p,
                tool_choice="required", # We always require tools to be called
            )

            choice = response.choices[0]
            msg = choice.message

            assistant_msg = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in msg.tool_calls
                ]
            print("=" * 80)
            print(f"Assistant message collected: {assistant_msg}")
            print("=" * 80)
            traj.messages.append(assistant_msg)

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    try:
                        tool_args = json.loads(tool_call.function.arguments or "{}")

                        if tool_call.function.name == "complete_task":
                            print("Task marked complete.")
                            traj.metrics["task_completed"] = True
                            task_completed = True
                            traj.messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": "Task marked complete.",
                            })
                        else:
                            print(f"Calling tool: {tool_call.function.name} with args: {tool_args}")
                            result = await call_mcp_tool(tool_call.function.name, tool_args, mcp_url)
                            content_text = get_content_text(result)

                            if len(content_text) > 20000:
                                raise Exception(f"Tool result too long: {len(content_text)} chars")

                            traj.messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": content_text,
                            })

                    except Exception as e:
                        if debug:
                            traceback.print_exc()
                        traj.logs.append(f"Tool error: {e}")
                        traj.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {str(e)}",
                        })

        except Exception as e:
            if debug:
                traceback.print_exc()
            traj.logs.append(f"Turn {num_turns} error: {e}")
            break

    if not task_completed and num_turns == scenario.max_turns:
        traj.metrics["ran_out_of_turns"] = True

    traj.metrics["num_turns"] = num_turns

    return traj


async def gather_trajectory_groups(
    inference_client: AsyncOpenAI,
    model_name: str,
    scenarios: List[McpScenario],
    rollouts_per_group: int,
    sampling_config: Optional[SamplingConfig] = None,
    debug: bool = False,
    mcp_url: Optional[str] = None,
) -> List[TrajectoryGroup]:
    """Gather multiple trajectory groups by running rollouts in parallel."""

    groups = []
    for scenario in scenarios:
        tasks = [
            rollout(inference_client, model_name, scenario, sampling_config, debug, mcp_url)
            for _ in range(rollouts_per_group)
        ]
        trajectories = await asyncio.gather(*tasks)

        groups.append(TrajectoryGroup(
            scenario_id=scenario.scenario_id or scenario.task_description,
            trajectories=list(trajectories),
            metadata={"task_description": scenario.task_description},
        ))

    return groups

"""Tests for trajectory rollout function with comprehensive mocking."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from train_agent.training.trajectory import (
    rollout,
    McpScenario,
    AssistantTurn,
    _collect_choice_logprobs,
)
from train_agent.model_schemas import SamplingConfig


class MockTokenLogprob:
    """Mock for token logprob object."""
    def __init__(self, logprob: float):
        self.logprob = logprob


class MockFunctionLogprobs:
    """Mock for function call logprobs."""
    def __init__(self, name_logprobs: List[float], arg_logprobs: List[float]):
        self.name = [MockTokenLogprob(lp) for lp in name_logprobs] if name_logprobs else None
        self.arguments = [MockTokenLogprob(lp) for lp in arg_logprobs] if arg_logprobs else None


class MockToolCallLogprobs:
    """Mock for tool call logprobs."""
    def __init__(self, name_logprobs: List[float] = None, arg_logprobs: List[float] = None):
        name_logprobs = name_logprobs or []
        arg_logprobs = arg_logprobs or []
        self.function = MockFunctionLogprobs(name_logprobs, arg_logprobs)


class MockChoiceLogprobs:
    """Mock for choice.logprobs object."""
    def __init__(self, content_logprobs: List[float] = None, tool_call_logprobs: List[MockToolCallLogprobs] = None):
        content_logprobs = content_logprobs or []
        self.content = [MockTokenLogprob(lp) for lp in content_logprobs] if content_logprobs else None
        self.tool_calls = tool_call_logprobs


class MockToolCall:
    """Mock for tool call in response."""
    def __init__(self, tool_id: str, name: str, arguments: str):
        self.id = tool_id
        self.type = "function"
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = arguments


class MockMessage:
    """Mock for chat completion message."""
    def __init__(self, content: str = None, tool_calls: List[MockToolCall] = None):
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    """Mock for chat completion choice."""
    def __init__(self, message: MockMessage, logprobs: MockChoiceLogprobs = None):
        self.message = message
        self.logprobs = logprobs


class MockChatCompletion:
    """Mock for chat completion response."""
    def __init__(self, choices: List[MockChoice]):
        self.choices = choices


class MockTokenizer:
    """Mock tokenizer for position tracking."""
    def __init__(self):
        self.call_count = 0
        self.token_counts = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        """Return progressively longer token sequences."""
        # Simulate tokens growing as conversation progresses
        base_tokens = 50  # System + user message
        tokens_per_turn = 20  # Each assistant turn adds tokens

        # Count assistant messages to determine length
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
        total_length = base_tokens + (assistant_count * tokens_per_turn)

        return list(range(total_length))


@pytest.fixture
def mock_tokenizer():
    """Fixture for mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def mock_inference_client():
    """Fixture for mock OpenAI client."""
    return AsyncMock()


@pytest.fixture
def basic_scenario():
    """Fixture for basic test scenario."""
    return McpScenario(
        task_description="Test task",
        max_turns=3,
        scenario_id="test-scenario-1"
    )


@pytest.fixture
def sampling_config():
    """Fixture for sampling configuration."""
    return SamplingConfig(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024
    )


@pytest.fixture
def mock_tool_schemas():
    """Fixture for mock tool schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "complete_task",
                "description": "Mark task as complete",
                "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}}
            }
        }
    ]


def test_collect_choice_logprobs_with_content():
    """Test logprob collection from content tokens."""
    logprobs = MockChoiceLogprobs(content_logprobs=[-0.1, -0.2, -0.3])
    result = _collect_choice_logprobs(logprobs)
    assert result == [-0.1, -0.2, -0.3]


def test_collect_choice_logprobs_with_tool_calls():
    """Test logprob collection from tool call tokens."""
    tool_call_logprobs = [
        MockToolCallLogprobs(
            name_logprobs=[-0.1, -0.2],
            arg_logprobs=[-0.3, -0.4, -0.5]
        )
    ]
    logprobs = MockChoiceLogprobs(tool_call_logprobs=tool_call_logprobs)
    result = _collect_choice_logprobs(logprobs)
    assert result == [-0.1, -0.2, -0.3, -0.4, -0.5]


def test_collect_choice_logprobs_mixed():
    """Test logprob collection from both content and tool calls."""
    tool_call_logprobs = [
        MockToolCallLogprobs(name_logprobs=[-0.4], arg_logprobs=[-0.5])
    ]
    logprobs = MockChoiceLogprobs(
        content_logprobs=[-0.1, -0.2, -0.3],
        tool_call_logprobs=tool_call_logprobs
    )
    result = _collect_choice_logprobs(logprobs)
    assert result == [-0.1, -0.2, -0.3, -0.4, -0.5]


def test_collect_choice_logprobs_empty():
    """Test logprob collection with no logprobs."""
    result = _collect_choice_logprobs(None)
    assert result == []


@pytest.mark.asyncio
async def test_rollout_task_completion(
    mock_inference_client,
    basic_scenario,
    sampling_config,
    mock_tokenizer,
    mock_tool_schemas
):
    """Test successful task completion with complete_task call."""

    # Mock tool schemas
    with patch('train_agent.training.trajectory.get_tool_schemas_from_mcp_with_complete_task_tool') as mock_get_schemas:
        mock_get_schemas.return_value = mock_tool_schemas

        # Mock LLM response that calls complete_task
        tool_call = MockToolCall(
            tool_id="call_123",
            name="complete_task",
            arguments='{"summary": "Task finished"}'
        )
        message = MockMessage(content=None, tool_calls=[tool_call])
        logprobs = MockChoiceLogprobs(
            tool_call_logprobs=[
                MockToolCallLogprobs(
                    name_logprobs=[-0.1, -0.2],
                    arg_logprobs=[-0.3, -0.4]
                )
            ]
        )
        choice = MockChoice(message=message, logprobs=logprobs)
        completion = MockChatCompletion(choices=[choice])

        mock_inference_client.chat.completions.create.return_value = completion

        # Run rollout
        trajectory = await rollout(
            inference_client=mock_inference_client,
            model_name="test-model",
            scenario=basic_scenario,
            sampling_config=sampling_config,
            debug=False,
            mcp_url="http://test-mcp",
            tokenizer=mock_tokenizer
        )

        # Verify trajectory
        assert trajectory.metrics["task_completed"] is True
        assert trajectory.metrics["ran_out_of_turns"] is False
        assert trajectory.metrics["num_turns"] == 1

        # Verify messages
        assert len(trajectory.messages) == 4  # system, user, assistant, tool
        assert trajectory.messages[0]["role"] == "system"
        assert trajectory.messages[1]["role"] == "user"
        assert trajectory.messages[2]["role"] == "assistant"
        assert trajectory.messages[3]["role"] == "tool"
        assert trajectory.messages[3]["content"] == "Task marked complete."

        # Verify assistant turns
        assert len(trajectory.assistant_turns) == 1
        turn = trajectory.assistant_turns[0]
        assert len(turn.logprobs) == 4  # 2 name + 2 args
        assert turn.start_pos == 50  # Base tokens from mock tokenizer
        assert turn.end_pos > turn.start_pos
        assert turn.turn_idx == 1


@pytest.mark.asyncio
async def test_rollout_with_regular_tool(
    mock_inference_client,
    basic_scenario,
    sampling_config,
    mock_tokenizer,
    mock_tool_schemas
):
    """Test rollout with regular tool call followed by complete_task."""

    with patch('train_agent.training.trajectory.get_tool_schemas_from_mcp_with_complete_task_tool') as mock_get_schemas:
        mock_get_schemas.return_value = mock_tool_schemas

        with patch('train_agent.training.trajectory.call_mcp_tool') as mock_call_tool:
            mock_call_tool.return_value = {"result": "Tool executed"}

            with patch('train_agent.training.trajectory.get_content_text') as mock_get_content:
                mock_get_content.return_value = "Tool result text"

                # First turn: call regular tool
                tool_call_1 = MockToolCall(
                    tool_id="call_1",
                    name="test_tool",
                    arguments='{"param": "value"}'
                )
                message_1 = MockMessage(content="Let me use a tool", tool_calls=[tool_call_1])
                logprobs_1 = MockChoiceLogprobs(content_logprobs=[-0.1, -0.2])

                # Second turn: call complete_task
                tool_call_2 = MockToolCall(
                    tool_id="call_2",
                    name="complete_task",
                    arguments='{"summary": "Done"}'
                )
                message_2 = MockMessage(content=None, tool_calls=[tool_call_2])
                logprobs_2 = MockChoiceLogprobs(content_logprobs=[-0.3])

                # Configure mock to return different responses
                mock_inference_client.chat.completions.create.side_effect = [
                    MockChatCompletion(choices=[MockChoice(message_1, logprobs_1)]),
                    MockChatCompletion(choices=[MockChoice(message_2, logprobs_2)])
                ]

                # Run rollout
                trajectory = await rollout(
                    inference_client=mock_inference_client,
                    model_name="test-model",
                    scenario=basic_scenario,
                    sampling_config=sampling_config,
                    debug=False,
                    mcp_url="http://test-mcp",
                    tokenizer=mock_tokenizer
                )

                # Verify task completed
                assert trajectory.metrics["task_completed"] is True
                assert trajectory.metrics["num_turns"] == 2

                # Verify tool was called
                mock_call_tool.assert_called_once_with(
                    "test_tool",
                    {"param": "value"},
                    "http://test-mcp"
                )

                # Verify assistant turns tracked
                assert len(trajectory.assistant_turns) == 2


@pytest.mark.asyncio
async def test_rollout_max_turns_exceeded(
    mock_inference_client,
    basic_scenario,
    sampling_config,
    mock_tokenizer,
    mock_tool_schemas
):
    """Test rollout running out of turns without completion."""

    with patch('train_agent.training.trajectory.get_tool_schemas_from_mcp_with_complete_task_tool') as mock_get_schemas:
        mock_get_schemas.return_value = mock_tool_schemas

        with patch('train_agent.training.trajectory.call_mcp_tool') as mock_call_tool:
            mock_call_tool.return_value = {"result": "Tool executed"}

            with patch('train_agent.training.trajectory.get_content_text') as mock_get_content:
                mock_get_content.return_value = "Tool result"

                # Always call regular tool (never complete_task)
                tool_call = MockToolCall(
                    tool_id="call_x",
                    name="test_tool",
                    arguments='{}'
                )
                message = MockMessage(content="Working...", tool_calls=[tool_call])
                logprobs = MockChoiceLogprobs(content_logprobs=[-0.1])

                mock_inference_client.chat.completions.create.return_value = MockChatCompletion(
                    choices=[MockChoice(message, logprobs)]
                )

                # Run rollout
                trajectory = await rollout(
                    inference_client=mock_inference_client,
                    model_name="test-model",
                    scenario=basic_scenario,
                    sampling_config=sampling_config,
                    debug=False,
                    mcp_url="http://test-mcp",
                    tokenizer=mock_tokenizer
                )

                # Verify ran out of turns
                assert trajectory.metrics["task_completed"] is False
                assert trajectory.metrics["ran_out_of_turns"] is True
                assert trajectory.metrics["num_turns"] == 3  # max_turns


@pytest.mark.asyncio
async def test_rollout_tool_error_handling(
    mock_inference_client,
    basic_scenario,
    sampling_config,
    mock_tokenizer,
    mock_tool_schemas
):
    """Test rollout handles tool execution errors gracefully."""

    with patch('train_agent.training.trajectory.get_tool_schemas_from_mcp_with_complete_task_tool') as mock_get_schemas:
        mock_get_schemas.return_value = mock_tool_schemas

        with patch('train_agent.training.trajectory.call_mcp_tool') as mock_call_tool:
            # First call: tool error
            # Second call: complete_task
            mock_call_tool.side_effect = [
                Exception("Tool failed"),
                None  # complete_task doesn't call mcp_tool
            ]

            # Turn 1: tool error
            tool_call_1 = MockToolCall(
                tool_id="call_1",
                name="test_tool",
                arguments='{}'
            )
            message_1 = MockMessage(content=None, tool_calls=[tool_call_1])

            # Turn 2: complete_task
            tool_call_2 = MockToolCall(
                tool_id="call_2",
                name="complete_task",
                arguments='{"summary": "Done despite error"}'
            )
            message_2 = MockMessage(content=None, tool_calls=[tool_call_2])

            mock_inference_client.chat.completions.create.side_effect = [
                MockChatCompletion(choices=[MockChoice(message_1)]),
                MockChatCompletion(choices=[MockChoice(message_2)])
            ]

            # Run rollout
            trajectory = await rollout(
                inference_client=mock_inference_client,
                model_name="test-model",
                scenario=basic_scenario,
                sampling_config=sampling_config,
                debug=False,
                mcp_url="http://test-mcp",
                tokenizer=mock_tokenizer
            )

            # Verify error was logged
            assert len(trajectory.logs) > 0
            assert "Tool error" in trajectory.logs[0]

            # Verify error message added to conversation
            error_messages = [m for m in trajectory.messages if m["role"] == "tool" and "Error:" in m["content"]]
            assert len(error_messages) == 1
            assert "Tool failed" in error_messages[0]["content"]

            # Verify task still completed
            assert trajectory.metrics["task_completed"] is True


@pytest.mark.asyncio
async def test_rollout_llm_error_handling(
    mock_inference_client,
    basic_scenario,
    sampling_config,
    mock_tokenizer,
    mock_tool_schemas
):
    """Test rollout handles LLM API errors gracefully."""

    with patch('train_agent.training.trajectory.get_tool_schemas_from_mcp_with_complete_task_tool') as mock_get_schemas:
        mock_get_schemas.return_value = mock_tool_schemas

        # LLM API raises exception
        mock_inference_client.chat.completions.create.side_effect = Exception("API error")

        # Run rollout
        trajectory = await rollout(
            inference_client=mock_inference_client,
            model_name="test-model",
            scenario=basic_scenario,
            sampling_config=sampling_config,
            debug=False,
            mcp_url="http://test-mcp",
            tokenizer=mock_tokenizer
        )

        # Verify error logged and rollout stopped
        assert len(trajectory.logs) > 0
        assert "error" in trajectory.logs[0].lower()
        assert trajectory.metrics["task_completed"] is False
        assert trajectory.metrics["num_turns"] == 1  # Failed on first turn


@pytest.mark.asyncio
async def test_rollout_without_tokenizer(
    mock_inference_client,
    basic_scenario,
    sampling_config,
    mock_tool_schemas
):
    """Test rollout works without tokenizer (no position tracking)."""

    with patch('train_agent.training.trajectory.get_tool_schemas_from_mcp_with_complete_task_tool') as mock_get_schemas:
        mock_get_schemas.return_value = mock_tool_schemas

        # Mock complete_task call
        tool_call = MockToolCall(
            tool_id="call_123",
            name="complete_task",
            arguments='{"summary": "Done"}'
        )
        message = MockMessage(content=None, tool_calls=[tool_call])
        completion = MockChatCompletion(choices=[MockChoice(message)])

        mock_inference_client.chat.completions.create.return_value = completion

        # Run rollout WITHOUT tokenizer
        trajectory = await rollout(
            inference_client=mock_inference_client,
            model_name="test-model",
            scenario=basic_scenario,
            sampling_config=sampling_config,
            debug=False,
            mcp_url="http://test-mcp",
            tokenizer=None  # No tokenizer
        )

        # Verify task completed but no assistant turns tracked
        assert trajectory.metrics["task_completed"] is True
        assert len(trajectory.assistant_turns) == 0  # No position tracking without tokenizer


@pytest.mark.asyncio
async def test_rollout_tool_result_too_long(
    mock_inference_client,
    sampling_config,
    mock_tokenizer,
    mock_tool_schemas
):
    """Test rollout handles tool results that exceed max length."""

    # Use single-turn scenario to avoid multiple error messages
    scenario = McpScenario(
        task_description="Test task",
        max_turns=1,
        scenario_id="test-tool-error"
    )

    with patch('train_agent.training.trajectory.get_tool_schemas_from_mcp_with_complete_task_tool') as mock_get_schemas:
        mock_get_schemas.return_value = mock_tool_schemas

        with patch('train_agent.training.trajectory.call_mcp_tool') as mock_call_tool:
            mock_call_tool.return_value = {"result": "x" * 30000}

            with patch('train_agent.training.trajectory.get_content_text') as mock_get_content:
                # Return text longer than 20000 chars
                mock_get_content.return_value = "x" * 25000

                # Turn 1: tool with long result
                tool_call = MockToolCall(
                    tool_id="call_1",
                    name="test_tool",
                    arguments='{}'
                )
                message = MockMessage(content=None, tool_calls=[tool_call])

                mock_inference_client.chat.completions.create.return_value = MockChatCompletion(
                    choices=[MockChoice(message)]
                )

                # Run rollout
                trajectory = await rollout(
                    inference_client=mock_inference_client,
                    model_name="test-model",
                    scenario=scenario,
                    sampling_config=sampling_config,
                    debug=False,
                    mcp_url="http://test-mcp",
                    tokenizer=mock_tokenizer
                )

                # Verify error was logged
                assert len(trajectory.logs) > 0
                assert "Tool result too long" in trajectory.logs[0]

                # Verify error message added
                error_messages = [m for m in trajectory.messages if m["role"] == "tool" and "Error:" in m["content"]]
                assert len(error_messages) == 1


@pytest.mark.asyncio
async def test_rollout_uses_default_sampling_config(
    mock_inference_client,
    basic_scenario,
    mock_tokenizer,
    mock_tool_schemas
):
    """Test rollout uses default sampling config when not provided."""

    with patch('train_agent.training.trajectory.get_tool_schemas_from_mcp_with_complete_task_tool') as mock_get_schemas:
        mock_get_schemas.return_value = mock_tool_schemas

        tool_call = MockToolCall(
            tool_id="call_123",
            name="complete_task",
            arguments='{"summary": "Done"}'
        )
        message = MockMessage(content=None, tool_calls=[tool_call])
        completion = MockChatCompletion(choices=[MockChoice(message)])

        mock_inference_client.chat.completions.create.return_value = completion

        # Run rollout without sampling_config
        trajectory = await rollout(
            inference_client=mock_inference_client,
            model_name="test-model",
            scenario=basic_scenario,
            sampling_config=None,  # Use default
            debug=False,
            mcp_url="http://test-mcp",
            tokenizer=mock_tokenizer
        )

        # Verify it used default SamplingConfig
        call_kwargs = mock_inference_client.chat.completions.create.call_args[1]
        assert "temperature" in call_kwargs
        assert "top_p" in call_kwargs
        assert "max_completion_tokens" in call_kwargs

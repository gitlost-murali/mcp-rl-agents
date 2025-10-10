"""Tests for vLLM engine (tests/inference/vllm_engine.py)."""

import os
import pytest
import time
import requests

from train_agent.inference import VLLMEngine, VLLMConfig, SamplingConfig


TEST_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def vllm_config():
    return VLLMConfig(
        model_name=TEST_MODEL,
        max_seq_length=2048,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1,
    )


@pytest.fixture(scope="module")
def engine(vllm_config):
    """Start vLLM server for tests and tear it down after."""
    engine = VLLMEngine(vllm_config)

    # Set the base URL for the client to use
    os.environ["VLLM_BASE_URL"] = "http://localhost:8011/v1"

    # Start the server
    engine.start_server(port=8011)

    # Re-initialize client with the correct URL
    engine._init_client()

    yield engine

    # Stop the server
    engine.stop_server()
    del engine


def test_engine_initialization(engine):
    assert engine.client is not None


@pytest.mark.asyncio
async def test_simple_generation(engine):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello and introduce yourself."}
    ]

    sampling_config = SamplingConfig(
        temperature=0.8,
        max_tokens=200
    )

    response = await engine.generate_with_messages(messages, sampling_config)

    assert response is not None
    assert hasattr(response, 'choices')
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None

    print(f"\nGenerated response: {response.choices[0].message.content[:200]}")


@pytest.mark.asyncio
async def test_streaming_generation(engine):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count from 1 to 5."}
    ]

    sampling_config = SamplingConfig(
        temperature=0.7,
        max_tokens=150
    )

    stream = await engine.generate_with_messages(messages, sampling_config, stream=True)

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    assert len(chunks) > 0
    print()


@pytest.mark.asyncio
async def test_tool_calling(engine):
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    sampling_config = SamplingConfig(
        temperature=0.5,
        max_tokens=100
    )

    response = await engine.generate_with_messages(messages, sampling_config, tools=tools)

    assert response is not None
    assert hasattr(response, 'choices')
    print(f"\nTool call response: {response.choices[0].message}")


@pytest.mark.asyncio
async def test_streaming_with_tools(engine):
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "Calculate 23 + 45"}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform arithmetic calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The arithmetic expression"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    sampling_config = SamplingConfig(
        temperature=0.5,
        max_tokens=100
    )

    stream = await engine.generate_with_messages(messages, sampling_config, tools=tools, stream=True)

    tool_calls = []
    async for chunk in stream:
        if chunk.choices[0].delta.tool_calls:
            tool_calls.extend(chunk.choices[0].delta.tool_calls)
            print(f"Tool call delta: {chunk.choices[0].delta.tool_calls}")

    print(f"\nCollected tool calls: {tool_calls}")


def test_engine_config_from_defaults(engine):
    assert engine.config.model_name == TEST_MODEL
    assert engine.config.trust_remote_code is True
    assert engine.config.tensor_parallel_size == 1

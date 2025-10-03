"""Tests for vLLM engine (tests/inference/vllm_engine.py)."""

import pytest
import torch

from train_agent.inference import VLLMEngine, VLLMConfig, SamplingConfig
from train_agent.config import MAX_TURNS


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
    engine = VLLMEngine(vllm_config)
    yield engine
    del engine
    torch.cuda.empty_cache()


def test_engine_initialization(engine):
    assert engine.llm is not None
    assert engine.tokenizer is not None


def test_basic_generation(engine):
    prompts = ["What is 2+2?", "Name a color."]

    sampling_config = SamplingConfig(
        temperature=0.7,
        max_tokens=50
    )

    results = engine.generate(prompts, sampling_config)

    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)
    assert all(len(r) > 0 for r in results)


def test_generate_with_messages(engine):
    messages_list = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."}
        ],
        [
            {"role": "system", "content": "You are a math expert."},
            {"role": "user", "content": "What is 5+3?"}
        ]
    ]

    sampling_config = SamplingConfig(
        temperature=0.8,
        max_tokens=100
    )

    results = engine.generate_with_messages(messages_list, sampling_config)

    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)
    assert all(len(r) > 0 for r in results)


def test_batched_generation(engine):
    prompts = [f"Count to {i}" for i in range(1, 6)]

    sampling_config = SamplingConfig(
        temperature=0.5,
        max_tokens=30
    )

    results = engine.generate(prompts, sampling_config)

    assert len(results) == 5
    assert all(isinstance(r, str) for r in results)


def test_get_tokenizer(engine):
    tokenizer = engine.get_tokenizer()

    assert tokenizer is not None

    test_text = "Hello, world!"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)

    assert isinstance(tokens, list)
    assert isinstance(decoded, str)


@pytest.mark.asyncio
async def test_simple_rollout_simulation(engine):
    system_prompt = (
        "You are a helpful AI assistant. "
        "When you complete the task, respond with 'TASK_COMPLETE: <summary>'."
    )

    task_description = "Say hello and introduce yourself."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please complete this task: {task_description}"}
    ]

    sampling_config = SamplingConfig(
        temperature=0.8,
        max_tokens=200
    )

    result = engine.generate_with_messages([messages], sampling_config)

    assert len(result) == 1
    assert isinstance(result[0], str)
    assert len(result[0]) > 0

    print(f"\nGenerated response: {result[0][:200]}")


@pytest.mark.asyncio
async def test_batched_rollout_simulation(engine):
    tasks = [
        "Count from 1 to 5.",
        "Name three colors.",
        "What is 2+2?",
    ]

    messages_list = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Task: {task}"}
        ]
        for task in tasks
    ]

    sampling_config = SamplingConfig(
        temperature=0.7,
        max_tokens=150
    )

    results = engine.generate_with_messages(messages_list, sampling_config)

    assert len(results) == len(tasks)
    assert all(isinstance(r, str) for r in results)
    assert all(len(r) > 0 for r in results)

    for task, result in zip(tasks, results):
        print(f"\nTask: {task}")
        print(f"Response: {result[:100]}")


@pytest.mark.asyncio
async def test_multi_turn_conversation(engine):
    conversation_history = [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is 5+3?"}
    ]

    sampling_config = SamplingConfig(
        temperature=0.5,
        max_tokens=100
    )

    max_turns = min(3, MAX_TURNS)

    for turn in range(max_turns):
        result = engine.generate_with_messages([conversation_history], sampling_config)

        assert len(result) == 1
        response = result[0]

        print(f"\nTurn {turn + 1}:")
        print(f"Response: {response[:100]}")

        conversation_history.append({"role": "assistant", "content": response})

        if turn < max_turns - 1:
            conversation_history.append({
                "role": "user",
                "content": "Thank you. Now what is 10-2?"
            })

    assert len(conversation_history) > 2


@pytest.mark.asyncio
async def test_rollout_with_different_sampling(engine):
    messages = [
        {"role": "system", "content": "You are a creative writer."},
        {"role": "user", "content": "Write a short sentence about a cat."}
    ]

    sampling_configs = [
        SamplingConfig(temperature=0.3, max_tokens=50),
        SamplingConfig(temperature=0.7, max_tokens=50),
        SamplingConfig(temperature=1.0, max_tokens=50),
    ]

    for i, config in enumerate(sampling_configs):
        result = engine.generate_with_messages([messages], config)

        assert len(result) == 1
        print(f"\nTemperature {config.temperature}: {result[0][:100]}")


def test_engine_config_from_defaults(engine):
    assert engine.config.model_name == TEST_MODEL
    assert engine.config.trust_remote_code is True
    assert engine.config.tensor_parallel_size == 1

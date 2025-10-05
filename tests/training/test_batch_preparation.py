"""Tests for batch_preparation alignment fix."""

import pytest
from transformers import AutoTokenizer

from train_agent.training.trajectory import Trajectory
from train_agent.training.batch_preparation import (
    tokenize_trajectory_with_loss_mask,
    validate_trajectory_alignment,
)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def test_simple_trajectory_alignment(tokenizer):
    """Test alignment with a simple single-turn trajectory."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."},
    ]

    # Simulate position tracking (as done in rollout)
    pre_tokens = tokenizer.apply_chat_template(
        messages[:2], tokenize=True, add_generation_prompt=False
    )
    post_tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )

    start_pos = len(pre_tokens)
    end_pos = len(post_tokens)
    num_assistant_tokens = end_pos - start_pos

    traj = Trajectory(
        messages=messages,
        assistant_turns=[{
            'logprobs': [-0.5] * num_assistant_tokens,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'turn_idx': 1,
        }],
    )

    # Should pass validation
    assert validate_trajectory_alignment(traj, tokenizer)

    # Should tokenize correctly
    result = tokenize_trajectory_with_loss_mask(traj, tokenizer, max_length=512)

    assert result['input_ids'].shape[0] == 512
    assert result['loss_mask'].sum() == num_assistant_tokens
    assert (result['old_logprobs'] != 0).sum() == num_assistant_tokens

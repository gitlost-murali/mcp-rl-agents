"""Tests for GRPO training utilities."""

import pytest
import numpy as np
import torch

from train_agent.training.grpo import (
    GroupRollout,
    calculate_group_advantages,
    calculate_batch_advantages,
    get_group_statistics,
    compute_drgrpo_loss,
)


def test_group_rollout_creation():
    trajectories = ["traj1", "traj2", "traj3"]
    rewards = [1.0, 2.0, 3.0]

    group = GroupRollout(
        scenario_id="test_scenario",
        trajectories=trajectories,
        rewards=rewards,
    )

    assert group.scenario_id == "test_scenario"
    assert group.num_rollouts == 3
    assert len(group.trajectories) == 3
    assert len(group.rewards) == 3


def test_group_rollout_validation_error():
    trajectories = ["traj1", "traj2"]
    rewards = [1.0, 2.0, 3.0]

    with pytest.raises(ValueError, match="must match"):
        GroupRollout(
            scenario_id="test_scenario",
            trajectories=trajectories,
            rewards=rewards,
        )


def test_group_rollout_properties():
    rewards = [1.0, 2.0, 3.0, 4.0]
    group = GroupRollout(
        scenario_id="test",
        trajectories=["t1", "t2", "t3", "t4"],
        rewards=rewards,
    )

    assert group.mean_reward == 2.5
    assert group.num_rollouts == 4
    assert group.std_reward == pytest.approx(np.std(rewards))


def test_get_trajectory():
    trajectories = ["traj1", "traj2", "traj3"]
    rewards = [1.0, 2.0, 3.0]

    group = GroupRollout(
        scenario_id="test",
        trajectories=trajectories,
        rewards=rewards,
    )

    traj, reward = group.get_trajectory(1)
    assert traj == "traj2"
    assert reward == 2.0


def test_get_trajectory_index_error():
    group = GroupRollout(
        scenario_id="test",
        trajectories=["t1", "t2"],
        rewards=[1.0, 2.0],
    )

    with pytest.raises(IndexError):
        group.get_trajectory(5)


def test_calculate_advantages_mean_normalized():
    group = GroupRollout(
        scenario_id="test",
        trajectories=["t1", "t2", "t3", "t4"],
        rewards=[1.0, 2.0, 3.0, 4.0],
    )

    advantages = calculate_group_advantages(group, advantage_type="mean_normalized")

    assert len(advantages) == 4
    assert advantages.mean() == pytest.approx(0.0, abs=1e-6)


def test_calculate_advantages_mean_centered():
    group = GroupRollout(
        scenario_id="test",
        trajectories=["t1", "t2", "t3"],
        rewards=[1.0, 2.0, 3.0],
    )

    advantages = calculate_group_advantages(group, advantage_type="mean_centered")

    assert advantages.mean() == pytest.approx(0.0)
    assert advantages[0] == pytest.approx(-1.0)
    assert advantages[2] == pytest.approx(1.0)


def test_calculate_advantages_max_relative():
    group = GroupRollout(
        scenario_id="test",
        trajectories=["t1", "t2", "t3"],
        rewards=[1.0, 2.0, 3.0],
    )

    advantages = calculate_group_advantages(group, advantage_type="max_relative")

    assert advantages[2] == 0.0
    assert advantages[1] == -1.0
    assert advantages[0] == -2.0


def test_calculate_advantages_percentile():
    group = GroupRollout(
        scenario_id="test",
        trajectories=["t1", "t2", "t3", "t4"],
        rewards=[1.0, 2.0, 3.0, 4.0],
    )

    advantages = calculate_group_advantages(group, advantage_type="percentile")

    assert len(advantages) == 4
    assert advantages.min() >= -0.5
    assert advantages.max() <= 0.5


def test_calculate_advantages_invalid_type():
    group = GroupRollout(
        scenario_id="test",
        trajectories=["t1", "t2"],
        rewards=[1.0, 2.0],
    )

    with pytest.raises(ValueError, match="Unknown advantage_type"):
        calculate_group_advantages(group, advantage_type="invalid")


def test_calculate_batch_advantages():
    groups = [
        GroupRollout(scenario_id="s1", trajectories=["t1", "t2"], rewards=[1.0, 2.0]),
        GroupRollout(scenario_id="s2", trajectories=["t3", "t4"], rewards=[3.0, 4.0]),
    ]

    batch_advantages = calculate_batch_advantages(groups)

    assert len(batch_advantages) == 2
    assert len(batch_advantages[0]) == 2
    assert len(batch_advantages[1]) == 2


def test_get_group_statistics():
    group = GroupRollout(
        scenario_id="test_scenario",
        trajectories=["t1", "t2", "t3", "t4"],
        rewards=[1.0, 2.0, 3.0, 4.0],
        group_id="group_1",
    )

    stats = get_group_statistics(group)

    assert stats["num_rollouts"] == 4
    assert stats["mean_reward"] == 2.5
    assert stats["min_reward"] == 1.0
    assert stats["max_reward"] == 4.0
    assert stats["median_reward"] == 2.5
    assert stats["reward_range"] == 3.0
    assert stats["scenario_id"] == "test_scenario"
    assert stats["group_id"] == "group_1"


def test_compute_drgrpo_loss_no_clipping():
    """Test Dr. GRPO loss when ratio is within clipping bounds."""
    # Small difference in logprobs -> ratio close to 1
    new_logprobs = torch.tensor([[-1.0, -2.0, -3.0]])  # [1, 3]
    old_logprobs = torch.tensor([[-1.1, -2.1, -3.1]])  # [1, 3]
    advantages = torch.tensor([1.0])  # [1]
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]])  # [1, 3] - all tokens included

    loss = compute_drgrpo_loss(new_logprobs, old_logprobs, advantages, loss_mask, clip_epsilon=0.2)

    # When no clipping occurs, loss = -sum(exp(new - old) * advantages * mask)
    prob_ratio = torch.exp(new_logprobs - old_logprobs)
    expected_loss = -(prob_ratio * advantages.unsqueeze(-1) * loss_mask).sum()

    assert loss.item() == pytest.approx(expected_loss.item(), abs=1e-5)


def test_compute_drgrpo_loss_with_clipping():
    """Test Dr. GRPO loss when ratio exceeds clipping bounds."""
    # Large difference in logprobs to trigger clipping
    new_logprobs = torch.tensor([[0.0, 0.0]])  # [1, 2]
    old_logprobs = torch.tensor([[-2.0, 2.0]])  # [1, 2] ratio: ~7.39, ~0.135
    advantages = torch.tensor([1.0])  # [1]
    loss_mask = torch.tensor([[1.0, 1.0]])  # [1, 2] - all tokens included

    loss = compute_drgrpo_loss(new_logprobs, old_logprobs, advantages, loss_mask, clip_epsilon=0.2)

    # First ratio ~7.39 should be clipped to 1.2
    # Second ratio ~0.135 should be clipped to 0.8
    expected_objective = min(1.2 * 1.0, torch.exp(torch.tensor(2.0)).item() * 1.0) + \
                         min(0.8 * 1.0, torch.exp(torch.tensor(-2.0)).item() * 1.0)
    expected_loss = -expected_objective

    assert loss.item() == pytest.approx(expected_loss, abs=1e-5)


def test_compute_drgrpo_loss_negative_advantages():
    """Test Dr. GRPO loss with negative advantages (bad rollouts)."""
    new_logprobs = torch.tensor([[0.0], [0.0]])  # [2, 1]
    old_logprobs = torch.tensor([[-1.0], [1.0]])  # [2, 1]
    advantages = torch.tensor([-1.0, -1.0])  # [2] - Both are bad rollouts
    loss_mask = torch.tensor([[1.0], [1.0]])  # [2, 1] - all tokens included

    loss = compute_drgrpo_loss(new_logprobs, old_logprobs, advantages, loss_mask, clip_epsilon=0.2)

    # With negative advantages, we want the loss to discourage these actions
    # The objective should be negative, making the loss positive
    assert loss.item() > 0


def test_compute_drgrpo_loss_zero_advantages():
    """Test Dr. GRPO loss with zero advantages."""
    new_logprobs = torch.tensor([[0.0, -1.0, -2.0]])  # [1, 3]
    old_logprobs = torch.tensor([[-1.0, -2.0, -3.0]])  # [1, 3]
    advantages = torch.tensor([0.0])  # [1]
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]])  # [1, 3] - all tokens included

    loss = compute_drgrpo_loss(new_logprobs, old_logprobs, advantages, loss_mask, clip_epsilon=0.2)

    # With zero advantages, loss should be zero
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_compute_drgrpo_loss_clipping_bounds():
    """Test that clipping bounds are correctly applied."""
    # Test upper bound clipping
    new_logprobs = torch.tensor([[0.0]])  # [1, 1]
    old_logprobs = torch.tensor([[-3.0]])  # [1, 1] ratio: e^3 ≈ 20.09
    advantages = torch.tensor([1.0])  # [1]
    loss_mask = torch.tensor([[1.0]])  # [1, 1] - all tokens included

    loss = compute_drgrpo_loss(new_logprobs, old_logprobs, advantages, loss_mask, clip_epsilon=0.2)

    # Ratio should be clipped to 1.2
    expected_loss = -1.2 * 1.0
    assert loss.item() == pytest.approx(expected_loss, abs=1e-5)

    # Test lower bound clipping
    new_logprobs = torch.tensor([[0.0]])  # [1, 1]
    old_logprobs = torch.tensor([[3.0]])  # [1, 1] ratio: e^-3 ≈ 0.0498
    advantages = torch.tensor([1.0])  # [1]
    loss_mask = torch.tensor([[1.0]])  # [1, 1] - all tokens included

    loss = compute_drgrpo_loss(new_logprobs, old_logprobs, advantages, loss_mask, clip_epsilon=0.2)

    # Ratio e^-3 ≈ 0.0498 should be clipped to 0.8
    # The clipped objective is 0.8 * 1.0 = 0.8
    # The unclipped objective is ~0.0498 * 1.0 = ~0.0498
    # min(0.8, 0.0498) = 0.0498, so loss = -0.0498
    expected_loss = -torch.exp(torch.tensor(-3.0)).item() * 1.0
    assert loss.item() == pytest.approx(expected_loss, abs=1e-5)


def test_compute_drgrpo_loss_custom_epsilon():
    """Test Dr. GRPO loss with custom clipping epsilon."""
    new_logprobs = torch.tensor([[0.0]])  # [1, 1]
    old_logprobs = torch.tensor([[-2.0]])  # [1, 1]
    advantages = torch.tensor([1.0])  # [1]
    loss_mask = torch.tensor([[1.0]])  # [1, 1] - all tokens included

    # With epsilon=0.1, clipping range is [0.9, 1.1]
    loss = compute_drgrpo_loss(new_logprobs, old_logprobs, advantages, loss_mask, clip_epsilon=0.1)

    # Ratio e^2 ≈ 7.39 should be clipped to 1.1
    expected_loss = -1.1 * 1.0
    assert loss.item() == pytest.approx(expected_loss, abs=1e-5)


def test_compute_drgrpo_loss_batch():
    """Test Dr. GRPO loss with a realistic batch."""
    batch_size = 8
    seq_len = 16
    torch.manual_seed(42)

    new_logprobs = torch.randn(batch_size, seq_len, requires_grad=True) * 0.5
    old_logprobs = torch.randn(batch_size, seq_len) * 0.5
    advantages = torch.randn(batch_size)
    loss_mask = torch.ones(batch_size, seq_len)  # All tokens included

    loss = compute_drgrpo_loss(new_logprobs, old_logprobs, advantages, loss_mask, clip_epsilon=0.2)

    # Loss should be a scalar
    assert loss.dim() == 0

    # Loss should be finite
    assert torch.isfinite(loss)

    # Verify gradient can flow through
    assert loss.requires_grad


def test_compute_drgrpo_loss_with_masking():
    """Test that loss_mask correctly filters out non-assistant tokens."""
    new_logprobs = torch.tensor([[0.0, 0.0, 0.0, 0.0]])  # [1, 4]
    old_logprobs = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])  # [1, 4]
    advantages = torch.tensor([1.0])  # [1]

    # Only include first two tokens (e.g., assistant tokens only)
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])  # [1, 4]

    loss_with_mask = compute_drgrpo_loss(new_logprobs, old_logprobs, advantages, loss_mask, clip_epsilon=0.2)

    # Compare with full mask
    full_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])  # [1, 4]
    loss_full = compute_drgrpo_loss(new_logprobs, old_logprobs, advantages, full_mask, clip_epsilon=0.2)

    # With mask, loss should be exactly half of full loss (since we mask out half the tokens)
    assert loss_with_mask.item() == pytest.approx(loss_full.item() / 2.0, abs=1e-5)

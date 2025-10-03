"""Tests for GRPO training utilities."""

import pytest
import numpy as np

from train_agent.training.grpo import (
    GroupRollout,
    calculate_group_advantages,
    calculate_batch_advantages,
    get_group_statistics,
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

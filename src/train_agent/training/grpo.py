"""
Group Relative Policy Optimization (GRPO) training utilities.

GRPO is a variant of policy gradient methods that computes advantages
relative to other rollouts in the same group, rather than using a baseline.
This approach is particularly useful for RL from AI feedback scenarios.
"""

from typing import Any, List, Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator


class GroupRollout(BaseModel):
    """Data structure for storing a group of rollouts with their trajectories and rewards."""

    scenario_id: str
    trajectories: List[Any]
    rewards: List[float]
    group_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

    @field_validator('rewards')
    @classmethod
    def validate_rewards_length(cls, v, info):
        trajectories = info.data.get('trajectories')
        if trajectories and len(v) != len(trajectories):
            raise ValueError(
                f"Number of trajectories ({len(trajectories)}) must match "
                f"number of rewards ({len(v)})"
            )
        return v

    @property
    def num_rollouts(self) -> int:
        return len(self.trajectories)

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards))

    @property
    def std_reward(self) -> float:
        return float(np.std(self.rewards))

    def get_trajectory(self, idx: int) -> tuple[Any, float]:
        if idx < 0 or idx >= self.num_rollouts:
            raise IndexError(f"Index {idx} out of range for group with {self.num_rollouts} rollouts")
        return self.trajectories[idx], self.rewards[idx]


def calculate_group_advantages(
    group: GroupRollout,
    advantage_type: str = "mean_normalized",
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Calculate advantages for each trajectory in a group relative to others in the group.

    Args:
        group: GroupRollout containing trajectories and rewards
        advantage_type: Type of advantage calculation to use:
            - "mean_normalized": (reward - mean) / (std + epsilon)
            - "mean_centered": reward - mean
            - "max_relative": reward - max_reward
            - "percentile": Rank-based advantages using percentiles
        epsilon: Small constant for numerical stability in normalization

    Returns:
        Array of advantages, one per trajectory in the group
    """
    rewards = np.array(group.rewards, dtype=np.float32)

    if advantage_type == "mean_normalized":
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        advantages = (rewards - mean_reward) / (std_reward + epsilon)

    elif advantage_type == "mean_centered":
        mean_reward = np.mean(rewards)
        advantages = rewards - mean_reward

    elif advantage_type == "max_relative":
        max_reward = np.max(rewards)
        advantages = rewards - max_reward

    elif advantage_type == "percentile":
        n = len(rewards)
        ranks = np.argsort(np.argsort(rewards))
        advantages = (ranks / (n - 1)) - 0.5

    else:
        raise ValueError(
            f"Unknown advantage_type: {advantage_type}. "
            f"Must be one of: mean_normalized, mean_centered, max_relative, percentile"
        )

    return advantages


def calculate_batch_advantages(
    groups: List[GroupRollout],
    advantage_type: str = "mean_normalized",
    epsilon: float = 1e-8
) -> List[np.ndarray]:
    """Calculate advantages for multiple groups in a batch."""
    return [
        calculate_group_advantages(group, advantage_type, epsilon)
        for group in groups
    ]


def create_group_rollout_from_trajectories(
    scenario_id: str,
    trajectories: List[Any],
    rewards: List[float],
    group_id: Optional[str] = None,
    metadata: Optional[dict] = None
) -> GroupRollout:
    """Factory function to create a GroupRollout from trajectories and rewards."""
    return GroupRollout(
        scenario_id=scenario_id,
        trajectories=trajectories,
        rewards=rewards,
        group_id=group_id,
        metadata=metadata or {}
    )


def get_group_statistics(group: GroupRollout) -> dict:
    """Compute statistics about a group of rollouts."""
    rewards = np.array(group.rewards)

    return {
        "num_rollouts": group.num_rollouts,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "median_reward": float(np.median(rewards)),
        "reward_range": float(np.max(rewards) - np.min(rewards)),
        "scenario_id": group.scenario_id,
        "group_id": group.group_id,
    }


def compute_weighted_advantages(
    group: GroupRollout,
    advantage_type: str = "mean_normalized",
    weight_by_length: bool = False,
    trajectory_lengths: Optional[List[int]] = None,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Calculate advantages with optional trajectory length weighting to account
    for varying episode durations.
    """
    advantages = calculate_group_advantages(group, advantage_type, epsilon)

    if weight_by_length:
        if trajectory_lengths is None:
            raise ValueError("trajectory_lengths must be provided when weight_by_length=True")
        if len(trajectory_lengths) != group.num_rollouts:
            raise ValueError(
                f"Number of trajectory_lengths ({len(trajectory_lengths)}) must match "
                f"number of rollouts ({group.num_rollouts})"
            )

        lengths = np.array(trajectory_lengths, dtype=np.float32)
        weights = lengths / np.mean(lengths)
        advantages = advantages * weights

    return advantages

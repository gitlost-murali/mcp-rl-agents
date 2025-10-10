"""
Batch preparation utilities for GRPO training.

This module handles:
1. Tokenizing trajectories using chat template
2. Creating loss masks based on assistant turn indices
3. Aligning old_logprobs with tokenized sequences
4. Creating PyTorch Dataset and DataLoader
"""

import torch
import numpy as np
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer #type: ignore

from train_agent.training.trajectory import Trajectory


def validate_trajectory_alignment(
    trajectory: Trajectory,
    tokenizer: PreTrainedTokenizer,
    verbose: bool = False,
) -> bool:
    """
    Validate that stored position boundaries match actual tokenization.

    This is crucial for ensuring old_logprobs and loss_mask are correctly aligned.

    Args:
        trajectory: Trajectory with assistant_turns data
        tokenizer: Same tokenizer used during rollout
        verbose: Print detailed alignment information

    Returns:
        True if alignment is valid, False otherwise
    """
    if not trajectory.assistant_turns:
        if verbose:
            print("⚠️  No assistant turns to validate")
        return True

    # Tokenize full conversation
    full_tokens = tokenizer.apply_chat_template(
        trajectory.messages,
        tokenize=True,
        add_generation_prompt=False,
    )

    total_mismatches = 0

    for turn_idx, turn in enumerate(trajectory.assistant_turns):
        start = turn.start_pos
        end = turn.end_pos
        expected_length = end - start
        actual_logprob_count = len(turn.logprobs)

        if expected_length != actual_logprob_count:
            total_mismatches += 1
            if verbose:
                print(f"❌ Turn {turn_idx + 1}: Position range [{start}, {end}) "
                      f"length={expected_length} != logprob_count={actual_logprob_count}")
        elif verbose:
            print(f"✅ Turn {turn_idx + 1}: Position range [{start}, {end}) "
                  f"matches logprob_count={actual_logprob_count}")

    if total_mismatches == 0:
        if verbose:
            print(f"✅ Alignment validated: All {len(trajectory.assistant_turns)} turns aligned correctly")
        return True
    else:
        print(f"❌ ALIGNMENT ERROR: {total_mismatches}/{len(trajectory.assistant_turns)} turns misaligned")
        return False


def tokenize_trajectory_with_loss_mask(
    trajectory: Trajectory,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 8192,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a trajectory and create aligned loss mask and old_logprobs.

    Strategy:
    1. Use tokenizer's chat template to tokenize the full conversation
    2. Use stored position boundaries from rollout to place loss_mask and old_logprobs
    3. This ensures exact alignment since positions were measured during rollout

    Args:
        trajectory: Trajectory object with messages and assistant_turns data
        tokenizer: HuggingFace tokenizer (must be same as used during rollout)
        max_length: Maximum sequence length

    Returns:
        Dictionary with:
        - 'input_ids': [seq_len]
        - 'attention_mask': [seq_len]
        - 'loss_mask': [seq_len] - 1 for assistant tokens, 0 for others
        - 'old_logprobs': [seq_len] - token-level old logprobs (0 for non-assistant)
    """
    # Tokenize full conversation using chat template
    full_encoding = tokenizer.apply_chat_template(
        trajectory.messages,
        tokenize=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = full_encoding["input_ids"].squeeze(0) # type: ignore
    attention_mask = full_encoding["attention_mask"].squeeze(0) #type:ignore

    # Create loss mask and old_logprobs arrays
    seq_len = input_ids.shape[0]
    loss_mask = torch.zeros(seq_len, dtype=torch.float32)
    old_logprobs = torch.zeros(seq_len, dtype=torch.float32)

    # Use stored position boundaries to place logprobs and loss mask
    for turn in trajectory.assistant_turns:
        start = turn.start_pos
        end = turn.end_pos
        turn_logprobs = turn.logprobs

        # Mark these positions in loss mask (handle truncation)
        actual_end = min(end, seq_len)
        loss_mask[start:actual_end] = 1.0

        # Place logprobs at these positions
        for i, logprob in enumerate(turn_logprobs):
            pos = start + i
            if pos < seq_len:
                old_logprobs[pos] = logprob
            else:
                break  # Sequence was truncated

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "old_logprobs": old_logprobs,
    }


class GRPODataset(Dataset):
    """
    PyTorch Dataset for GRPO training.

    Stores tokenized trajectories with advantages and aligned old_logprobs.
    """

    def __init__(
        self,
        trajectories: List[Trajectory],
        advantages: np.ndarray,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 8192,
    ):
        """
        Args:
            trajectories: List of Trajectory objects
            advantages: Numpy array of advantages [num_trajectories]
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.trajectories = trajectories
        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-tokenize all trajectories
        print(f"Tokenizing {len(trajectories)} trajectories with loss masks...")
        self.tokenized_data = []

        for i, traj in enumerate(trajectories):
            tokenized = tokenize_trajectory_with_loss_mask(traj, tokenizer, max_length)
            self.tokenized_data.append(tokenized)

            if (i + 1) % 10 == 0:
                print(f"  Tokenized {i + 1}/{len(trajectories)} trajectories")

        print(f"Tokenization complete!")

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Returns:
            Dictionary with format expected by GRPOLightningModule:
            {
                'input_ids': [seq_len],
                'attention_mask': [seq_len],
                'loss_mask': [seq_len],
                'old_logprobs': [seq_len],
                'advantages': scalar,
            }
        """
        return {
            "input_ids": self.tokenized_data[idx]["input_ids"],
            "attention_mask": self.tokenized_data[idx]["attention_mask"],
            "loss_mask": self.tokenized_data[idx]["loss_mask"],
            "old_logprobs": self.tokenized_data[idx]["old_logprobs"],
            "advantages": self.advantages[idx],
        }


def prepare_training_batches(
    trajectories: List[Trajectory],
    advantages: np.ndarray,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 8192,
    shuffle: bool = True,
) -> DataLoader:
    """
    Prepare a DataLoader for GRPO training.

    Args:
        trajectories: List of Trajectory objects
        advantages: Numpy array of advantages [num_trajectories]
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for training
        max_length: Maximum sequence length
        shuffle: Whether to shuffle the dataset

    Returns:
        PyTorch DataLoader yielding batches for GRPOLightningModule
    """
    dataset = GRPODataset(
        trajectories=trajectories,
        advantages=advantages,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Use 0 for debugging, can increase for performance
        pin_memory=True,
    )

    return dataloader


def prepare_batches_from_trajectory_groups(
    trajectory_groups: List[Any],
    advantages_list: List[np.ndarray],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 8192,
) -> DataLoader:
    """
    Prepare training batches from multiple trajectory groups.

    Flattens multiple trajectory groups into a single DataLoader.

    Args:
        trajectory_groups: List of TrajectoryGroup objects
        advantages_list: List of advantage arrays, one per group
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for training
        max_length: Maximum sequence length

    Returns:
        DataLoader with all trajectories from all groups
    """
    # Flatten trajectories and advantages
    all_trajectories = []
    all_advantages = []

    for group, advantages in zip(trajectory_groups, advantages_list):
        all_trajectories.extend(group.trajectories)
        all_advantages.extend(advantages)

    all_advantages = np.array(all_advantages, dtype=np.float32)

    print(f"Preparing batches from {len(trajectory_groups)} groups, "
          f"{len(all_trajectories)} total trajectories")

    return prepare_training_batches(
        trajectories=all_trajectories,
        advantages=all_advantages,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True,
    )

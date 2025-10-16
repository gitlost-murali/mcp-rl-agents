# @title Let's generate our train and validation scenarios!

import json
import os
import random
import time

from dotenv import load_dotenv

# Import the generate_scenarios function from art.mcp and logging utilities

from train_agent.utils.settings import settings
from train_agent.utils.mcp_utils import list_tools_and_resources
from train_agent.config import TRAINING_CONFIG, NUM_TEST_INPUTS, INPUT_GENERATION_MODEL


openrouter_key = settings.openrouter_key


async def generate_scenarios_from_tools_and_resources(tools_list: list[dict], 
                                                      resources_list: list[dict],
                                                      expected_total: int) -> list[dict]:

    raise NotImplementedError("Not implemented")

def split_scenarios_into_train_and_val(scenarios: list[dict]) -> tuple[list[dict], list[dict]]:
    random.shuffle(scenarios)
    train_n = TRAINING_CONFIG["num_training_inputs"]
    raw_train_scenarios = scenarios[:train_n]
    raw_val_scenarios = scenarios[train_n:]

    print(f"Train: {len(raw_train_scenarios)} | Val: {len(raw_val_scenarios)}")

    print("Sample (train) preview:")
    print(raw_train_scenarios[:5])

    print("Sample (val) preview:")
    print(raw_val_scenarios[:5], min(5, len(raw_val_scenarios)))

    print("Done.")

    return raw_train_scenarios, raw_val_scenarios

def save_train_and_val_scenarios(train_scenarios: list[dict], val_scenarios: list[dict], filename: str):
    with open(filename, "w") as f:
        json.dump({"train": train_scenarios, "val": val_scenarios}, f, indent=4)

def load_train_and_val_scenarios(filename: str) -> tuple[list[dict], list[dict]]:
    with open(filename, "r") as f:
        data = json.load(f)
        return data["train"], data["val"]
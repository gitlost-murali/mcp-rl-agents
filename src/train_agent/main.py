import asyncio
import os

from train_agent.data.dataset_generator import (
    generate_scenarios_from_tools_and_resources,
    save_train_and_val_scenarios,
    split_scenarios_into_train_and_val,
)
from train_agent.utils.mcp_utils import (
    convert_tools_and_resources_to_dicts,
    list_tools_and_resources,
)
from train_agent.utils.settings import settings
from train_agent.config import DATASET_FILENAME, TRAINING_CONFIG, NUM_TEST_INPUTS


async def generate_dataset_if_not_exists(dataset_filename: str):
    if os.path.exists(dataset_filename):
        print("Dataset already exists, skipping generation.")
        return

    mcp_url = settings.mcp_url
    tools, resources = await list_tools_and_resources(mcp_url)
    tools_list, resources_list = convert_tools_and_resources_to_dicts(tools, resources)
    print("Tools:", [t.name for t in tools])
    expected_total = TRAINING_CONFIG["num_training_inputs"] + NUM_TEST_INPUTS
    scenarios = await generate_scenarios_from_tools_and_resources(
        tools_list, resources_list, expected_total
    )
    raw_train_scenarios, raw_val_scenarios = split_scenarios_into_train_and_val(
        scenarios
    )
    save_train_and_val_scenarios(
        raw_train_scenarios, raw_val_scenarios, dataset_filename
    )


async def main():
    await generate_dataset_if_not_exists(DATASET_FILENAME)


def cli():
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())

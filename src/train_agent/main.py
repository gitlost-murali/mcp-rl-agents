import asyncio
import os

from train_agent.model_schemas import GRPOConfig

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments:True" in conf:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""  # or set to something benign like "max_split_size_mb:512"

from train_agent.data.dataset_generator import (
    generate_scenarios_from_tools_and_resources,
    load_train_and_val_scenarios,
    save_train_and_val_scenarios,
    split_scenarios_into_train_and_val,
)
from train_agent.train import ModelTrainer
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
    model_trainer = ModelTrainer(GRPOConfig.from_config())
    raw_train_scenarios, raw_val_scenarios = load_train_and_val_scenarios(DATASET_FILENAME)
    
    await model_trainer.train(raw_train_scenarios[:2])
    await model_trainer.test(raw_val_scenarios[:2])


def cli():
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())

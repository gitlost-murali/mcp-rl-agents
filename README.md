# Train Agent

MCP RL training system using GRPO (Group Relative Policy Optimization) for training language models to use MCP tools. Inspired from [OpenPipe's ART MCP-RL](https://github.com/OpenPipe/ART).

## Setup

```bash
# Install dependencies
uv sync

# Configure environment (.env file)
mcp_url=<your-mcp-server-url>
mcp_bearer_token=<token>
openrouter_key=<key>
```

## Usage

```bash
# Run training
python -m train_agent.main

# Run tests
pytest -m "not slow and not gpu"

# Lint
ruff check --fix src/ tests/
```

## Architecture

- **Online GRPO**: Fresh rollouts collected each step from current policy
- **vLLM Integration**: Manages GPU memory via server start/stop cycles
- **Token-Level Training**: GRPO loss with proper position tracking and masking
- **Multi-GPU**: FSDP strategy with vLLM tensor parallelism

Key files: `train.py`, `training/lightning_module.py`, `training/grpo.py`, `inference/vllm_engine.py`
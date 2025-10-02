# MCP Agent Training System - Project Goals

## Vision

Create a specialized agent training system that enables users to automatically train high-performance language model agents for any MCP (Model Context Protocol) server. The core value proposition is simple yet powerful:

**Input**: An MCP server URL + A base LLM
**Output**: A fine-tuned agent that excels at using the specific tools exposed by that MCP server

This transforms generic language models into specialized, tool-proficient agents through automated reinforcement learning from AI feedback (RLAIF).

**This project trains models specifically for a given MCP server's tool set using GRPO (Group Relative Policy Optimization) with LLM-as-judge feedback.**

## Core Workflow

### 1. Discovery Phase
- Connect to any MCP server via HTTP/SSE transport
- Automatically discover available tools and resources
- Extract tool schemas, descriptions, and parameters

### 2. Dataset Generation Phase (Optional - deprioritized for later)
- Generate diverse, realistic task scenarios that require using the discovered tools
- Create scenarios with varying complexity (single-tool, multi-tool, error-recovery)
- Validate scenarios are achievable and meaningful
- Split into training and validation sets
- **Note**: This step can be manual initially - users provide their own task scenarios

### 3. Training Phase
- Execute multiple rollouts per training scenario
- Agent attempts to complete tasks by calling MCP tools
- Track full conversation trajectories (messages, tool calls, results)
- Score trajectories using LLM-as-judge (RLAIF - Reinforcement Learning from AI Feedback)
- Train model using **GRPO (Group Relative Policy Optimization)** - simpler than DPO, groups rollouts and assigns relative rewards
- Iteratively improve through multiple epochs

### 4. Evaluation Phase
- Test trained model on held-out validation scenarios
- Measure task completion rate, tool usage efficiency, error recovery
- Compare against base model performance

## Current Architecture (ART-Based)

### Dependencies
- **OpenPipe ART**: Core RL training framework (the dependency we want to remove)
  - `art.TrainableModel`: Model wrapper with vLLM inference + LoRA training
  - `art.gather_trajectory_groups`: Parallel rollout execution
  - `art.ruler_score_group`: RLAIF scoring via judge models
  - `art.LocalBackend`: Local training orchestration
  - Integration with torchtune for LoRA fine-tuning

- **FastMCP**: MCP client library for tool discovery and execution
- **vLLM**: Inference engine (embedded in ART)
- **torchtune**: PyTorch fine-tuning (embedded in ART)
- **OpenRouter/OpenAI**: Judge models for RLAIF scoring

### Key Components

#### 1. MCP Integration (`mcp_utils.py`)
- Connects to remote MCP servers via HTTP transport with bearer token auth
- Lists available tools and resources
- Executes tool calls and returns structured results
- Converts MCP schemas to OpenAI tool calling format
- Injects synthetic `complete_task` tool for episode termination

#### 2. Configuration (`config.py`)
- Model selection (currently Qwen3-4B-Instruct)
- Training hyperparameters (learning rate, rollouts per group, epochs)
- Resource limits (max turns, sequence length, GPU memory)
- Dataset sizing (training/validation split)
- Judge model configuration (o4-mini via OpenRouter)

#### 3. Dataset Generation (`main.py`, missing `data/dataset_generator.py`)
- Referenced but not present in current codebase
- Should generate scenarios from tool schemas
- Expected to use o3 model for synthetic scenario creation
- Saves to JSON for reuse across training runs

#### 4. Training Loop (`train.py`)
- **Rollout function**: Core agent execution loop
  - Builds system prompt explaining MCP agent role
  - Iterates for max_turns or until task completion
  - Calls LLM with messages + available tools
  - Executes tool calls via MCP client
  - Appends results back to conversation
  - Tracks success metrics (task_completed, ran_out_of_turns, num_turns)

- **ModelTrainer class**:
  - Initializes ART TrainableModel with LoRA + vLLM config
  - Registers model with LocalBackend
  - Orchestrates training epochs over dataset
  - Gathers trajectory groups (multiple rollouts per scenario)
  - Scores trajectories using RULER (judge model assigns relative preferences)
  - Trains model to optimize for high-scoring trajectories
  - Checkpoints to `.art/` directory

#### 5. Utilities
- `settings.py`: Environment variable management (MCP URL, tokens, API keys)
- `debug_utils.py`: Timestamped logging for debugging rollouts

### Current Limitations

1. **Heavy ART Dependency**: Tightly coupled to OpenPipe's proprietary framework
   - Hard to customize training algorithms
   - Limited to ART's supported models and configurations
   - Opaque internals (vLLM + torchtune abstractions)
   - Dependency on ART's backend infrastructure

2. **Missing Dataset Generator**: Critical component not in repository
   - Unclear how scenarios are generated from tool schemas
   - No visibility into prompt engineering for scenario creation
   - Can't iterate on dataset quality

3. **Fixed RLAIF Approach**: Relies solely on RULER scoring
   - No support for other reward signals (ground truth, simulation, human feedback)
   - Expensive judge model calls (o4-mini)
   - No reward shaping or curriculum learning

4. **Limited Observability**: Minimal insights during training
   - Basic logging but no rich metrics dashboards
   - Hard to debug why models aren't improving
   - No trajectory visualization or analysis tools

5. **Inference Coupling**: Trained models require vLLM for deployment
   - Can't easily export to other serving frameworks
   - LoRA adapters tied to ART's checkpoint format

## Rewrite Objectives

### 1. Remove ART Dependency - Implement GRPO Training
Replace with custom, transparent implementation focused on GRPO:

**Training Engine**:
- **GRPO (Group Relative Policy Optimization)** implementation using PyTorch + Transformers
- GRPO is simpler than DPO: group multiple rollouts per prompt, assign relative rewards, optimize policy
- **PyTorch Lightning** for training orchestration with clear separation of concerns:
  - LightningModule: Model definition, training/validation steps, loss computation
  - Trainer: Multi-GPU distribution (DDP/FSDP), checkpointing, logging
  - Clean abstractions make code more maintainable and testable
- Explicit control over training loop, batching, gradient updates
- Flexible optimizer configuration (AdamW, custom learning rate schedules)

**Why GRPO over DPO?**
- GRPO: Generate multiple completions per prompt, judge ranks them, train on relative preferences within group
- DPO: Requires paired preference data (chosen vs rejected), more complex setup
- GRPO is simpler and more natural for RL from AI feedback scenarios

**Inference Engine**:
- **vLLM only** for inference (batched generation for parallel rollouts)
- Focus on getting core training working, not inference flexibility

**LoRA Training**:
- Direct use of PEFT library for LoRA adapters
- Configurable rank, alpha, target modules
- Merge adapters to base model or serve separately

### 2. Dataset Generation (DEPRIORITIZED - Last Priority)
- Users can provide manual task scenarios initially
- Automated generation comes later, after core training works
- When implemented: LLM-based scenario generation from tool schemas

### 3. LLM-as-Judge Reward System
Keep it simple and focused:

**AI Feedback (RLAIF) - Primary Approach**:
- LLM judge scores trajectory quality (task completion, tool usage correctness)
- Configurable judge prompts and criteria
- Support for different judge models (GPT-4o-mini, Claude, etc.)

**Optional Enhancements** (lower priority):
- Rule-based bonuses/penalties (penalize excessive tool calls, reward efficiency)
- Ground truth validation for servers with known correct outputs

### 4. Observability
Focus on essential metrics:

**Training Metrics**:
- Loss curves, gradient norms, learning rate schedules
- Task completion rates over time
- Log locally with optional Weights & Biases integration

**Trajectory Logging**:
- Save trajectories to disk for offline analysis
- Basic statistics (success rate, avg turns, tool usage)

## Technical Requirements

### Core Technologies
- **Python 3.12+**: Modern async/await, type hints
- **PyTorch 2.0+**: Training framework
- **PyTorch Lightning**: Training orchestration with clear abstractions (LightningModule, Trainer)
- **Transformers**: GRPO implementation, model loading, tokenization
- **PEFT**: LoRA adapters
- **vLLM**: Inference engine for rollouts
- **FastMCP**: MCP protocol client
- **Anthropic/OpenAI SDKs**: Judge model APIs

### Model Support
- Start with: Qwen3-4B, Llama 3 8B, Phi-4
- Support any HuggingFace model with chat template
- Configurable context lengths (8K - 128K)

### Hardware Requirements
- **Training**: Multi-GPU setup (we have compute, optimize for this)
- Distributed training via PyTorch Lightning (handles DDP/FSDP automatically)

## Success Criteria

### Phase 1: Core GRPO Training (Priority: Highest)
- [ ] GRPO implementation with PyTorch + Transformers
- [ ] PyTorch Lightning module for clean training orchestration
- [ ] LoRA training pipeline with PEFT
- [ ] vLLM inference for rollouts
- [ ] LLM-as-judge reward scoring
- [ ] Multi-GPU training with PyTorch Lightning Trainer
- [ ] Reproduce/exceed current training results without ART dependency

### Phase 2: Observability & Tooling (Priority: High)
- [ ] Local logging with optional Weights & Biases
- [ ] Trajectory logging to disk
- [ ] Training metrics (loss curves, success rates)
- [ ] Simple CLI for training and evaluation

### Phase 3: Dataset Generation (Priority: Last)
- [ ] Automated scenario generation from MCP tool schemas (optional, manual scenarios work initially)
- [ ] LLM-based task generation when needed

## Future Enhancements (Optional, Lower Priority)

### Potential Improvements (evaluate later)
- **Context Management**: Intelligently summarize long tool outputs
- Other enhancements to be determined based on initial results

## Non-Goals

- Building MCP servers (we consume existing servers)
- General-purpose agent frameworks (focused on tool use, not reasoning)
- Inference optimization (we have compute)
- Export to GGUF or specialized serving formats
- Extensive benchmarking across many MCP servers
- Advanced training techniques (curriculum learning, distillation, reward modeling)

## Metrics for Evaluation

### Training Efficiency
- Time to reach target performance
- GPU hours required
- Cost of judge model API calls

### Agent Performance
- Task completion rate (% of scenarios successfully completed)
- Tool usage efficiency (# of tool calls per task)
- Turn efficiency (# of conversation turns to completion)
- Error recovery rate (% of failed tool calls successfully recovered)

### Generalization
- Performance on held-out validation scenarios
- Transfer to related but unseen tasks

---

## Getting Started (Post-Rewrite)

```bash
# 1. Install dependencies
uv pip install -e .

# 2. Configure MCP server
export MCP_URL="https://your-mcp-server.com"
export MCP_BEARER_TOKEN="your-token"

# 3. Prepare dataset (manual JSON file with task scenarios)
# Example: data/scenarios.json
# [
#   {"task": "Search for Python tutorials and summarize top 3"},
#   {"task": "Find weather in San Francisco and compare to New York"}
# ]

# 4. Train agent with GRPO
train-agent train \
  --dataset data/scenarios.json \
  --base-model Qwen/Qwen3-4B-Instruct \
  --output-dir checkpoints/my-agent \
  --judge-model openai/gpt-4o-mini \
  --num-gpus 4

# 5. Evaluate
train-agent eval \
  --model checkpoints/my-agent/final \
  --dataset data/scenarios.json \
  --split validation
```

## Conclusion

This rewrite transforms the project from an opaque, ART-dependent prototype into a clean, focused system for training specialized MCP agents using GRPO. By removing the ART dependency and implementing GRPO directly, we gain:

1. **Simplicity**: GRPO is straightforward - group rollouts, judge ranks them, optimize
2. **Control**: Full visibility into training process with PyTorch + Transformers + Accelerate
3. **Focus**: Core training first, optional features later (dataset generation is last priority)
4. **Performance**: Multi-GPU training optimized for our compute resources

The end result is a tool that trains agents specifically for MCP tool use via GRPO with LLM-as-judge feedback.

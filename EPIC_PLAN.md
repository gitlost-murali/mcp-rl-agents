# Epic Planner: MCP Agent Training System Rewrite

## Branch Strategy
Branch: `epic-planner-grpo-rewrite`

## Overview
Transform the ART-dependent prototype into a clean GRPO-based training system using PyTorch Lightning + Transformers.

---

## **EPIC 1: Core GRPO Training Infrastructure** (Priority: HIGHEST)

### Ticket 1.1: Remove ART Dependency & Setup Base Architecture ✅
- [x] Remove `openpipe-art` from dependencies in pyproject.toml
- [x] Add PyTorch Lightning to dependencies
- [x] Add Transformers library to dependencies
- [x] Add PEFT library to dependencies
- [x] Add vLLM to dependencies
- [x] Create `src/train_agent/training/` module directory
- [x] Create `src/train_agent/inference/` module directory
- [x] Create `src/train_agent/rewards/` module directory
- [x] Update imports in train.py to remove ART references
- [x] Update imports in config.py to remove ART references

### Ticket 1.2: Implement vLLM Inference Engine
- [ ] Create `src/train_agent/inference/__init__.py`
- [ ] Create `src/train_agent/inference/vllm_engine.py`
- [ ] Implement VLLMEngine class with initialization
- [ ] Add batched generation support for parallel rollouts
- [ ] Add LoRA adapter loading support
- [ ] Add tokenizer management
- [ ] Add sampling parameters configuration
- [ ] Add GPU memory management settings
- [ ] Test basic inference functionality
- [ ] Test batched inference functionality

### Ticket 1.3: Implement GRPO Algorithm Core
- [ ] Create `src/train_agent/training/__init__.py`
- [ ] Create `src/train_agent/training/grpo.py`
- [ ] Implement group-based rollout data structure
- [ ] Implement advantage calculation within groups
- [ ] Implement relative reward normalization
- [ ] Implement policy gradient loss computation
- [ ] Add KL divergence penalty computation
- [ ] Add entropy bonus computation
- [ ] Test advantage calculation logic
- [ ] Test loss computation with mock data

### Ticket 1.4: Create PyTorch Lightning Training Module
- [ ] Create `src/train_agent/training/lightning_module.py`
- [ ] Implement GRPOLightningModule class
- [ ] Implement `__init__` with model and config
- [ ] Implement `forward` method
- [ ] Implement `training_step` with GRPO loss
- [ ] Implement `validation_step`
- [ ] Implement `configure_optimizers` with AdamW
- [ ] Add learning rate scheduler configuration
- [ ] Add gradient clipping configuration
- [ ] Add checkpointing hooks
- [ ] Add logging hooks for metrics

### Ticket 1.5: Integrate PEFT for LoRA Training
- [ ] Create `src/train_agent/training/lora_config.py`
- [ ] Define LoRAConfig dataclass with parameters
- [ ] Add rank configuration (default: 16)
- [ ] Add alpha configuration (default: 32)
- [ ] Add target_modules configuration
- [ ] Add dropout configuration
- [ ] Integrate LoraConfig with Lightning module
- [ ] Add adapter loading functionality
- [ ] Add adapter saving functionality
- [ ] Add adapter merging functionality
- [ ] Test LoRA integration with base model

### Ticket 1.6: Update Rollout Function
- [ ] Create `src/train_agent/training/trajectory.py`
- [ ] Define Trajectory dataclass (messages, reward, metrics)
- [ ] Define TrajectoryGroup dataclass
- [ ] Refactor rollout() function to use VLLMEngine
- [ ] Remove art.Trajectory wrapper references
- [ ] Update message tracking logic
- [ ] Maintain MCP tool calling functionality
- [ ] Update reward tracking
- [ ] Update metrics tracking
- [ ] Test rollout with new engine

### Ticket 1.7: Multi-GPU Training Setup
- [ ] Configure Lightning Trainer for DDP strategy
- [ ] Configure Lightning Trainer for FSDP strategy (optional)
- [ ] Add distributed batch gathering logic
- [ ] Add gradient synchronization config
- [ ] Add GPU memory optimization settings
- [ ] Configure mixed precision training (fp16/bf16)
- [ ] Test on single GPU
- [ ] Test on multi-GPU setup
- [ ] Add device placement utilities

---

## **EPIC 2: Reward System & LLM-as-Judge** (Priority: HIGH)

### Ticket 2.1: Implement LLM Judge Infrastructure
- [ ] Create `src/train_agent/rewards/__init__.py`
- [ ] Create `src/train_agent/rewards/judge.py`
- [ ] Define JudgeConfig dataclass
- [ ] Implement LLMJudge base class
- [ ] Add support for OpenAI models (GPT-4o-mini)
- [ ] Add support for Anthropic models (Claude)
- [ ] Add support for OpenRouter models
- [ ] Implement configurable judge prompts
- [ ] Add batch scoring functionality
- [ ] Add retry logic with exponential backoff
- [ ] Add error handling for API failures
- [ ] Test judge scoring with mock trajectories

### Ticket 2.2: Replace RULER Scoring
- [ ] Create `src/train_agent/rewards/scorer.py`
- [ ] Implement score_trajectory_group function
- [ ] Migrate logic from art.ruler_score_group
- [ ] Implement relative scoring within groups
- [ ] Add score normalization (mean=0, std=1)
- [ ] Add reward clipping configuration
- [ ] Integrate with LLMJudge
- [ ] Test scoring with various trajectory groups
- [ ] Compare results with original RULER

### Ticket 2.3: Add Rule-Based Rewards (Optional)
- [ ] Create `src/train_agent/rewards/rules.py`
- [ ] Define RuleBasedReward base class
- [ ] Implement efficiency penalty (excessive tool calls)
- [ ] Implement success bonus reward
- [ ] Implement turn efficiency bonus
- [ ] Implement error recovery bonus
- [ ] Make rules configurable via config
- [ ] Add reward combination logic
- [ ] Test rule-based rewards
- [ ] Document reward configuration

---

## **EPIC 3: Training Pipeline & Orchestration** (Priority: HIGH)

### Ticket 3.1: Refactor ModelTrainer Class
- [ ] Update `train.py::ModelTrainer.__init__`
- [ ] Replace art.TrainableModel with Lightning module
- [ ] Remove art.LocalBackend dependency
- [ ] Implement PyTorch Lightning Trainer initialization
- [ ] Update train() method for new architecture
- [ ] Implement trajectory gathering loop
- [ ] Integrate trajectory scoring
- [ ] Update checkpoint management
- [ ] Remove register_model() method
- [ ] Update test() method for evaluation
- [ ] Test end-to-end training flow

### Ticket 3.2: Dataset Processing Pipeline
- [ ] Create `src/train_agent/data/__init__.py`
- [ ] Create `src/train_agent/data/dataset.py`
- [ ] Define ScenarioDataset class (PyTorch Dataset)
- [ ] Implement `__init__` with JSON loading
- [ ] Implement `__getitem__` method
- [ ] Implement `__len__` method
- [ ] Add train/val split functionality
- [ ] Add data augmentation hooks (optional)
- [ ] Create DataLoader wrapper
- [ ] Test dataset loading
- [ ] Test iteration over dataset

### Ticket 3.3: Training Configuration System
- [ ] Refactor `config.py` for new architecture
- [ ] Add GRPO hyperparameters section
- [ ] Add rollouts_per_group configuration
- [ ] Add advantage normalization settings
- [ ] Add KL penalty coefficient
- [ ] Add Lightning Trainer configurations
- [ ] Add num_gpus, precision, strategy settings
- [ ] Add vLLM engine configurations
- [ ] Add LoRA configuration section
- [ ] Add judge configuration section
- [ ] Document all configuration options

### Ticket 3.4: CLI Improvements
- [ ] Update `main.py` with argparse or click
- [ ] Add `train` subcommand
- [ ] Add `eval` subcommand
- [ ] Add `infer` subcommand
- [ ] Add config file loading (--config flag)
- [ ] Add verbose logging options (--verbose, --debug)
- [ ] Add checkpoint loading flags
- [ ] Add output directory configuration
- [ ] Test all CLI commands
- [ ] Add CLI help documentation

---

## **EPIC 4: Observability & Metrics** (Priority: MEDIUM)

### Ticket 4.1: Local Logging System
- [ ] Create `src/train_agent/logging/__init__.py`
- [ ] Create `src/train_agent/logging/local_logger.py`
- [ ] Implement LocalLogger class
- [ ] Log training loss over time
- [ ] Log gradient norms
- [ ] Log learning rate
- [ ] Log task completion rates
- [ ] Log average rewards
- [ ] Save metrics to JSON/CSV files
- [ ] Create timestamped log directories
- [ ] Test logging functionality

### Ticket 4.2: Weights & Biases Integration
- [ ] Create `src/train_agent/logging/wandb_logger.py`
- [ ] Implement WandBLogger wrapper
- [ ] Integrate with Lightning WandbLogger
- [ ] Log training metrics to W&B
- [ ] Log validation metrics to W&B
- [ ] Log gradient histograms
- [ ] Log system stats (GPU, memory)
- [ ] Log trajectory samples as tables
- [ ] Create W&B dashboard config
- [ ] Make W&B optional via config flag
- [ ] Test W&B logging

### Ticket 4.3: Trajectory Persistence
- [ ] Create `src/train_agent/logging/trajectory_logger.py`
- [ ] Implement TrajectoryLogger class
- [ ] Save trajectories to JSONL format
- [ ] Include trajectory metadata (task, scenario)
- [ ] Include rewards and metrics
- [ ] Include timestamps
- [ ] Add trajectory filtering options
- [ ] Add trajectory sampling (avoid saving all)
- [ ] Create trajectory analysis utilities
- [ ] Test trajectory saving and loading

### Ticket 4.4: Training Statistics
- [ ] Create `src/train_agent/logging/statistics.py`
- [ ] Calculate success rate over epochs
- [ ] Calculate average turns to completion
- [ ] Track tool usage patterns
- [ ] Measure turn efficiency metrics
- [ ] Calculate error recovery rates
- [ ] Generate summary reports (text/markdown)
- [ ] Create plots for key metrics
- [ ] Save statistics to disk
- [ ] Test statistics computation

---

## **EPIC 5: Testing & Validation** (Priority: MEDIUM)

### Ticket 5.1: Unit Tests for GRPO
- [ ] Create `tests/` directory structure
- [ ] Create `tests/test_grpo.py`
- [ ] Test advantage calculation function
- [ ] Test reward normalization
- [ ] Test policy loss computation
- [ ] Test KL penalty computation
- [ ] Test gradient flow (mock backward pass)
- [ ] Mock rollout data for testing
- [ ] Add edge case tests (empty groups, etc.)
- [ ] Ensure 80%+ code coverage for GRPO
- [ ] Run tests with pytest

### Ticket 5.2: Integration Tests
- [ ] Create `tests/test_integration.py`
- [ ] Test end-to-end training (1 epoch, 2 scenarios)
- [ ] Test MCP connection and tool discovery
- [ ] Test vLLM inference engine
- [ ] Test trajectory gathering
- [ ] Test scoring pipeline
- [ ] Test multi-GPU training (if available)
- [ ] Test checkpoint saving and loading
- [ ] Test resuming from checkpoint
- [ ] Add CI/CD configuration for tests

### Ticket 5.3: Evaluation Pipeline
- [ ] Create `src/train_agent/evaluation/__init__.py`
- [ ] Create `src/train_agent/evaluation/evaluator.py`
- [ ] Implement Evaluator class
- [ ] Run trained model on validation set
- [ ] Calculate task completion rate
- [ ] Calculate average tool calls per task
- [ ] Calculate average turns per task
- [ ] Calculate error recovery rate
- [ ] Compare metrics to base model
- [ ] Generate evaluation report
- [ ] Test evaluation pipeline

### Ticket 5.4: Reproduce ART Results
- [ ] Prepare same dataset used with ART
- [ ] Train model with new GRPO implementation
- [ ] Run evaluation on same validation set
- [ ] Compare final completion rates
- [ ] Compare training efficiency (time, GPU hours)
- [ ] Compare model quality metrics
- [ ] Document comparison results
- [ ] Validate GRPO matches/exceeds ART performance

---

## **EPIC 6: Documentation & Polish** (Priority: LOW)

### Ticket 6.1: Update README
- [ ] Update project description
- [ ] Add installation instructions (uv pip install)
- [ ] Add quick start guide
- [ ] Add configuration examples
- [ ] Add training command examples
- [ ] Add evaluation command examples
- [ ] Add troubleshooting section
- [ ] Add hardware requirements
- [ ] Add links to detailed documentation
- [ ] Test README instructions on fresh environment

### Ticket 6.2: API Documentation
- [ ] Add docstrings to all public functions
- [ ] Add docstrings to all classes
- [ ] Document configuration parameters
- [ ] Create architecture overview document
- [ ] Create training algorithm explanation
- [ ] Create GRPO vs DPO comparison doc
- [ ] Generate API docs with Sphinx (optional)
- [ ] Create architecture diagrams (optional)
- [ ] Document reward system design

### Ticket 6.3: Example Scenarios
- [ ] Create `examples/` directory
- [ ] Add sample dataset (examples/sample_scenarios.json)
- [ ] Add basic training script (examples/train_basic.sh)
- [ ] Add multi-GPU training script (examples/train_multigpu.sh)
- [ ] Add evaluation script (examples/evaluate.sh)
- [ ] Add Jupyter notebook for analysis (examples/analyze_results.ipynb)
- [ ] Test all example scripts
- [ ] Document examples in README

---

## **EPIC 7: Dataset Generation (Optional)** (Priority: LAST)

### Ticket 7.1: Implement LLM-Based Generation
- [ ] Create `src/train_agent/data/generator.py`
- [ ] Implement ScenarioGenerator class
- [ ] Extract tool schemas from MCP server
- [ ] Create prompts for scenario generation
- [ ] Use LLM (o3/o4-mini) to generate scenarios
- [ ] Parse generated scenarios into JSON format
- [ ] Add diversity parameters (complexity, tool coverage)
- [ ] Add batch generation support
- [ ] Save generated datasets to file
- [ ] Test scenario generation

### Ticket 7.2: Scenario Validation
- [ ] Create `src/train_agent/data/validator.py`
- [ ] Implement ScenarioValidator class
- [ ] Check scenario achievability (tools exist)
- [ ] Ensure tool coverage (all tools used)
- [ ] Filter invalid or malformed scenarios
- [ ] Add quality scoring for scenarios
- [ ] Add deduplication logic
- [ ] Test validation pipeline
- [ ] Document validation criteria

---

## Implementation Order

**Phase 1 (Week 1-2)**: Epic 1 (Core GRPO) - Tickets 1.1 through 1.7
**Phase 2 (Week 3)**: Epic 2 (Rewards) + Epic 3 (Pipeline) - Tickets 2.1-2.3, 3.1-3.4
**Phase 3 (Week 4)**: Epic 4 (Observability) + Epic 5 (Testing) - Tickets 4.1-4.4, 5.1-5.4
**Phase 4 (Week 5+)**: Epic 6 (Documentation) + Epic 7 (Dataset Gen - Optional) - Tickets 6.1-6.3, 7.1-7.2

---

## Success Criteria

### Phase 1 Complete When:
- [ ] Code runs without ART dependency
- [ ] GRPO training loop executes successfully
- [ ] Multi-GPU training works
- [ ] Checkpoints are saved and loadable

### Phase 2 Complete When:
- [ ] LLM judge scoring works
- [ ] Training improves model performance
- [ ] CLI commands functional

### Phase 3 Complete When:
- [ ] Metrics are logged and visualizable
- [ ] Tests pass with >80% coverage
- [ ] Results match or exceed ART baseline

### Phase 4 Complete When:
- [ ] Documentation is complete
- [ ] Examples run successfully
- [ ] Project is ready for external users

---

## Notes

- This plan removes ART dependency completely
- GRPO is simpler than DPO for RL from AI feedback
- PyTorch Lightning provides clean abstractions
- Focus on core training first, polish later
- Dataset generation is lowest priority (manual scenarios work initially)

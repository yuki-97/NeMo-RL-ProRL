# GRPO-DAPO Usage Guide

## Overview

GRPO-DAPO (Difficulty-Aware Policy Optimization) extends GRPO with intelligent instance filtering to focus training on samples that provide the most learning signal. This is inspired by the `AsyncLLMServerManagerDAPO` approach used in the verl_internal framework.

## Key Features

### Difficulty-Aware Filtering

DAPO filters training instances based on trajectory success rates:

- **Easy instances** (all trajectories succeed) → Filtered out
- **Hard instances** (no trajectories succeed) → Filtered out  
- **Mixed instances** (some succeed, some fail) → Used for training

This approach focuses the model's learning on instances at the right difficulty level.

### Dynamic Data Streaming

Instead of iterating through a fixed dataloader, DAPO continuously:
1. Fetches new instances from the dataloader
2. Processes them with OpenHands environment
3. Filters based on success rates
4. Continues until the target batch size is reached

This ensures efficient GPU utilization and focuses on valuable training signals.

## Architecture

### Components Created

1. **`OpenhandsEnvironmentDAPO`** (`nemo_rl/environments/openhands_environment_dapo.py`)
   - Extends `OpenhandsEnvironment` with DAPO filtering capabilities
   - Manages job queues with priority scheduling
   - Handles dynamic data streaming from dataloader
   - Filters easy/hard instances

2. **`grpo_train_dapo`** (`nemo_rl/algorithms/grpo_dapo.py`)
   - Training loop adapted for DAPO
   - Calls `env.run_async_rollout_dapo(batch_size)` instead of iterating dataloader
   - Environment handles all data management internally

3. **`run_grpo_openhands_dapo.py`** (`examples/run_grpo_openhands_dapo.py`)
   - Entry point for GRPO-DAPO training
   - Sets up environment with DAPO support
   - Configures dataloader for streaming

## Usage

### Configuration

Create a configuration file with DAPO enabled:

```yaml
grpo:
  dapo:
    enable: true  # Enable DAPO filtering
  
  num_generations_per_prompt: 4  # Number of trajectories per instance

data:
  train_batch_size: 32  # Target batch size after filtering

env:
  do_rollout_in_env: true  # Required for DAPO
```

See `examples/configs/grpo_math_1B_dapo.yaml` for a complete example.

### Running Training

```bash
python examples/run_grpo_openhands_dapo.py \
    --config examples/configs/grpo_math_1B_dapo.yaml \
    grpo.max_num_steps=10000
```

### Key Differences from Standard GRPO

| Aspect | Standard GRPO | GRPO-DAPO |
|--------|--------------|-----------|
| Data iteration | Fixed batches from dataloader | Dynamic streaming with filtering |
| Batch processing | All instances processed | Only mixed-difficulty instances used |
| Training efficiency | May waste compute on easy/hard samples | Focuses on informative samples |
| Implementation | `grpo_train()` | `grpo_train_dapo()` |
| Environment | `OpenhandsEnvironment` | `OpenhandsEnvironmentDAPO` |

## Implementation Details

### Filtering Logic

The `filter_easy_hard_instance()` method implements the core DAPO filtering:

```python
def filter_easy_hard_instance(self, all_responses: dict) -> tuple[dict, list]:
    """Filter instances where all or none trajectories are resolved."""
    for instance_id in keys:
        if len(all_responses[instance_id]) == self.num_trajectories:
            resolved_count = sum(
                1 for traj in all_responses[instance_id].values() 
                if traj['resolved']
            )
            
            # Filter if all resolved or none resolved
            if resolved_count in [0, self.num_trajectories]:
                all_responses.pop(instance_id)
                filtered_instance_ids.append(instance_id)
```

### Job Queue Management

DAPO uses priority queues for efficient job scheduling:

```python
self.job_queue = asyncio.PriorityQueue()  # Priority-based scheduling
```

Jobs are dispatched to available OpenHands servers dynamically, and results are collected as they complete.

### Batch Management

The environment maintains an internal batch accumulator:

```python
self.all_input_batch = None  # Accumulated input batches
```

New data is fetched from the dataloader when needed, and filtered instances are removed while remaining instances are retained for future processing.

## Comparison with verl_internal

This implementation mirrors the structure of verl_internal's DAPO approach:

| verl_internal | NeMo-RL-ProRL |
|---------------|---------------|
| `AsyncLLMServerManager` | `OpenhandsEnvironment` |
| `AsyncLLMServerManagerDAPO` | `OpenhandsEnvironmentDAPO` |
| `RayPPOTrainer` | `grpo_train()` |
| `RayPPOTrainerDAPO` | `grpo_train_dapo()` |

The key adaptation is using OpenHands environment for multi-turn agent interactions instead of direct LLM server calls.

## Performance Considerations

### Benefits

- **Improved sample efficiency**: Focus on informative instances
- **Better convergence**: Avoid wasting gradient updates on easy/hard samples
- **Dynamic difficulty**: Adapts to model improvement over training

### Trade-offs

- **Increased latency per batch**: Must process more instances to reach target batch size
- **Memory overhead**: Maintains queue of pending instances
- **Complexity**: More complex data flow compared to standard training

## Debugging

Enable detailed logging to monitor DAPO behavior:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Key metrics to monitor:
- `resolved_ratio`: Percentage of trajectories that succeed
- `finish_ratio`: Percentage of trajectories that complete properly
- `resolved_instance_ratio`: Percentage of instances with at least one success
- Number of filtered instances (easy/hard)

## Extending DAPO

To customize filtering behavior, modify `filter_easy_hard_instance()`:

```python
# Example: Keep instances with 25-75% success rate
def filter_easy_hard_instance(self, all_responses: dict):
    for instance_id in keys:
        resolved_count = ...
        resolved_ratio = resolved_count / self.num_trajectories
        
        # Custom threshold
        if resolved_ratio < 0.25 or resolved_ratio > 0.75:
            all_responses.pop(instance_id)
            filtered_instance_ids.append(instance_id)
```

## Troubleshooting

### Common Issues

1. **"DAPO must be enabled"**: Ensure `grpo.dapo.enable=true` in config
2. **"data_loader must be set"**: The environment's `data_loader` is set in `run_grpo_openhands_dapo.py`
3. **Slow batch generation**: Adjust `num_generations_per_prompt` or filtering thresholds

### Validation

To verify DAPO is working correctly:
- Check logs for "Filtered instance X with resolved ratio Y/Z"
- Monitor `resolved_instance_ratio` - should be between 0 and 1
- Verify batch sizes match `train_batch_size` after filtering

## References

- Original GRPO implementation: `nemo_rl/algorithms/grpo.py`
- verl_internal DAPO: `verl_internal/verl/trainer/ppo/ray_trainer_dapo.py`
- AsyncLLM DAPO: `verl_internal/verl/nvidia/rollout/async_server_dapo.py`


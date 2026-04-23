# Congestion Classification Pipeline

Two-phase pipeline to train a model that predicts when MAPF-GPT will fail due to congestion.

## Overview

| Phase | Description | Labels | Purpose |
|-------|-------------|--------|---------|
| **Phase 1** | Train classifier on auto-labels | 6,844 from solver diff | Learn from clear failures |
| **Phase 2** | Fine-tune with human feedback | ~39 human labels | Refine edge cases |

---

## Phase 1: Data Collection & Training

### Step 1: Collect Data

```bash
# Collect congestion data from episodes
python -c "
from finetuning.congestion_data_collector import *
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig
from lacam.inference import LacamInference, LacamInferenceConfig
from finetuning.scenario_generators import make_pogema_maze_instance
from pogema_toolbox.registry import ToolboxRegistry

ToolboxRegistry.setup_logger('DEBUG')

# Setup (same as delta_data_generator.py)
learnable_algo = MAPFGPTInference(MAPFGPTInferenceConfig(device='cuda', path_to_weights='weights/model-2M.pt'))
fast_solver = LacamInference(LacamInferenceConfig(time_limit=2, timeouts=[2], lacam_lib_path='lacam/liblacam.so'))
expert_solver = LacamInference(LacamInferenceConfig(time_limit=10, timeouts=[10], lacam_lib_path='lacam/liblacam.so'))

# Create environments
envs = [make_pogema_maze_instance(num_agents=32, max_episode_steps=256, map_seed=i, scenario_seed=i) for i in range(100)]

# Collect
cfg = CongestionDataCollectorConfig(diff_threshold=3)
data = collect_congestion_data(envs, learnable_algo, fast_solver, expert_solver, cfg)

# Save
save_congestion_dataset(data, 'dataset/congestion/train.arrow')
"
```

### Step 2: Train Classifier on Confident Buckets

```bash
python finetuning/train_congestion_classifier.py \
    --data dataset/congestion/train.arrow \
    --output out/congestion_classifier.pt \
    --epochs 10 \
    --batch_size 256
```

By default, Phase 1 trains only on:
- `diff < 1` -> confident negative
- `diff > 3` -> confident positive

Samples with `1 <= diff <= 3` are marked `uncertain` and excluded from
supervised pretraining.

---

## Phase 2: Human Feedback + Contrastive

### Step 1: Export High-Value Review Samples

```bash
python finetuning/export_active_learning_samples.py \
    --data dataset/congestion/train.arrow \
    --model out/congestion_classifier.pt \
    --output dataset/congestion/review_samples.json \
    --num_samples 100
```

### Step 2: Edit Human Labels

Open `dataset/congestion/review_samples.json` and:
1. Review samples where the model is likely wrong or uncertain
2. Change `human_label` field (0 = pass, 1 = fail)
3. Add optional notes

### Step 3: Fine-tune with Contrastive Loss

```bash
python finetuning/contrastive_finetune.py \
    --base_model out/congestion_classifier.pt \
    --data dataset/congestion/train.arrow \
    --human_labels dataset/congestion/review_samples.json \
    --output out/congestion_classifier_refined.pt \
    --epochs 5
```

---

## File Summary

| File | Purpose |
|------|---------|
| `finetuning/congestion_data_collector.py` | Collect raw inputs + auto-labels |
| `finetuning/train_congestion_classifier.py` | Phase 1 training |
| `finetuning/generate_human_labels.py` | Sample a manual review template directly from the dataset |
| `finetuning/export_active_learning_samples.py` | Export disagreement and uncertain cases for human review |
| `finetuning/contrastive_finetune.py` | Phase 2 fine-tuning |

---

## Testing

### Quick Test: Run Data Collection on Small Sample

```bash
python -c "
from finetuning.congestion_data_collector import main
main()
" 2>&1 | head -50
```

### Test Training on Synthetic Data

```bash
# Create small synthetic dataset
python -c "
import numpy as np
import pyarrow as pa

# Synthetic: 1000 samples, 256 features
inputs = np.random.randint(0, 256, size=(1000, 256), dtype=np.int8)
labels = np.random.randint(0, 2, size=(1000,), dtype=np.int8)

table = pa.table({'inputs': pa.array(inputs.tolist()), 'labels': pa.array(labels.tolist())})
pa.ipc.new_file(table, 'dataset/congestion/test.arrow').write()
print('Created test dataset')
"

# Train
python finetuning/train_congestion_classifier.py \
    --data dataset/congestion/test.arrow \
    --output out/test_classifier.pt \
    --epochs 2
```

---

## Key Concepts

### Label Definition
- **Confident positive**: `expert_makespan - fast_makespan > 3`
- **Confident negative**: `expert_makespan - fast_makespan < 1`
- **Uncertain**: `1 <= diff <= 3`

### Why This Works
1. **Auto-labels are cheap**: Clear solver-gap cases provide weak supervision
2. **Human labels are valuable**: Review the cases the model is uncertain about or disagrees with
3. **Contrastive learns patterns**: Not just threshold, but "failure patterns"

### Input to Classifier
- Raw 256-dim tensor (same as MAPF-GPT observation)
- Contains: cost2go grid + nearby agent positions/goals + previous actions

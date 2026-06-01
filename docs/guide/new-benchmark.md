# Add a New Benchmark

Extend SkillOpt with your own benchmark in ~100 lines of code.

## Overview

To add a benchmark, you need:

1. **Data Loader** — Subclass `SplitDataLoader` to load your split data
2. **Environment Adapter** — Subclass `EnvAdapter` and implement rollout/reflect hooks
3. **Config** — YAML configuration file
4. **Registration** — Add your adapter to the train script registry

## Step 1: Create the Benchmark Package

```bash
mkdir -p skillopt/envs/my_benchmark
touch skillopt/envs/my_benchmark/__init__.py
```

## Step 2: Implement the Data Loader

Create `skillopt/envs/my_benchmark/dataloader.py`:

```python
from skillopt.datasets.base import SplitDataLoader


class MyBenchmarkDataLoader(SplitDataLoader):
    """Load benchmark items from raw data and/or split directories."""

    def load_raw_items(self, data_path: str) -> list[dict]:
        # For ratio mode, parse your source dataset from data_path.
        # Return list[dict] where each item has at least a stable "id".
        return super().load_raw_items(data_path)

    def load_split_items(self, split_path: str) -> list[dict]:
        # For split_dir mode, parse one split directory.
        return super().load_split_items(split_path)
```

## Step 3: Implement the Environment Adapter

Create `skillopt/envs/my_benchmark/adapter.py`:

```python
from skillopt.envs.base import EnvAdapter
from skillopt.envs.my_benchmark.dataloader import MyBenchmarkDataLoader

class MyBenchmarkAdapter(EnvAdapter):
    def __init__(self, split_dir: str = "", data_path: str = "", **kwargs):
        self.dataloader = MyBenchmarkDataLoader(split_dir=split_dir, data_path=data_path, **kwargs)

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        return self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs).payload

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        return self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs).payload

    def rollout(self, env_manager, skill_content: str, out_dir: str, **kwargs) -> list[dict]:
        # Run target model on each item in env_manager and return list[dict].
        # Required keys per row: "id", "hard" (0/1), "soft" (0.0-1.0)
        raise NotImplementedError

    def reflect(self, results: list[dict], skill_content: str, out_dir: str, **kwargs) -> list[dict | None]:
        # Convert failure/success analysis into RawPatch-like dicts.
        raise NotImplementedError

    def get_task_types(self) -> list[str]:
        return ["my_benchmark"]
```

## Step 4: Register the Benchmark

Add your adapter to `_register_builtins()` in `scripts/train.py`:

```python
from skillopt.envs.my_benchmark.adapter import MyBenchmarkAdapter

_ENV_REGISTRY["my_benchmark"] = MyBenchmarkAdapter
```

## Step 5: Create Config

Create `configs/my_benchmark/default.yaml`:

```yaml
_base_: ../_base_/default.yaml

env:
  name: my_benchmark
  data_path: data/my_benchmark
  split_mode: ratio
  split_ratio: "2:1:7"

train:
  num_epochs: 4
  batch_size: 40

optimizer:
  learning_rate: 4
  lr_scheduler: cosine
  use_slow_update: true
  use_meta_skill: true

gradient:
  analyst_workers: 16
```

## Step 6: Run

```bash
python scripts/train.py --config configs/my_benchmark/default.yaml
```

## Tips

!!! tip
    - Use a small `batch_size` (10-20) for initial testing
    - Start from `skillopt/envs/_template/` and adapt from there
    - Use an existing adapter (for example `skillopt/envs/officeqa/adapter.py`) as a concrete reference

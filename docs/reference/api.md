# API Reference

## Core Classes

### `EnvAdapter`

Abstract base class for benchmark environments (`skillopt/envs/base.py`).

```python
class EnvAdapter(ABC):
    def setup(self, cfg: dict) -> None
    def get_dataloader(self) -> BaseDataLoader | None
    def build_train_env(self, batch_size: int, seed: int, **kwargs)
    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs)
    def rollout(self, env_manager, skill_content: str, out_dir: str, **kwargs) -> list[dict]
    def reflect(self, results: list[dict], skill_content: str, out_dir: str, **kwargs) -> list[dict | None]
    def get_task_types(self) -> list[str]
```

The rollout contract expects result rows with at least:

```python
{"id": str, "hard": int, "soft": float}
```

### `BaseDataLoader` / `SplitDataLoader`

Data loader abstractions (`skillopt/datasets/base.py`).

```python
class BaseDataLoader(ABC):
    def setup(self, cfg: dict) -> None
    def build_train_batch(self, batch_size: int, seed: int, **kwargs) -> BatchSpec
    def build_eval_batch(self, env_num: int, split: str, seed: int, **kwargs) -> BatchSpec

class SplitDataLoader(BaseDataLoader):
    def load_raw_items(self, data_path: str) -> list[dict]
    def load_split_items(self, split_path: str) -> list[dict]
    def get_split_items(self, split: str) -> list[dict]
```

### `BatchSpec`

Represents one concrete batch request.

```python
@dataclass(slots=True)
class BatchSpec:
    phase: str
    split: str
    seed: int
    batch_size: int
    payload: object | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### `RolloutResult` / `RawPatch`

Typed helpers for stage I/O in `skillopt/types.py`.

```python
@dataclass
class RolloutResult:
    id: str
    hard: int
    soft: float
    # optional benchmark-specific fields

@dataclass
class RawPatch:
    patch: Patch
    source_type: Literal["failure", "success"] = "failure"
```

For detailed source code, see the [`skillopt/`](https://github.com/microsoft/SkillOpt/tree/main/skillopt) directory.

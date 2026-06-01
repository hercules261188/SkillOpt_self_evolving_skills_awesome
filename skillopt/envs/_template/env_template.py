"""
Benchmark Environment Template
===============================
Copy this file and implement the TODO sections to add a new benchmark.

The EnvAdapter is responsible for:
1. Building train/eval environment payloads
2. Running rollout and returning scored result rows
3. Reflecting on results and returning patch candidates
"""
from __future__ import annotations

from skillopt.datasets.base import BatchSpec
from skillopt.envs._template.loader_template import TemplateBenchmarkDataLoader
from skillopt.envs.base import EnvAdapter


class TemplateBenchmarkAdapter(EnvAdapter):
    """
    Environment adapter for <Your Benchmark Name>.

    Rename this class and implement the abstract methods below.
    """

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        seed: int = 42,
        limit: int = 0,
        **kwargs,
    ) -> None:
        self.dataloader = TemplateBenchmarkDataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )
        # TODO: initialize benchmark-specific runtime options from kwargs

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def rollout(self, env_manager, skill_content: str, out_dir: str, **kwargs) -> list[dict]:
        """
        Run one batch and return list[dict] with at least:
        {"id": str, "hard": int, "soft": float}
        """
        raise NotImplementedError("Implement rollout() for your benchmark")

    def reflect(self, results: list[dict], skill_content: str, out_dir: str, **kwargs) -> list[dict | None]:
        """
        Reflect on rollout results and return patch dicts (or None entries).
        """
        raise NotImplementedError("Implement reflect() for your benchmark")

    def get_task_types(self) -> list[str]:
        return ["your_benchmark"]

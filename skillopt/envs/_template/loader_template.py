"""
Benchmark Data Loader Template
================================
Copy this file and implement the TODO sections to load your benchmark data.

The SplitDataLoader is responsible for:
1. Loading raw data from disk for ratio split mode
2. Loading items from train/val/test directories for split_dir mode
3. Returning list[dict] items used by the training loop
"""
from __future__ import annotations

from skillopt.datasets.base import SplitDataLoader


class TemplateBenchmarkDataLoader(SplitDataLoader):
    """
    Data loader for <Your Benchmark Name>.

    Rename this class and implement the methods below.
    """

    def load_raw_items(self, data_path: str) -> list[dict]:
        """
        Parse raw benchmark data for split_mode="ratio".

        Return a list of normalized item dicts.
        """
        # TODO: customize when your raw source format differs.
        return super().load_raw_items(data_path)

    def load_split_items(self, split_path: str) -> list[dict]:
        """
        Parse one split directory for split_mode="split_dir".

        split_path points to train/, val/, or test/.
        """
        # TODO: customize when each split directory has a custom layout.
        return super().load_split_items(split_path)

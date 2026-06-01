from skillopt.datasets.base import SplitDataLoader
from skillopt.envs._template.env_template import TemplateBenchmarkAdapter
from skillopt.envs._template.loader_template import TemplateBenchmarkDataLoader


def test_template_adapter_is_concrete():
    adapter = TemplateBenchmarkAdapter()
    assert adapter.get_task_types() == ["your_benchmark"]


def test_template_loader_uses_split_dataloader():
    loader = TemplateBenchmarkDataLoader()
    assert isinstance(loader, SplitDataLoader)

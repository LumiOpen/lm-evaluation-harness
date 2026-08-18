import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch


def load_mbpp_utils():
    path = Path(__file__).parents[1] / "lm_eval/tasks/mbpp/utils.py"
    spec = importlib.util.spec_from_file_location("test_mbpp_utils_module", path)
    module = importlib.util.module_from_spec(spec)
    metric = Mock()
    metric.compute.return_value = [{"pass@1": 0.0}]
    evaluate = ModuleType("evaluate")
    evaluate.load = Mock(return_value=metric)
    with patch.dict(sys.modules, {"evaluate": evaluate}):
        spec.loader.exec_module(module)
    return module


def test_extract_code_blocks_preserves_bare_code_identifiers():
    extract = load_mbpp_utils().extract_code_blocks

    assert extract("def square(x):\n    return x * x\n```").startswith("def square")
    assert extract(
        "import math\n\ndef root(x):\n    return math.sqrt(x)\n```"
    ).startswith("import math")


def test_extract_code_blocks_preserves_explicit_fences():
    extract = load_mbpp_utils().extract_code_blocks

    assert extract("```python\ndef square(x):\n    return x * x\n```").startswith(
        "def square"
    )
    assert extract("```\ndef square(x):\n    return x * x\n```").startswith(
        "def square"
    )

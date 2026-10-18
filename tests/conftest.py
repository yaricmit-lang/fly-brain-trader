import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session", autouse=True)
def synthetic_connectome():
    """Build the toy connectome once so a fresh clone can run the whole suite."""
    from flybrain.config import load_config

    out = load_config().path("connectome") / "graph_synthetic.npz"
    if not out.exists():
        spec = importlib.util.spec_from_file_location(
            "synthetic_connectome", ROOT / "scripts" / "synthetic_connectome.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    return out

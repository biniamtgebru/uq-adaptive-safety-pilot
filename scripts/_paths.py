"""Path helper: makes `src/` importable and defines output folders (created on demand)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ARTIFACTS, RESULTS, FIGURES = ROOT / "artifacts", ROOT / "results", ROOT / "figures"
for _d in (ARTIFACTS, RESULTS, FIGURES):
    _d.mkdir(exist_ok=True)

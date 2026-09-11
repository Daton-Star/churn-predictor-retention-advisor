"""Make src/ importable as top-level modules (config, data_prep, explain,
genai_advisor) without installing the project as a package -- matches how
the scripts themselves run (`python src/train_model.py` from repo root).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

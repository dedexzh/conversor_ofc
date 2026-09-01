import sys
from pathlib import Path

_RAIZ_CONVERSOR = Path(__file__).resolve().parents[1]
if str(_RAIZ_CONVERSOR) not in sys.path:
    sys.path.insert(0, str(_RAIZ_CONVERSOR))

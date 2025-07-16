from pathlib import Path

from .utils import NP_FLOATS


def load_sound(path: str | Path, sr: int | None = None) -> NP_FLOATS: ...
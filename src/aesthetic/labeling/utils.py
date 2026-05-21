from pathlib import Path
from typing import Callable, Generator, Optional


def dirwalk(path: Path, cond: Optional[Callable] = None) -> Generator[Path, None, None]:
    """Walk through directory and yield files that meet the condition."""
    for p in path.iterdir():
        if p.is_dir():
            yield from dirwalk(p, cond)
        else:
            if isinstance(cond, Callable):
                if not cond(p):
                    continue
            yield p


def format_time(seconds: float) -> str:
    """Helper to format seconds into H:M:S"""
    if seconds < 0:
        seconds = 0
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

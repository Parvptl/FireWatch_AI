"""Filesystem helpers: path expansion and keeping artifacts out of the Git repository."""

from __future__ import annotations

import os
from pathlib import Path


def expand_path(value: str | os.PathLike) -> Path:
    """Expand ``~`` and environment variables, returning an absolute path."""
    return Path(os.path.expandvars(os.fspath(value))).expanduser().absolute()


def find_git_root(start: str | os.PathLike) -> Path | None:
    """Return the nearest ancestor of ``start`` (inclusive) containing ``.git``, if any."""
    p = Path(start).absolute()
    for candidate in (p, *p.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def project_repo_root() -> Path | None:
    """Git root of the checkout this package is imported from (editable installs)."""
    return find_git_root(Path(__file__).resolve().parent)


def is_within(path: str | os.PathLike, parent: str | os.PathLike) -> bool:
    """True if ``path`` equals or lies inside ``parent`` (after resolving symlinks)."""
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def ensure_outside_repo(path: str | os.PathLike, what: str) -> None:
    """Raise ``ValueError`` if ``path`` is inside the FireWatch-AI Git checkout.

    Checks both the checkout this package lives in and the Git root of the
    current working directory.
    """
    roots = {r for r in (project_repo_root(), find_git_root(Path.cwd())) if r is not None}
    for root in roots:
        if is_within(path, root):
            raise ValueError(
                f"{what} must be outside the Git repository: {path} is inside {root}. "
                "Use a directory such as ~/firewatch_runs instead."
            )

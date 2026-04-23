"""Path helpers used by the command-line tools."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src import config


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root when it is not absolute."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (config.PROJECT_ROOT / candidate).resolve()


def ensure_directory(path: str | Path) -> Path:
    """Create a directory when needed and return its resolved path."""
    directory = resolve_project_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_image_file(path: Path) -> bool:
    """Return True when the path has a supported image extension."""
    return path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS


def list_image_files(directory: str | Path) -> list[Path]:
    """List supported image files directly inside a directory."""
    directory_path = resolve_project_path(directory)
    if not directory_path.exists():
        return []
    return sorted(path for path in directory_path.iterdir() if is_image_file(path))


def as_posix_path(path: str | Path) -> str:
    """Return a POSIX-style path string for YAML files and logs."""
    return Path(path).as_posix()


def existing_paths(paths: Iterable[str | Path]) -> list[Path]:
    """Return the subset of paths that exist on disk."""
    return [resolve_project_path(path) for path in paths if resolve_project_path(path).exists()]


def resolve_model_reference(model: str | Path) -> str:
    """Resolve local model paths while leaving Ultralytics model names untouched."""
    model_text = str(model)
    path = Path(model_text).expanduser()

    looks_like_path = path.is_absolute() or len(path.parts) > 1 or model_text.startswith(".")
    if looks_like_path:
        return str(resolve_project_path(path))
    return model_text

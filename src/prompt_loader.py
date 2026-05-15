
"""
Prompt version management.

Conventions
-----------
- Prompt files live in <repo_root>/prompts/ and are named  v<semver>.yaml
  e.g.  prompts/v1.0.0.yaml,  prompts/v1.1.0.yaml
- The "latest" alias resolves to the highest semver present on disk.
- All files are validated against PromptConfig on load so bad YAML surfaces
  immediately rather than at classification time.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from .models import PromptConfig


_REPO_ROOT   = Path(__file__).resolve().parent.parent   # …/regression-detector/
_PROMPTS_DIR = _REPO_ROOT / "prompts"

_SEMVER_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")




def list_prompts(prompts_dir: Optional[Path] = None) -> list[str]:
    """
    Return all available prompt versions sorted ascending by semver.

    Returns a list of version strings (without the leading 'v'), e.g.
    ['1.0.0', '1.1.0', '2.0.0'].
    """
    directory = prompts_dir or _PROMPTS_DIR
    if not directory.exists():
        return []

    versions: list[tuple[int, int, int]] = []
    for f in directory.glob("v*.yaml"):
        m = _SEMVER_RE.match(f.stem)
        if m:
            versions.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))

    versions.sort()
    return [f"{maj}.{min_}.{pat}" for maj, min_, pat in versions]


def load_prompt(
    version: str = "latest",
    prompts_dir: Optional[Path] = None,
) -> PromptConfig:
    """
    Load and validate a PromptConfig from disk.

    Parameters
    ----------
    version     : semver string ('1.0.0') or the special alias 'latest'
    prompts_dir : override the default prompts directory (useful in tests)

    Raises
    ------
    FileNotFoundError  if the requested version does not exist
    ValueError         if the YAML fails PromptConfig validation
    """
    directory = prompts_dir or _PROMPTS_DIR
    resolved = _resolve_version(version, directory)
    return _load_and_cache(resolved, directory)




def _resolve_version(version: str, directory: Path) -> str:
    if version == "latest":
        available = list_prompts(directory)
        if not available:
            raise FileNotFoundError(
                f"No prompt YAML files found in {directory}"
            )
        return available[-1]   
    return version


@lru_cache(maxsize=32)
def _load_and_cache(version: str, directory: Path) -> PromptConfig:
    """
    LRU-cached loader so repeated calls to load_prompt('1.0.0') within
    the same process hit the filesystem only once.
    """
    path = directory / f"v{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt version '{version}' not found.  "
            f"Expected file: {path}  "
            f"Available: {list_prompts(directory)}"
        )

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(raw)}")

    try:
        return PromptConfig.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Invalid prompt config in {path}: {exc}") from exc


def invalidate_cache() -> None:
    """Clear the LRU cache (useful in tests that write temp YAML files)."""
    _load_and_cache.cache_clear()
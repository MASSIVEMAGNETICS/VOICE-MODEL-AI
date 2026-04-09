"""
RVC Model Manager — discovers, validates and manages .pth / .index files.

Supports:
  - Scanning the models/ directory tree
  - Caching model metadata
  - Downloading pre-trained models (HuBERT, RMVPE)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
CACHE_FILE  = MODELS_DIR / ".model_cache.json"

# ── Pre-trained utility model URLs ───────────────────────────────────────────
PRETRAINED_URLS: Dict[str, str] = {
    "hubert_base.pt": (
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
    ),
    "rmvpe.pt": (
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
    ),
    "fcpe.pt": (
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/fcpe.pt"
    ),
}


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class VoiceModel:
    name: str
    pth_path: str
    index_path: Optional[str] = None
    sample_rate: Optional[int] = None   # 40000 or 48000
    f0_conditioned: bool = True
    version: str = "v2"                 # "v1" or "v2"
    size_mb: float = 0.0
    sha256: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "VoiceModel":
        return VoiceModel(**d)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def _size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 ** 2), 2)


def _detect_sr_from_name(name: str) -> Optional[int]:
    n = name.lower()
    if "48k" in n:
        return 48000
    if "40k" in n:
        return 40000
    if "32k" in n:
        return 32000
    return None


def _detect_version_from_name(name: str) -> str:
    return "v1" if "v1" in name.lower() else "v2"


# ── Cache management ─────────────────────────────────────────────────────────

def _load_cache() -> Dict[str, dict]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(data: Dict[str, dict]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Scanning ─────────────────────────────────────────────────────────────────

def scan_models(models_dir: Path = MODELS_DIR) -> List[VoiceModel]:
    """
    Recursively scan *models_dir* for .pth files and pair them with
    matching .index files.

    A match is determined by:
      1. Exact stem match (model.pth ↔ model.index)
      2. Closest .index in the same directory
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    cache = _load_cache()
    models: List[VoiceModel] = []

    pth_files = sorted(models_dir.rglob("*.pth"))

    # Build a directory → [index files] lookup
    dir_indices: Dict[Path, List[Path]] = {}
    for idx in models_dir.rglob("*.index"):
        dir_indices.setdefault(idx.parent, []).append(idx)

    updated = False
    for pth in pth_files:
        key = str(pth.relative_to(models_dir))
        cached = cache.get(key)

        # Check if cached entry is still valid (same mtime)
        mtime = pth.stat().st_mtime
        if cached and cached.get("_mtime") == mtime:
            m = VoiceModel.from_dict({k: v for k, v in cached.items() if not k.startswith("_")})
            models.append(m)
            continue

        # Pair with .index
        index_path: Optional[str] = None
        same_dir_indices = dir_indices.get(pth.parent, [])
        # 1. exact stem
        exact = [i for i in same_dir_indices if i.stem == pth.stem]
        if exact:
            index_path = str(exact[0])
        elif same_dir_indices:
            # pick any .index in the same folder
            index_path = str(same_dir_indices[0])

        m = VoiceModel(
            name=pth.stem,
            pth_path=str(pth),
            index_path=index_path,
            sample_rate=_detect_sr_from_name(pth.stem),
            version=_detect_version_from_name(pth.stem),
            size_mb=_size_mb(pth),
        )
        models.append(m)

        entry = m.to_dict()
        entry["_mtime"] = mtime
        cache[key] = entry
        updated = True

    # Remove stale cache entries
    valid_keys = {str(p.relative_to(models_dir)) for p in pth_files}
    for k in list(cache.keys()):
        if k not in valid_keys:
            del cache[k]
            updated = True

    if updated:
        _save_cache(cache)

    return models


def get_model_names(models_dir: Path = MODELS_DIR) -> List[str]:
    """Return a list of model display names, suitable for a Gradio dropdown."""
    return [m.name for m in scan_models(models_dir)]


def get_model_by_name(name: str, models_dir: Path = MODELS_DIR) -> Optional[VoiceModel]:
    """Look up a VoiceModel by its stem name."""
    for m in scan_models(models_dir):
        if m.name == name:
            return m
    return None


# ── Pre-trained model downloader ─────────────────────────────────────────────

def _download_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)

    def _report(count: int, block: int, total: int) -> None:
        if total > 0:
            pct = min(100, count * block * 100 // total)
            print(f"\r  {dest.name}: {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_report)
    print()  # newline after progress


def ensure_pretrained(name: str, dest_dir: Optional[Path] = None) -> Path:
    """
    Ensure a pre-trained utility model (e.g. hubert_base.pt) is present.

    Downloads from HuggingFace if missing.
    """
    dest_dir = dest_dir or (ROOT / "assets" / "pretrained")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name

    if dest.exists():
        return dest

    if name not in PRETRAINED_URLS:
        raise FileNotFoundError(
            f"No download URL for '{name}'. "
            f"Available: {list(PRETRAINED_URLS.keys())}"
        )

    _download_with_progress(PRETRAINED_URLS[name], dest)
    return dest


def ensure_all_pretrained(dest_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Download all utility models (HuBERT, RMVPE, FCPE) if missing."""
    return {name: ensure_pretrained(name, dest_dir) for name in PRETRAINED_URLS}


# ── Model deletion ────────────────────────────────────────────────────────────

def delete_model(name: str, models_dir: Path = MODELS_DIR) -> bool:
    """Delete a model's .pth (and paired .index if found). Returns True if deleted."""
    model = get_model_by_name(name, models_dir)
    if model is None:
        return False
    pth = Path(model.pth_path)
    if pth.exists():
        pth.unlink()
    if model.index_path:
        idx = Path(model.index_path)
        if idx.exists():
            idx.unlink()
    # Invalidate cache entry
    cache = _load_cache()
    key = str(pth.relative_to(models_dir))
    cache.pop(key, None)
    _save_cache(cache)
    return True


# ── Model duplication / rename ────────────────────────────────────────────────

def duplicate_model(name: str, new_name: str, models_dir: Path = MODELS_DIR) -> Optional[VoiceModel]:
    """Copy a model under a new name."""
    model = get_model_by_name(name, models_dir)
    if model is None:
        return None
    src_pth = Path(model.pth_path)
    dst_pth = src_pth.parent / f"{new_name}.pth"
    shutil.copy2(str(src_pth), str(dst_pth))
    if model.index_path:
        src_idx = Path(model.index_path)
        dst_idx = src_pth.parent / f"{new_name}.index"
        shutil.copy2(str(src_idx), str(dst_idx))
    return scan_models(models_dir)[-1]  # refreshed entry

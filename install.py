"""
Voice Model Studio — Cross-platform installer
Run this script directly: python install.py
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQ_FILE = ROOT / "requirements.txt"
CONFIG_FILE = ROOT / "configs" / "default.json"
PYTHON_MIN = (3, 10)

# ── Rich-style console helpers (no external deps required here) ──────────────

def _c(code: str, text: str) -> str:
    """Wrap text in ANSI colour if supported."""
    if platform.system() == "Windows" and not os.environ.get("WT_SESSION"):
        return text
    return f"\033[{code}m{text}\033[0m"

def info(msg: str) -> None:  print(_c("36", f"[INFO]  {msg}"))
def ok(msg: str) -> None:    print(_c("32", f"[ OK ]  {msg}"))
def warn(msg: str) -> None:  print(_c("33", f"[WARN]  {msg}"))
def err(msg: str) -> None:   print(_c("31", f"[ERR ]  {msg}"), file=sys.stderr)
def header(msg: str) -> None:
    print("\n" + _c("35;1", "━" * 60))
    print(_c("35;1", f"  {msg}"))
    print(_c("35;1", "━" * 60) + "\n")


# ── Python version check ─────────────────────────────────────────────────────

def check_python() -> None:
    v = sys.version_info
    if v < PYTHON_MIN:
        err(f"Python {PYTHON_MIN[0]}.{PYTHON_MIN[1]}+ is required. "
            f"You have {v.major}.{v.minor}.")
        sys.exit(1)
    ok(f"Python {v.major}.{v.minor}.{v.micro} detected.")


# ── Virtual-environment management ──────────────────────────────────────────

def create_venv(force: bool = False) -> None:
    if VENV_DIR.exists() and not force:
        ok("Virtual environment already exists — skipping.")
        return
    info("Creating virtual environment …")
    venv.create(str(VENV_DIR), with_pip=True, clear=force)
    ok(f"Virtual environment created at {VENV_DIR}")


def _venv_python() -> str:
    if platform.system() == "Windows":
        return str(VENV_DIR / "Scripts" / "python.exe")
    return str(VENV_DIR / "bin" / "python")


def _venv_pip() -> str:
    if platform.system() == "Windows":
        return str(VENV_DIR / "Scripts" / "pip.exe")
    return str(VENV_DIR / "bin" / "pip")


# ── Package installation ─────────────────────────────────────────────────────

def install_packages(extras: Optional[list[str]] = None) -> None:
    pip = _venv_pip()
    info("Upgrading pip …")
    subprocess.check_call([pip, "install", "--upgrade", "pip", "setuptools", "wheel"])

    info("Installing requirements …")
    subprocess.check_call([pip, "install", "-r", str(REQ_FILE)])
    ok("All packages installed.")

    if extras:
        info(f"Installing extra packages: {extras}")
        subprocess.check_call([pip, "install"] + extras)


# ── CUDA / GPU detection ─────────────────────────────────────────────────────

def detect_cuda() -> Optional[str]:
    """Return CUDA version string if available, else None."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            out = subprocess.check_output(
                ["nvcc", "--version"], stderr=subprocess.DEVNULL, text=True
            )
            for line in out.splitlines():
                if "release" in line.lower():
                    # e.g. "Cuda compilation tools, release 12.1, ..."
                    parts = line.split(",")
                    for part in parts:
                        part = part.strip()
                        if part.startswith("release"):
                            return part.split()[-1]
        except Exception:
            pass
    return None


def install_torch_cuda(cuda_ver: str) -> None:
    """Install the CUDA-enabled torch wheel."""
    # Map detected CUDA version to PyTorch index
    cuda_short = cuda_ver.replace(".", "")[:3]  # "12.1" → "121"
    index_map = {
        "118": "https://download.pytorch.org/whl/cu118",
        "121": "https://download.pytorch.org/whl/cu121",
        "124": "https://download.pytorch.org/whl/cu124",
    }
    index_url = index_map.get(cuda_short, index_map["121"])
    pip = _venv_pip()
    info(f"Installing CUDA {cuda_ver} torch from {index_url} …")
    subprocess.check_call([
        pip, "install",
        "torch", "torchaudio", "torchvision",
        "--index-url", index_url,
    ])
    ok(f"CUDA-enabled PyTorch installed.")


# ── FFmpeg check ─────────────────────────────────────────────────────────────

def check_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        ok("ffmpeg detected in PATH.")
    else:
        warn("ffmpeg not found in PATH.")
        if platform.system() == "Windows":
            warn("Download ffmpeg from https://www.gyan.dev/ffmpeg/builds/ and add it to PATH.")
        elif platform.system() == "Darwin":
            warn("Install ffmpeg via:  brew install ffmpeg")
        else:
            warn("Install ffmpeg via:  sudo apt install ffmpeg  (or your distro's package manager)")


# ── Directory scaffolding ────────────────────────────────────────────────────

def create_dirs() -> None:
    dirs = [
        ROOT / "models",
        ROOT / "outputs",
        ROOT / "uploads",
        ROOT / "logs",
        ROOT / "assets",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    ok("Project directories ready.")


# ── Write a simple .env file ─────────────────────────────────────────────────

def write_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        return
    env_path.write_text(
        "# Voice Model Studio — environment overrides\n"
        "# Uncomment and edit as needed\n"
        "# HOST=0.0.0.0\n"
        "# PORT=7860\n"
        "# SHARE=false\n"
        "# GPU_ID=0\n",
        encoding="utf-8",
    )
    ok(".env file created (edit to customise).")


# ── Main entry point ─────────────────────────────────────────────────────────

def main(force: bool = False, skip_torch_cuda: bool = False) -> None:
    header("Voice Model Studio — Installer")

    check_python()
    create_dirs()

    cuda = detect_cuda()
    if cuda:
        ok(f"CUDA {cuda} detected — GPU acceleration available.")
    else:
        warn("No CUDA detected — CPU-only mode.")

    create_venv(force=force)

    if cuda and not skip_torch_cuda:
        install_torch_cuda(cuda)

    install_packages()
    check_ffmpeg()
    write_env()

    header("Installation complete!")
    py = _venv_python()
    print(f"  Run the studio with:\n\n    {py} app.py\n")
    print("  Or use the provided start script:\n")
    if platform.system() == "Windows":
        print("    start.bat\n")
    else:
        print("    ./start.sh\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voice Model Studio installer")
    parser.add_argument("--force", action="store_true", help="Recreate virtual environment")
    parser.add_argument("--cpu-only", action="store_true", dest="cpu_only",
                        help="Skip CUDA torch installation")
    args = parser.parse_args()
    main(force=args.force, skip_torch_cuda=args.cpu_only)

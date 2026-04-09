"""
Audio processing utilities for Voice Model Studio.

Handles loading, converting, normalising and saving audio in all
common formats using soundfile, librosa, pydub and ffmpeg.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Optional imports — graceful degradation when heavy deps absent
try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

try:
    import noisereduce as nr
    HAS_NR = True
except ImportError:
    HAS_NR = False

SUPPORTED_FORMATS = {
    "wav", "mp3", "flac", "ogg", "aac", "m4a",
    "wma", "aiff", "opus", "webm", "mp4",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_librosa() -> None:
    if not HAS_LIBROSA:
        raise ImportError("librosa and soundfile are required. Run: pip install librosa soundfile")


def _sanitize_path(path: str | Path) -> str:
    """
    Return the resolved string form of *path*.

    Raises ValueError if the resolved path points outside the filesystem root
    or if the argument contains a null byte (which is a path-injection signal).
    """
    s = str(path)
    if "\x00" in s:
        raise ValueError("Path contains a null byte and is rejected.")
    return str(Path(s).resolve())


def _run_ffmpeg(args: list[str]) -> None:
    """
    Run an ffmpeg command with *args*, raising RuntimeError on failure.

    All arguments must already be strings; shell=False prevents injection.
    """
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + [str(a) for a in args]
    result = subprocess.run(cmd, capture_output=True, shell=False)  # nosec B603
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed: {result.stderr.decode(errors='replace')}"
        )


# ── Loading ──────────────────────────────────────────────────────────────────

def load_audio(
    path: str | Path,
    sr: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file from *any* supported format.

    Returns
    -------
    audio : np.ndarray  — float32 samples, shape (samples,) or (channels, samples)
    sr    : int         — actual sample rate used
    """
    path = Path(_sanitize_path(path))
    suffix = path.suffix.lower().lstrip(".")

    if suffix in ("wav", "flac", "ogg"):
        _ensure_librosa()
        audio, actual_sr = librosa.load(str(path), sr=sr, mono=mono)
        return audio, actual_sr

    # For exotic formats, convert to WAV via ffmpeg first
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        _run_ffmpeg(["-i", str(path), "-ar", str(sr), "-ac", "1" if mono else "2", tmp_path])
        _ensure_librosa()
        audio, actual_sr = librosa.load(tmp_path, sr=sr, mono=mono)
        return audio, actual_sr
    finally:
        os.unlink(tmp_path)


# ── Saving ───────────────────────────────────────────────────────────────────

def save_audio(
    audio: np.ndarray,
    path: str | Path,
    sr: int = 44100,
    fmt: Optional[str] = None,
    bitrate: str = "320k",
) -> Path:
    """
    Save a numpy audio array to *any* supported format.

    Saves as WAV first, then converts via ffmpeg for non-WAV targets.

    Returns the resolved output path.
    """
    _ensure_librosa()
    path = Path(_sanitize_path(path))
    fmt = fmt or path.suffix.lower().lstrip(".")

    # Always write a temp WAV first
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, sr, subtype="PCM_16")

        if fmt == "wav":
            import shutil as _shutil
            path.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(tmp_path, str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            extra: list[str] = []
            if fmt == "mp3":
                extra = ["-b:a", bitrate]
            elif fmt in ("ogg", "opus"):
                extra = ["-b:a", bitrate]
            _run_ffmpeg(["-i", tmp_path] + extra + [str(path)])
    finally:
        os.unlink(tmp_path)

    return path


# ── Normalisation ────────────────────────────────────────────────────────────

def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize audio to *target_peak* amplitude."""
    peak = np.abs(audio).max()
    if peak < 1e-6:
        return audio
    return (audio / peak * target_peak).astype(np.float32)


def normalize_rms(audio: np.ndarray, target_rms_db: float = -18.0) -> np.ndarray:
    """RMS-normalize audio to *target_rms_db* dBFS."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    target_rms = 10 ** (target_rms_db / 20)
    return (audio * (target_rms / rms)).astype(np.float32)


# ── Noise reduction ──────────────────────────────────────────────────────────

def denoise_audio(
    audio: np.ndarray,
    sr: int,
    prop_decrease: float = 0.75,
    stationary: bool = False,
) -> np.ndarray:
    """
    Reduce background noise using noisereduce.

    Parameters
    ----------
    audio           : float32 numpy array
    sr              : sample rate
    prop_decrease   : noise reduction strength 0–1
    stationary      : use stationary noise reduction
    """
    if not HAS_NR:
        raise ImportError("noisereduce is required: pip install noisereduce")
    reduced = nr.reduce_noise(
        y=audio,
        sr=sr,
        prop_decrease=prop_decrease,
        stationary=stationary,
    )
    return reduced.astype(np.float32)


# ── Resampling ────────────────────────────────────────────────────────────────

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio from *orig_sr* to *target_sr*."""
    _ensure_librosa()
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


# ── Silence trimming ─────────────────────────────────────────────────────────

def trim_silence(
    audio: np.ndarray,
    sr: int,
    top_db: int = 30,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trim leading/trailing silence.

    Returns (trimmed_audio, index_array).
    """
    _ensure_librosa()
    trimmed, idx = librosa.effects.trim(
        audio, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    return trimmed, idx


# ── Format conversion helper ──────────────────────────────────────────────────

def convert_audio_format(
    input_path: str | Path,
    output_path: str | Path,
    output_sr: Optional[int] = None,
) -> Path:
    """
    Convert audio file to a different format (any → any) via ffmpeg.

    Parameters
    ----------
    input_path  : source file (any supported format)
    output_path : destination file (format inferred from extension)
    output_sr   : optional target sample rate
    """
    input_path  = Path(_sanitize_path(input_path))
    output_path = Path(_sanitize_path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    args: list[str] = ["-i", str(input_path)]
    if output_sr:
        args += ["-ar", str(output_sr)]
    args.append(str(output_path))

    _run_ffmpeg(args)
    return output_path


# ── Audio information ─────────────────────────────────────────────────────────

def get_audio_info(path: str | Path) -> dict:
    """Return basic info (duration, sr, channels) of an audio file."""
    _ensure_librosa()
    path = Path(path)
    y, sr = librosa.load(str(path), sr=None, mono=False)
    channels = 1 if y.ndim == 1 else y.shape[0]
    duration = y.shape[-1] / sr
    peak_db = float(20 * np.log10(np.abs(y).max() + 1e-9))
    rms_db  = float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9))
    return {
        "path": str(path),
        "duration_s": round(duration, 3),
        "sample_rate": sr,
        "channels": channels,
        "peak_dBFS": round(peak_db, 2),
        "rms_dBFS": round(rms_db, 2),
    }


# ── Pitch shifting (for preview, not RVC) ────────────────────────────────────

def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift audio pitch by *semitones* semitones."""
    _ensure_librosa()
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones).astype(np.float32)


# ── Time stretching ──────────────────────────────────────────────────────────

def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Stretch (or compress) audio in time by *rate* factor."""
    _ensure_librosa()
    return librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)


# ── Bytes ↔ numpy ─────────────────────────────────────────────────────────────

def bytes_to_audio(data: bytes, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Decode raw audio bytes to a numpy array."""
    _ensure_librosa()
    buf = io.BytesIO(data)
    audio, actual_sr = librosa.load(buf, sr=sr, mono=True)
    return audio, actual_sr


def audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Encode a numpy array to WAV bytes."""
    buf = io.BytesIO()
    import soundfile as sf
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()

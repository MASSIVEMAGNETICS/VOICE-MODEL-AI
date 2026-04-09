"""
RVC Model Training Pipeline.

Handles:
  - Dataset preparation (audio slicing, feature extraction)
  - Training the RVC model
  - Monitoring and checkpointing

This module orchestrates the standard RVC training pipeline:
  1. Audio pre-processing and slicing
  2. F0 extraction
  3. HuBERT / ContentVec feature extraction
  4. Model training (VC or VC+F0)
  5. FAISS index building

Usage
-----
    from modules.rvc.training import TrainingConfig, RVCTrainer

    cfg = TrainingConfig(
        model_name="MyVoice",
        dataset_dir="datasets/MyVoice",
        sample_rate=40000,
        epochs=100,
    )
    trainer = RVCTrainer(cfg)
    trainer.prepare_dataset()
    trainer.train()
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
LOGS_DIR   = ROOT / "logs"


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    model_name: str
    dataset_dir: str

    # Audio settings
    sample_rate: int   = 40000          # 32000 / 40000 / 48000
    hop_length: int    = 160

    # Training hyper-parameters
    epochs: int        = 100
    batch_size: int    = 4
    save_every: int    = 10
    cache_all: bool    = True
    fp16: bool         = True
    g_lr: float        = 1e-4
    d_lr: float        = 1e-4

    # F0
    f0_method: str     = "rmvpe"       # rmvpe / harvest / dio / crepe
    f0_conditioned: bool = True

    # Architecture
    version: str       = "v2"          # "v1" or "v2"
    gpu_id: str        = "0"

    # Feature extraction
    feature_dim: int   = 768           # 256 for v1, 768 for v2

    # Output
    output_dir: str    = ""            # defaults to models/<model_name>/

    # Callbacks
    on_epoch_end: Optional[Callable[[int, dict], None]] = field(default=None, repr=False)
    on_log: Optional[Callable[[str], None]]             = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(MODELS_DIR / self.model_name)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("on_epoch_end", None)
        d.pop("on_log", None)
        return d


# ── Dataset preparation ───────────────────────────────────────────────────────

class DatasetPrep:
    """
    Converts raw audio files into the format expected by RVC training:
      - Resamples to target sample rate
      - Slices into segments
      - Extracts F0
      - Extracts content features via HuBERT / ContentVec
    """

    SEGMENT_SECONDS = 15        # max clip length
    SILENCE_THRESHOLD = -42     # dBFS

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.src_dir   = Path(cfg.dataset_dir)
        self.work_dir  = Path(cfg.output_dir)
        self.waves_dir = self.work_dir / "sliced_audios"
        self.feats_dir = self.work_dir / "3_feature"
        self.f0_dir    = self.work_dir / "2a_f0"
        self.f0nsf_dir = self.work_dir / "2b_f0nsf"

    def _log(self, msg: str) -> None:
        logger.info(msg)
        if self.cfg.on_log:
            self.cfg.on_log(msg)

    def prepare(self, progress_cb: Optional[Callable[[float, str], None]] = None) -> None:
        """Full dataset preparation pipeline."""
        self._log("=== Dataset Preparation ===")
        self._make_dirs()
        self._log(f"Source: {self.src_dir}")

        audio_files = self._discover_audio()
        if not audio_files:
            raise FileNotFoundError(
                f"No audio files found in {self.src_dir}. "
                f"Supported: wav, mp3, flac, ogg, m4a"
            )
        self._log(f"Found {len(audio_files)} audio files.")

        # 1. Slice audio
        for i, af in enumerate(audio_files):
            if progress_cb:
                progress_cb(i / len(audio_files) * 0.3, f"Slicing {af.name}")
            self._slice_audio(af)

        # 2. Extract F0
        if self.cfg.f0_conditioned:
            slices = list(self.waves_dir.rglob("*.wav"))
            for i, sl in enumerate(slices):
                if progress_cb:
                    progress_cb(0.3 + i / len(slices) * 0.3, f"F0: {sl.name}")
                self._extract_f0(sl)

        # 3. Extract content features
        slices = list(self.waves_dir.rglob("*.wav"))
        for i, sl in enumerate(slices):
            if progress_cb:
                progress_cb(0.6 + i / len(slices) * 0.4, f"Features: {sl.name}")
            self._extract_features(sl)

        self._log("Dataset preparation complete.")
        if progress_cb:
            progress_cb(1.0, "Done")

    def _make_dirs(self) -> None:
        for d in [self.waves_dir, self.feats_dir, self.f0_dir, self.f0nsf_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _discover_audio(self) -> List[Path]:
        exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        return [p for p in self.src_dir.rglob("*") if p.suffix.lower() in exts]

    def _slice_audio(self, path: Path) -> None:
        """Slice a single audio file into max-SEGMENT_SECONDS clips."""
        try:
            from modules.audio.processing import load_audio, save_audio, normalize_audio
            audio, sr = load_audio(str(path), sr=self.cfg.sample_rate, mono=True)
            audio = normalize_audio(audio)

            seg_len = int(self.SEGMENT_SECONDS * sr)
            n_segs  = max(1, len(audio) // seg_len)

            for i in range(n_segs):
                seg = audio[i * seg_len: (i + 1) * seg_len]
                if len(seg) < sr * 0.5:   # skip very short clips
                    continue
                out = self.waves_dir / f"{path.stem}_{i:04d}.wav"
                save_audio(seg, out, sr=sr)
        except Exception as e:
            logger.warning("Could not slice %s: %s", path, e)

    def _extract_f0(self, path: Path) -> None:
        """Extract and save F0 for a sliced audio file."""
        try:
            from modules.audio.processing import load_audio
            from modules.rvc.inference import F0Extractor

            audio, sr = load_audio(str(path), sr=self.cfg.sample_rate, mono=True)
            extractor = F0Extractor(sr=sr)
            f0 = extractor.extract(audio, method=self.cfg.f0_method)

            import numpy as np
            stem = path.stem
            np.save(str(self.f0_dir    / f"{stem}.npy"), f0)
            np.save(str(self.f0nsf_dir / f"{stem}.npy"), f0)
        except Exception as e:
            logger.warning("F0 extraction failed for %s: %s", path, e)

    def _extract_features(self, path: Path) -> None:
        """Extract HuBERT / ContentVec features for a sliced audio file."""
        try:
            import torch
            import numpy as np
            from modules.audio.processing import load_audio

            audio, sr = load_audio(str(path), sr=16000, mono=True)  # HuBERT always 16 kHz
            try:
                from modules.rvc.feature_extract import HubertFeatureExtractor
                from modules.rvc.model_manager import ensure_pretrained, ASSETS_DIR
                hubert_path = ensure_pretrained("hubert_base.pt")
                device = torch.device("cpu")
                extractor = HubertFeatureExtractor(str(hubert_path), device)
                feats = extractor.extract(audio, sr)  # (1, T, D)
                np.save(str(self.feats_dir / f"{path.stem}.npy"), feats[0].cpu().numpy())
            except Exception as e:
                logger.warning("Feature extraction unavailable: %s", e)
        except Exception as e:
            logger.warning("Feature extraction failed for %s: %s", path, e)


# ── FAISS index builder ────────────────────────────────────────────────────────

def build_index(
    features_dir: str | Path,
    output_path: str | Path,
    n_ivf_clusters: int = 512,
) -> Path:
    """
    Build a FAISS IVF index from extracted feature .npy files.

    Parameters
    ----------
    features_dir    : directory of .npy feature files (shape T x D each)
    output_path     : where to write the .index file
    n_ivf_clusters  : number of IVF clusters
    """
    try:
        import faiss
        import numpy as np
    except ImportError:
        raise ImportError("faiss-cpu and numpy are required: pip install faiss-cpu numpy")

    features_dir = Path(features_dir)
    output_path  = Path(output_path)

    npy_files = sorted(features_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy feature files found in {features_dir}")

    all_feats = np.concatenate([np.load(str(f)) for f in npy_files], axis=0)
    all_feats = all_feats.astype(np.float32)

    dim = all_feats.shape[1]
    logger.info("Building FAISS index: %d vectors × %d dims", len(all_feats), dim)

    quantizer = faiss.IndexFlatL2(dim)
    k = min(n_ivf_clusters, len(all_feats) // 39)
    k = max(k, 1)
    index = faiss.IndexIVFFlat(quantizer, dim, k, faiss.METRIC_L2)
    index.train(all_feats)
    index.add(all_feats)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    logger.info("FAISS index written to %s (%d vectors)", output_path, index.ntotal)
    return output_path


# ── Trainer ───────────────────────────────────────────────────────────────────

class RVCTrainer:
    """
    Orchestrates the full RVC training pipeline.

    In a production setup this would launch the RVC training script
    as a subprocess.  Here we provide the scaffold that:
      1. Prepares the dataset
      2. Saves a training config
      3. Calls the training entry-point if available
      4. Builds the FAISS index
    """

    def __init__(self, cfg: TrainingConfig):
        self.cfg      = cfg
        self.work_dir = Path(cfg.output_dir)

    def _log(self, msg: str) -> None:
        logger.info(msg)
        if self.cfg.on_log:
            self.cfg.on_log(msg)

    def prepare_dataset(
        self,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> None:
        """Prepare training data (slice, extract F0 + features)."""
        prep = DatasetPrep(self.cfg)
        prep.prepare(progress_cb=progress_cb)

    def save_config(self) -> Path:
        cfg_path = self.work_dir / "train_config.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(self.cfg.to_dict(), indent=2), encoding="utf-8")
        self._log(f"Config saved to {cfg_path}")
        return cfg_path

    def train(
        self,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """
        Run training.

        Returns the path to the final .pth checkpoint.
        """
        self.save_config()
        self._log(f"=== Training '{self.cfg.model_name}' for {self.cfg.epochs} epochs ===")
        self._log(f"  SR={self.cfg.sample_rate}  batch={self.cfg.batch_size}  GPU={self.cfg.gpu_id}")

        # Attempt to invoke the rvc-python train script if available
        pth_out = self._run_training(progress_cb)

        # Build FAISS index
        feats_dir = self.work_dir / "3_feature"
        if feats_dir.exists():
            index_out = MODELS_DIR / f"{self.cfg.model_name}.index"
            try:
                build_index(feats_dir, index_out)
                self._log(f"FAISS index → {index_out}")
            except Exception as e:
                self._log(f"[WARN] Index build failed: {e}")

        return pth_out

    def _run_training(self, progress_cb) -> Path:
        """
        Try to run the actual training process.

        If the full RVC training dependencies are not present, this saves
        a placeholder checkpoint so the rest of the pipeline can continue.
        """
        out_pth = MODELS_DIR / f"{self.cfg.model_name}.pth"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            import torch
            # Check if a train entry-point is available
            train_script = ROOT / "lib" / "rvc" / "train.py"
            if train_script.exists():
                self._log("Running external RVC training script …")
                cmd = [
                    sys.executable, str(train_script),
                    "--model-name", self.cfg.model_name,
                    "--dataset-dir", self.cfg.dataset_dir,
                    "--output-dir", self.cfg.output_dir,
                    "--sr", str(self.cfg.sample_rate),
                    "--epochs", str(self.cfg.epochs),
                    "--batch-size", str(self.cfg.batch_size),
                    "--f0-method", self.cfg.f0_method,
                    "--gpu", self.cfg.gpu_id,
                    "--version", self.cfg.version,
                ]
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    raise RuntimeError("Training script exited with error")
            else:
                # Simulate training: create a minimal checkpoint so downstream
                # code sees a valid (but identity) model.
                self._log("[INFO] Full training engine not found — creating placeholder checkpoint.")
                self._log("[INFO] Install rvc-python for full training support.")
                checkpoint = {
                    "weight": {},
                    "config": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], 512, 1, [16, 16, 4, 4, 4], self.cfg.feature_dim, self.cfg.sample_rate],
                    "epoch":  self.cfg.epochs,
                    "f0":     int(self.cfg.f0_conditioned),
                    "version": self.cfg.version,
                    "model_name": self.cfg.model_name,
                    "sr": self.cfg.sample_rate,
                }
                torch.save(checkpoint, str(out_pth))
                self._log(f"Placeholder checkpoint written to {out_pth}")

        except ImportError:
            self._log("[WARN] torch not available — skipping model checkpoint creation.")

        return out_pth

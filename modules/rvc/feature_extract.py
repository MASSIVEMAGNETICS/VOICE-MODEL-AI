"""
HuBERT / ContentVec feature extractor for RVC.

Wraps the HuBERT model (loaded from a .pt checkpoint) and extracts
768-dimensional content features at 50 fps from 16 kHz audio.

If the checkpoint is not available this module degrades gracefully:
the pipeline will use a zero-tensor placeholder so the rest of the
code can still run (useful for testing / CI without heavy weights).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torchaudio
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class HubertFeatureExtractor:
    """
    Extracts HuBERT / ContentVec soft content features from 16 kHz mono audio.

    Parameters
    ----------
    model_path  : path to hubert_base.pt or contentvec_base.pt
    device      : torch device
    """

    TARGET_SR = 16000

    def __init__(self, model_path: Union[str, Path], device: "torch.device"):
        if not HAS_TORCH:
            raise ImportError("torch is required: pip install torch torchaudio")

        self.device = device
        self.model  = self._load(Path(model_path))
        self.model.eval()
        self.model.to(device)

    def _load(self, path: Path) -> "nn.Module":
        """Load checkpoint — supports fairseq, torchhub and bare state-dict formats."""
        import torch

        logger.info("Loading HuBERT from %s", path)
        cpt = torch.load(str(path), map_location="cpu", weights_only=False)

        # ── fairseq format ───────────────────────────────────────────────────
        if isinstance(cpt, dict) and "model" in cpt:
            try:
                import fairseq
                models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                    [str(path)], suffix=""
                )
                model = models[0]
                model.eval()
                return model
            except Exception as e:
                logger.warning("fairseq load failed (%s), trying torchhub …", e)

        # ── torchhub / HuggingFace-style ─────────────────────────────────────
        try:
            model = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True)
            # load weights if the checkpoint looks like a state-dict
            if isinstance(cpt, dict) and not ("model" in cpt or "cfg" in cpt):
                model.load_state_dict(cpt, strict=False)
            return model
        except Exception as e:
            logger.warning("torchhub HuBERT load failed (%s) — using zero extractor", e)
            return _ZeroExtractor()

    @torch.no_grad()
    def extract(self, audio: np.ndarray, sr: int) -> "torch.Tensor":
        """
        Extract features from *audio* (1-D float32 numpy array).

        Returns
        -------
        features : torch.Tensor of shape (1, T, 768)
        """
        import torch

        # Resample to 16 kHz if needed
        if sr != self.TARGET_SR:
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, self.TARGET_SR)
            audio = audio_t[0].numpy()

        wav = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)  # (1, N)

        # Try the standard fairseq / HuBERT interface first
        try:
            # fairseq interface
            feats, _ = self.model.extract_features(
                wav, padding_mask=torch.zeros_like(wav, dtype=torch.bool),
                output_layer=9,
            )
            return feats  # (1, T, 768)
        except (AttributeError, TypeError):
            pass

        try:
            # bshall/hubert-soft interface
            return self.model(wav)  # (1, T, 256)
        except Exception:
            pass

        # fallback to zero features
        n_frames = max(1, audio.shape[0] // 320)
        return torch.zeros(1, n_frames, 768, device=self.device)


class _ZeroExtractor(torch.nn.Module if HAS_TORCH else object):
    """Dummy extractor that returns zeros (used when model weights are absent)."""

    def __init__(self):
        if HAS_TORCH:
            super().__init__()

    def forward(self, wav):  # type: ignore[override]
        import torch
        n_frames = max(1, wav.shape[-1] // 320)
        return torch.zeros(wav.shape[0], n_frames, 768, device=wav.device)

    def extract_features(self, source, padding_mask=None, output_layer=None):
        import torch
        n_frames = max(1, source.shape[-1] // 320)
        feats = torch.zeros(source.shape[0], n_frames, 768, device=source.device)
        return feats, None

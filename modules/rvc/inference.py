"""
RVC Voice Conversion Inference Pipeline.

Supports:
  - Loading RVC v1 / v2 .pth model files (SynthesizerTrnMs / SynthesizerTrnMsNSFsid variants)
  - F0 extraction: rmvpe, crepe, harvest, dio, pm
  - FAISS-based voice feature retrieval via .index files
  - GPU and CPU inference
  - Batch-processing and streaming-ready design

Usage
-----
    from modules.rvc.inference import RVCInferencePipeline

    pipe = RVCInferencePipeline()
    pipe.load_model("MyVoice")
    output_audio = pipe.convert(
        input_audio_path="input.wav",
        f0_method="rmvpe",
        transpose=0,
        index_rate=0.75,
    )
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import librosa
    import soundfile as sf
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import pyworld
    HAS_PYWORLD = True
except ImportError:
    HAS_PYWORLD = False

try:
    import crepe
    HAS_CREPE = True
except ImportError:
    HAS_CREPE = False

ROOT       = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT / "assets" / "pretrained"
MODELS_DIR = ROOT / "models"

F0Method = Literal["rmvpe", "crepe", "harvest", "dio", "pm"]


# ── Utility functions ─────────────────────────────────────────────────────────

def _ensure_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("torch is required for RVC inference. pip install torch torchaudio")

def _ensure_librosa() -> None:
    if not HAS_LIBROSA:
        raise ImportError("librosa and soundfile are required. pip install librosa soundfile")


def _get_device(gpu_id: str = "0") -> "torch.device":
    _ensure_torch()
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_pth(path: str | Path, device: "torch.device") -> dict:
    """Load a .pth checkpoint safely."""
    _ensure_torch()
    return torch.load(str(path), map_location=device, weights_only=False)


# ── F0 Extraction ─────────────────────────────────────────────────────────────

class F0Extractor:
    """
    Extracts fundamental frequency (pitch) from audio using the chosen method.
    """

    def __init__(self, sr: int = 16000, hop_length: int = 160):
        self.sr = sr
        self.hop_length = hop_length
        # RMVPE model cache
        self._rmvpe_model = None

    # ─── public API ──────────────────────────────────────────────────────────

    def extract(
        self,
        audio: np.ndarray,
        method: F0Method = "rmvpe",
        f0_min: float = 50.0,
        f0_max: float = 1100.0,
    ) -> np.ndarray:
        """
        Extract F0 contour.

        Returns a 1-D float32 array of per-frame fundamental frequencies in Hz
        (0 for unvoiced frames).
        """
        methods = {
            "rmvpe":   self._extract_rmvpe,
            "crepe":   self._extract_crepe,
            "harvest": self._extract_harvest,
            "dio":     self._extract_dio,
            "pm":      self._extract_pm,
        }
        fn = methods.get(method)
        if fn is None:
            raise ValueError(f"Unknown F0 method: {method!r}. Choose from {list(methods)}")
        return fn(audio, f0_min=f0_min, f0_max=f0_max)

    # ─── individual extractors ────────────────────────────────────────────────

    def _extract_rmvpe(self, audio: np.ndarray, f0_min: float, f0_max: float) -> np.ndarray:
        """RMVPE — most accurate method, requires rmvpe.pt."""
        try:
            from modules.rvc.rmvpe import RMVPE
            if self._rmvpe_model is None:
                rmvpe_path = ASSETS_DIR / "rmvpe.pt"
                if not rmvpe_path.exists():
                    logger.warning("rmvpe.pt not found, falling back to harvest")
                    return self._extract_harvest(audio, f0_min=f0_min, f0_max=f0_max)
                self._rmvpe_model = RMVPE(str(rmvpe_path), is_half=False, device=torch.device("cpu"))
            f0 = self._rmvpe_model.infer_from_audio(audio, thred=0.03)
            return f0.astype(np.float32)
        except Exception as e:
            logger.warning("RMVPE failed (%s), falling back to harvest", e)
            return self._extract_harvest(audio, f0_min=f0_min, f0_max=f0_max)

    def _extract_crepe(self, audio: np.ndarray, f0_min: float, f0_max: float) -> np.ndarray:
        """CREPE — deep-learning pitch tracker."""
        if not HAS_CREPE:
            logger.warning("crepe not installed, falling back to harvest")
            return self._extract_harvest(audio, f0_min=f0_min, f0_max=f0_max)
        _, frequency, confidence, _ = crepe.predict(
            audio, self.sr, model_capacity="full",
            viterbi=True, verbose=0,
            step_size=int(self.hop_length / self.sr * 1000),
        )
        frequency = frequency.astype(np.float32)
        frequency[confidence < 0.5] = 0.0
        return frequency

    def _extract_harvest(self, audio: np.ndarray, f0_min: float, f0_max: float) -> np.ndarray:
        """WORLD Harvest — robust pitch extractor."""
        if not HAS_PYWORLD:
            logger.warning("pyworld not installed, falling back to dio")
            return self._extract_dio(audio, f0_min=f0_min, f0_max=f0_max)
        audio_d = audio.astype(np.float64)
        f0, t = pyworld.harvest(
            audio_d, self.sr,
            f0_floor=f0_min, f0_ceil=f0_max,
            frame_period=self.hop_length / self.sr * 1000,
        )
        f0 = pyworld.stonemask(audio_d, f0, t, self.sr)
        return f0.astype(np.float32)

    def _extract_dio(self, audio: np.ndarray, f0_min: float, f0_max: float) -> np.ndarray:
        """WORLD DIO — fast pitch extractor."""
        if not HAS_PYWORLD:
            return self._extract_pm(audio, f0_min=f0_min, f0_max=f0_max)
        audio_d = audio.astype(np.float64)
        f0, t = pyworld.dio(
            audio_d, self.sr,
            f0_floor=f0_min, f0_ceil=f0_max,
            frame_period=self.hop_length / self.sr * 1000,
        )
        f0 = pyworld.stonemask(audio_d, f0, t, self.sr)
        return f0.astype(np.float32)

    def _extract_pm(self, audio: np.ndarray, f0_min: float, f0_max: float) -> np.ndarray:
        """Parselmouth (Praat) pitch extractor — lightweight fallback."""
        try:
            import parselmouth
            snd = parselmouth.Sound(audio, sampling_frequency=self.sr)
            pitch = snd.to_pitch(
                time_step=self.hop_length / self.sr,
                pitch_floor=f0_min,
                pitch_ceiling=f0_max,
            )
            f0 = np.array([
                pitch.get_value_at_time(t) or 0.0
                for t in pitch.xs()
            ], dtype=np.float32)
            return f0
        except Exception as e:
            logger.warning("pm extractor failed: %s", e)
            # final fallback: zero array
            n_frames = max(1, int(len(audio) / self.hop_length))
            return np.zeros(n_frames, dtype=np.float32)


# ── FAISS retrieval helper ────────────────────────────────────────────────────

class IndexRetriever:
    """
    Wraps a FAISS .index file for voice feature retrieval (the "index_rate" step).
    """

    def __init__(self, index_path: str | Path):
        if not HAS_FAISS:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")
        self.index = faiss.read_index(str(index_path))
        logger.info("Loaded FAISS index with %d vectors", self.index.ntotal)

    def search(self, features: np.ndarray, k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """Return (distances, neighbour_indices) for each row of *features*."""
        features = np.ascontiguousarray(features, dtype=np.float32)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = min(128, self.index.ntotal)
        return self.index.search(features, k)

    def blend(
        self,
        features: np.ndarray,
        index_rate: float = 0.75,
        k: int = 8,
    ) -> np.ndarray:
        """
        Replace *features* with a blend of retrieved neighbours.

        Parameters
        ----------
        features    : (T, D) array of feature vectors
        index_rate  : mixing ratio 0 (no retrieval) – 1 (full retrieval)
        k           : nearest neighbours to average
        """
        if index_rate == 0 or features.shape[0] == 0:
            return features

        _, I = self.search(features, k=k)
        # Retrieve stored vectors
        stored = np.stack([self.index.reconstruct(int(i)) for i in I.flatten()]).reshape(
            features.shape[0], k, -1
        )
        retrieved = stored.mean(axis=1)
        return ((1 - index_rate) * features + index_rate * retrieved).astype(np.float32)


# ── RVC Model Wrapper ─────────────────────────────────────────────────────────

class RVCModel:
    """
    Lightweight wrapper around a loaded RVC .pth checkpoint.

    The actual model architecture (SynthesizerTrnMs*) lives inside the
    checkpoint's "weight" key.  This class handles:
      - Loading the checkpoint
      - Running the VC forward pass
      - Unloading / cleanup
    """

    def __init__(self, pth_path: str | Path, device: "torch.device"):
        _ensure_torch()
        self.device = device
        self.pth_path = Path(pth_path)
        self.net_g = None
        self.tgt_sr: int = 40000
        self.version: str = "v2"
        self.if_f0: bool = True
        self._load()

    def _load(self) -> None:
        logger.info("Loading RVC checkpoint: %s", self.pth_path)
        cpt = _load_pth(self.pth_path, self.device)

        self.tgt_sr   = cpt.get("config", [None] * 17)[16] or 40000
        self.version  = cpt.get("version", "v2")
        self.if_f0    = cpt.get("f0", 1) == 1

        try:
            from modules.rvc.synthesizer import SynthesizerFactory
            self.net_g = SynthesizerFactory.build(cpt, self.device)
            self.net_g.load_state_dict(cpt["weight"], strict=False)
            self.net_g.eval()
            self.net_g.to(self.device)
            logger.info(
                "Model loaded: sr=%d, version=%s, f0=%s",
                self.tgt_sr, self.version, self.if_f0,
            )
        except Exception as e:
            logger.warning(
                "Could not build synthesizer from checkpoint (%s). "
                "Ensure rvc-python dependencies are installed.", e,
            )
            self.net_g = None

    def unload(self) -> None:
        del self.net_g
        self.net_g = None
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ── Main pipeline ─────────────────────────────────────────────────────────────

class RVCInferencePipeline:
    """
    High-level interface for RVC voice conversion.

    Example
    -------
        pipe = RVCInferencePipeline(gpu_id="0")
        pipe.load_model("MyVoice")
        out_path = pipe.convert("input.wav", transpose=0, f0_method="rmvpe")
    """

    def __init__(self, gpu_id: str = "0"):
        _ensure_torch()
        self.device   = _get_device(gpu_id)
        self.model    : Optional[RVCModel]         = None
        self.retriever: Optional[IndexRetriever]   = None
        self.f0_extractor = F0Extractor()
        self._hubert: Optional[object] = None
        logger.info("RVC pipeline on device: %s", self.device)

    # ─── Model loading ────────────────────────────────────────────────────────

    def load_model(
        self,
        model_name: str,
        models_dir: Path = MODELS_DIR,
    ) -> None:
        from modules.rvc.model_manager import get_model_by_name
        vm = get_model_by_name(model_name, models_dir)
        if vm is None:
            raise FileNotFoundError(
                f"Model '{model_name}' not found in {models_dir}. "
                f"Place a .pth file there and try again."
            )
        self._load_from_vm(vm)

    def load_model_from_path(self, pth_path: str | Path, index_path: Optional[str | Path] = None) -> None:
        from modules.rvc.model_manager import VoiceModel
        vm = VoiceModel(
            name=Path(pth_path).stem,
            pth_path=str(pth_path),
            index_path=str(index_path) if index_path else None,
        )
        self._load_from_vm(vm)

    def _load_from_vm(self, vm) -> None:
        if self.model is not None:
            self.model.unload()
        self.model = RVCModel(vm.pth_path, self.device)
        self.retriever = None
        if vm.index_path and HAS_FAISS:
            try:
                self.retriever = IndexRetriever(vm.index_path)
            except Exception as e:
                logger.warning("Could not load FAISS index: %s", e)

    def unload(self) -> None:
        if self.model:
            self.model.unload()
        self.model = None
        self.retriever = None
        self._hubert = None
        gc.collect()

    # ─── HuBERT feature extractor ─────────────────────────────────────────────

    def _get_hubert(self):
        if self._hubert is not None:
            return self._hubert
        try:
            from modules.rvc.feature_extract import HubertFeatureExtractor
            hubert_path = ASSETS_DIR / "hubert_base.pt"
            if not hubert_path.exists():
                raise FileNotFoundError("hubert_base.pt not found. Run: python app.py --download-models")
            self._hubert = HubertFeatureExtractor(str(hubert_path), self.device)
        except Exception as e:
            logger.error("HuBERT not available: %s", e)
            raise
        return self._hubert

    # ─── Conversion ────────────────────────────────────────────────────────────

    def convert(
        self,
        input_audio_path: str | Path,
        output_path: Optional[str | Path] = None,
        f0_method: F0Method = "rmvpe",
        transpose: int = 0,
        index_rate: float = 0.75,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        auto_pitch: bool = False,
    ) -> Path:
        """
        Convert an input audio file using the currently-loaded RVC model.

        Parameters
        ----------
        input_audio_path : path to input wav/mp3/flac/etc.
        output_path      : where to save (defaults to outputs/<name>.wav)
        f0_method        : pitch extraction method
        transpose        : semitone shift (+/- 12 for octave change)
        index_rate       : FAISS retrieval blend 0–1
        filter_radius    : median filter radius for F0 smoothing
        resample_sr      : output sample rate (0 = use model default)
        rms_mix_rate     : blend ratio of input/output RMS
        protect          : protection level for consonants 0–0.5
        auto_pitch       : auto-detect gender and set pitch

        Returns
        -------
        Path to the output wav file.
        """
        _ensure_librosa()

        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        input_audio_path = Path(input_audio_path)
        t0 = time.time()

        # 1. Load and pre-process input audio
        from modules.audio.processing import load_audio, normalize_audio
        audio, sr = load_audio(str(input_audio_path), sr=16000, mono=True)
        audio = normalize_audio(audio)

        if auto_pitch and transpose == 0:
            transpose = self._auto_transpose(audio, sr)
            logger.info("Auto-detected transpose: %+d semitones", transpose)

        # 2. Extract F0
        f0 = self.f0_extractor.extract(audio, method=f0_method)
        if filter_radius > 1:
            from scipy.signal import medfilt
            f0 = medfilt(f0, kernel_size=filter_radius).astype(np.float32)
        f0 = np.nan_to_num(f0)

        # Apply transpose
        if transpose != 0:
            voiced = f0 > 0
            f0[voiced] *= 2 ** (transpose / 12)

        # 3. Convert
        output_audio = self._run_vc(audio, sr, f0, index_rate, rms_mix_rate, protect)

        # 4. Resample if requested
        tgt_sr = self.model.tgt_sr
        if resample_sr and resample_sr != tgt_sr:
            from modules.audio.processing import resample_audio
            output_audio = resample_audio(output_audio, tgt_sr, resample_sr)
            tgt_sr = resample_sr

        # 5. Save
        if output_path is None:
            out_dir = ROOT / "outputs"
            out_dir.mkdir(exist_ok=True)
            output_path = out_dir / f"{input_audio_path.stem}_vc_{int(time.time())}.wav"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import soundfile as sf
        sf.write(str(output_path), output_audio, tgt_sr, subtype="PCM_16")

        elapsed = time.time() - t0
        logger.info("Conversion complete in %.2fs → %s", elapsed, output_path)
        return output_path

    def _run_vc(
        self,
        audio: np.ndarray,
        sr: int,
        f0: np.ndarray,
        index_rate: float,
        rms_mix_rate: float,
        protect: float,
    ) -> np.ndarray:
        """
        Core voice conversion forward pass.

        If the synthesizer model is unavailable (missing deps), returns
        the identity-converted audio (passthrough with pitch shift applied
        via librosa) so the system degrades gracefully.
        """
        _ensure_torch()

        if self.model is None or self.model.net_g is None:
            logger.warning("Synthesizer not available — returning passthrough audio")
            return audio

        with torch.no_grad():
            try:
                # ── Extract content features ──────────────────────────────
                hubert = self._get_hubert()
                feats = hubert.extract(audio, sr)  # (1, T, D)

                # ── FAISS retrieval blending ───────────────────────────────
                if self.retriever and index_rate > 0:
                    feats_np = feats[0].cpu().numpy()
                    feats_np = self.retriever.blend(feats_np, index_rate=index_rate)
                    feats = torch.from_numpy(feats_np).unsqueeze(0).to(self.device)

                # ── Prepare tensors ───────────────────────────────────────
                feats = feats.to(self.device)
                f0_tensor = torch.from_numpy(f0).float().to(self.device)
                f0_coarse = self._f0_to_coarse(f0_tensor)

                # Match lengths
                n_frames = f0.shape[0]
                feats = torch.nn.functional.interpolate(
                    feats.transpose(1, 2), size=n_frames, mode="nearest"
                ).transpose(1, 2)

                lengths = torch.tensor([n_frames], device=self.device)

                # ── Forward ───────────────────────────────────────────────
                if self.model.if_f0:
                    audio_out = self.model.net_g.infer(
                        feats, lengths, f0_coarse, f0_tensor, protect
                    )[0][0, 0].data.cpu().float().numpy()
                else:
                    audio_out = self.model.net_g.infer(
                        feats, lengths, protect
                    )[0][0, 0].data.cpu().float().numpy()

                # ── RMS blending ───────────────────────────────────────────
                if rms_mix_rate < 1.0:
                    in_rms  = np.sqrt(np.mean(audio ** 2)) + 1e-9
                    out_rms = np.sqrt(np.mean(audio_out ** 2)) + 1e-9
                    audio_out *= (in_rms / out_rms) ** (1 - rms_mix_rate)

                return audio_out.astype(np.float32)

            except Exception as e:
                logger.error("VC forward pass failed: %s", e, exc_info=True)
                return audio

    @staticmethod
    def _f0_to_coarse(f0: "torch.Tensor") -> "torch.Tensor":
        """Map continuous F0 to coarse bin indices (256 bins)."""
        f0_mel_min = 1127 * np.log(1 + 50  / 700)
        f0_mel_max = 1127 * np.log(1 + 1100 / 700)
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel = torch.clamp(f0_mel, min=1, max=255)
        f0_mel[f0 == 0] = 0
        return f0_mel.long()

    @staticmethod
    def _auto_transpose(audio: np.ndarray, sr: int) -> int:
        """
        Heuristic: estimate whether audio is male/female and suggest transpose.
        Returns semitone adjustment for male→female conversion (positive) or
        female→male (negative), or 0 if uncertain.
        """
        try:
            import parselmouth
            snd = parselmouth.Sound(audio, sampling_frequency=sr)
            pitch = snd.to_pitch()
            mean_f0 = pitch.get_mean(unit="Hertz")
            if mean_f0 and mean_f0 < 160:  # likely male
                return 12  # shift up an octave
            elif mean_f0 and mean_f0 > 200:  # likely female
                return -12
        except Exception:
            pass
        return 0

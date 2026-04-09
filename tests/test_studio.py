"""
Tests for Voice Model Studio core modules.

These tests verify the module structure and logic without requiring
heavy ML dependencies (torch, librosa etc.) or GPU hardware.
They use graceful-degradation paths to run in any CI environment.
"""

import json
import sys
import types
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Config ────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_config_file_exists(self):
        assert (ROOT / "configs" / "default.json").exists()

    def test_config_keys(self):
        cfg = json.loads((ROOT / "configs" / "default.json").read_text())
        for key in ("app", "inference", "training", "tts", "audio", "paths"):
            assert key in cfg, f"Missing key '{key}' in default.json"

    def test_config_inference_defaults(self):
        cfg = json.loads((ROOT / "configs" / "default.json").read_text())
        inf = cfg["inference"]
        assert inf["default_f0_method"] in ("rmvpe", "crepe", "harvest", "dio", "pm")
        assert -24 <= inf["default_transpose"] <= 24
        assert 0.0 <= inf["default_index_rate"] <= 1.0

    def test_config_tts_defaults(self):
        cfg = json.loads((ROOT / "configs" / "default.json").read_text())
        tts = cfg["tts"]
        assert tts["default_engine"] in ("edge-tts", "gtts", "pyttsx3")
        assert isinstance(tts["default_voice"], str)


# ── App entry point ────────────────────────────────────────────────────────────

class TestApp:
    def test_load_config(self):
        from app import load_config
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "app" in cfg

    def test_env_overrides(self):
        from app import load_config, _apply_env_overrides
        import os
        cfg = load_config()
        os.environ["PORT"] = "9999"
        cfg = _apply_env_overrides(cfg)
        assert cfg["app"]["port"] == 9999
        del os.environ["PORT"]

    def test_parse_args_defaults(self):
        from app import parse_args
        with patch("sys.argv", ["app.py"]):
            args = parse_args()
        assert args.host == ""
        assert args.port == 0
        assert args.share is False

    def test_build_ui(self):
        """UI should build without raising even without torch."""
        import gradio
        from app import load_config, build_ui
        cfg = load_config()
        demo = build_ui(cfg)
        assert demo is not None


# ── Audio processing ──────────────────────────────────────────────────────────

class TestAudioProcessing:
    """Tests that do not require librosa / soundfile (pure numpy logic)."""

    def test_normalize_audio_peak(self):
        from modules.audio.processing import normalize_audio
        audio = np.array([0.0, 0.5, 1.0, -0.8], dtype=np.float32)
        normed = normalize_audio(audio, target_peak=0.95)
        assert abs(np.abs(normed).max() - 0.95) < 1e-5

    def test_normalize_audio_zero(self):
        from modules.audio.processing import normalize_audio
        audio = np.zeros(100, dtype=np.float32)
        normed = normalize_audio(audio)
        assert np.allclose(normed, 0)

    def test_normalize_audio_output_type(self):
        from modules.audio.processing import normalize_audio
        audio = np.random.randn(1000).astype(np.float32)
        normed = normalize_audio(audio)
        assert normed.dtype == np.float32

    def test_normalize_rms(self):
        from modules.audio.processing import normalize_rms
        audio = np.random.randn(16000).astype(np.float32)
        normed = normalize_rms(audio, target_rms_db=-18.0)
        rms_db = 20 * np.log10(np.sqrt(np.mean(normed ** 2)) + 1e-9)
        assert abs(rms_db - (-18.0)) < 0.5  # within 0.5 dB


# ── Model manager ─────────────────────────────────────────────────────────────

class TestModelManager:
    def test_scan_models_empty(self, tmp_path):
        from modules.rvc.model_manager import scan_models
        result = scan_models(tmp_path)
        assert result == []

    def test_scan_models_finds_pth(self, tmp_path):
        from modules.rvc.model_manager import scan_models
        (tmp_path / "MyVoice_40k_v2.pth").write_bytes(b"\x00" * 100)
        models = scan_models(tmp_path)
        assert len(models) == 1
        assert models[0].name == "MyVoice_40k_v2"

    def test_scan_models_pairs_index(self, tmp_path):
        from modules.rvc.model_manager import scan_models
        (tmp_path / "test_model.pth").write_bytes(b"\x00" * 100)
        (tmp_path / "test_model.index").write_bytes(b"\x00" * 50)
        models = scan_models(tmp_path)
        assert models[0].index_path is not None
        assert "test_model.index" in models[0].index_path

    def test_detect_sr_from_name(self):
        from modules.rvc.model_manager import _detect_sr_from_name
        assert _detect_sr_from_name("voice_40k_v2") == 40000
        assert _detect_sr_from_name("voice_48k") == 48000
        assert _detect_sr_from_name("unknown_model") is None

    def test_detect_version_from_name(self):
        from modules.rvc.model_manager import _detect_version_from_name
        assert _detect_version_from_name("model_v1") == "v1"
        assert _detect_version_from_name("model_v2") == "v2"
        assert _detect_version_from_name("model") == "v2"  # default

    def test_get_model_by_name_missing(self, tmp_path):
        from modules.rvc.model_manager import get_model_by_name
        result = get_model_by_name("does_not_exist", tmp_path)
        assert result is None

    def test_voice_model_to_dict(self):
        from modules.rvc.model_manager import VoiceModel
        vm = VoiceModel(name="test", pth_path="/path/to/test.pth")
        d = vm.to_dict()
        assert d["name"] == "test"
        assert d["pth_path"] == "/path/to/test.pth"

    def test_voice_model_from_dict(self):
        from modules.rvc.model_manager import VoiceModel
        d = {"name": "test", "pth_path": "/p.pth", "index_path": None,
             "sample_rate": 40000, "f0_conditioned": True,
             "version": "v2", "size_mb": 0.0, "sha256": None}
        vm = VoiceModel.from_dict(d)
        assert vm.name == "test"
        assert vm.version == "v2"


# ── F0 Extractor (no torch required) ─────────────────────────────────────────

class TestF0Extractor:
    def test_instantiation(self):
        from modules.rvc.inference import F0Extractor
        extractor = F0Extractor(sr=16000, hop_length=160)
        assert extractor.sr == 16000

    def test_extract_pm_fallback(self):
        """pm extractor should return a zero array when parselmouth is absent."""
        from modules.rvc.inference import F0Extractor
        extractor = F0Extractor(sr=16000, hop_length=160)
        audio = np.zeros(16000, dtype=np.float32)
        # Use the internal pm extractor directly (pure python fallback)
        with patch("modules.rvc.inference.HAS_PYWORLD", False):
            f0 = extractor._extract_pm(audio, f0_min=50, f0_max=1100)
        assert isinstance(f0, np.ndarray)
        assert f0.dtype == np.float32

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed",
    )
    def test_f0_to_coarse(self):
        """F0 coarse mapping should stay in [0, 255]."""
        import torch
        from modules.rvc.inference import RVCInferencePipeline
        f0 = torch.tensor([0.0, 100.0, 220.0, 440.0, 880.0, 1100.0])
        coarse = RVCInferencePipeline._f0_to_coarse(f0)
        assert coarse.min() >= 0
        assert coarse.max() <= 255


# ── TTS Synthesizer ────────────────────────────────────────────────────────────

class TestTTSSynthesizer:
    def test_instantiation_edge(self):
        from modules.tts.synthesis import TTSSynthesizer
        s = TTSSynthesizer("edge-tts")
        assert s.engine_name == "edge-tts"

    def test_instantiation_gtts(self):
        from modules.tts.synthesis import TTSSynthesizer
        s = TTSSynthesizer("gtts")
        assert s.engine_name == "gtts"

    def test_instantiation_pyttsx3(self):
        from modules.tts.synthesis import TTSSynthesizer
        s = TTSSynthesizer("pyttsx3")
        assert s.engine_name == "pyttsx3"

    def test_rate_to_edge(self):
        from modules.tts.synthesis import TTSSynthesizer
        assert TTSSynthesizer._rate_to_edge(1.0) == "+0%"
        assert TTSSynthesizer._rate_to_edge(1.5) == "+50%"
        assert TTSSynthesizer._rate_to_edge(0.75) == "-25%"

    def test_invalid_engine(self):
        from modules.tts.synthesis import TTSSynthesizer
        with pytest.raises(ValueError):
            TTSSynthesizer("unknown_engine").synthesize("hello")


# ── Training config ────────────────────────────────────────────────────────────

class TestTrainingConfig:
    def test_defaults(self, tmp_path):
        from modules.rvc.training import TrainingConfig
        cfg = TrainingConfig(model_name="test", dataset_dir=str(tmp_path))
        assert cfg.epochs == 100
        assert cfg.sample_rate == 40000
        assert cfg.version == "v2"
        assert cfg.f0_method == "rmvpe"

    def test_output_dir_default(self, tmp_path):
        from modules.rvc.training import TrainingConfig
        cfg = TrainingConfig(model_name="MyModel", dataset_dir=str(tmp_path))
        assert "MyModel" in cfg.output_dir

    def test_to_dict(self, tmp_path):
        from modules.rvc.training import TrainingConfig
        cfg = TrainingConfig(model_name="x", dataset_dir=str(tmp_path))
        d = cfg.to_dict()
        assert "model_name" in d
        assert "on_epoch_end" not in d   # callbacks excluded from serialisation

    def test_trainer_save_config(self, tmp_path):
        from modules.rvc.training import TrainingConfig, RVCTrainer
        cfg = TrainingConfig(
            model_name="SaveTest",
            dataset_dir=str(tmp_path),
            output_dir=str(tmp_path / "output"),
        )
        trainer = RVCTrainer(cfg)
        cfg_path = trainer.save_config()
        assert cfg_path.exists()
        saved = json.loads(cfg_path.read_text())
        assert saved["model_name"] == "SaveTest"


# ── Dataset preparation (file I/O only) ───────────────────────────────────────

class TestDatasetPrep:
    def test_discover_audio(self, tmp_path):
        from modules.rvc.training import DatasetPrep, TrainingConfig
        # Create dummy audio files
        for name in ["a.wav", "b.mp3", "c.flac", "ignore.txt"]:
            (tmp_path / name).write_bytes(b"\x00" * 100)
        cfg = TrainingConfig(model_name="m", dataset_dir=str(tmp_path))
        prep = DatasetPrep(cfg)
        found = prep._discover_audio()
        names = {f.name for f in found}
        assert "a.wav" in names
        assert "b.mp3" in names
        assert "c.flac" in names
        assert "ignore.txt" not in names


# ── Install script ────────────────────────────────────────────────────────────

class TestInstaller:
    def test_check_python_passes(self):
        """Current Python should pass version check."""
        from install import check_python
        check_python()  # should not raise

    def test_venv_python_path_format(self, tmp_path):
        import platform
        from install import _venv_python, VENV_DIR
        # Patch VENV_DIR temporarily
        import install as inst
        orig = inst.VENV_DIR
        inst.VENV_DIR = tmp_path / ".venv"
        path = _venv_python()
        inst.VENV_DIR = orig
        if platform.system() == "Windows":
            assert "Scripts" in path
        else:
            assert "bin" in path

    def test_create_dirs(self, tmp_path):
        import install as inst
        orig = inst.ROOT
        inst.ROOT = tmp_path
        inst.create_dirs()
        inst.ROOT = orig
        for d in ("models", "outputs", "uploads", "logs", "assets"):
            assert (tmp_path / d).is_dir()


# ── Project structure ─────────────────────────────────────────────────────────

class TestProjectStructure:
    FILES = [
        "app.py",
        "install.py",
        "requirements.txt",
        "setup.bat",
        "setup.sh",
        "start.bat",
        "start.sh",
        "configs/default.json",
        "modules/__init__.py",
        "modules/audio/__init__.py",
        "modules/audio/processing.py",
        "modules/rvc/__init__.py",
        "modules/rvc/inference.py",
        "modules/rvc/training.py",
        "modules/rvc/model_manager.py",
        "modules/rvc/feature_extract.py",
        "modules/rvc/synthesizer.py",
        "modules/rvc/rmvpe.py",
        "modules/tts/__init__.py",
        "modules/tts/synthesis.py",
        "modules/ui/__init__.py",
        "modules/ui/inference_tab.py",
        "modules/ui/tts_tab.py",
        "modules/ui/training_tab.py",
        "modules/ui/audio_tools_tab.py",
        "modules/ui/settings_tab.py",
    ]

    @pytest.mark.parametrize("relpath", FILES)
    def test_file_exists(self, relpath):
        assert (ROOT / relpath).exists(), f"Missing file: {relpath}"


# ── RMVPE ─────────────────────────────────────────────────────────────────────

class TestRMVPE:
    def test_harvest_fallback_zeros(self):
        from modules.rvc.rmvpe import RMVPE
        audio = np.zeros(16000, dtype=np.float32)
        f0 = RMVPE._harvest_fallback(audio)
        assert isinstance(f0, np.ndarray)
        assert f0.dtype == np.float32
        assert len(f0) > 0

    def test_instantiation_missing_model(self, tmp_path):
        from modules.rvc.rmvpe import RMVPE
        # Instantiation with a non-existent path should not raise
        rmvpe = RMVPE(tmp_path / "nonexistent.pt")
        assert rmvpe._ready is False

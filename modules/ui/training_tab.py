"""
Training UI Tab.

Provides a Gradio interface for training a new RVC voice model:
  1. Upload / point to a dataset directory
  2. Configure training hyper-parameters
  3. Run dataset preparation (slicing + F0 + feature extraction)
  4. Run training
  5. Monitor progress via a live log window
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"
DATASETS   = ROOT / "datasets"


# ── Handlers ──────────────────────────────────────────────────────────────────

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-][A-Za-z0-9_\- ]{0,63}$")


def _safe_model_name(name: str) -> str:
    """Raise ValueError if *name* contains unsafe characters."""
    if not _SAFE_NAME_RE.match(name.strip()):
        raise ValueError(
            f"Model name {name!r} contains invalid characters. "
            "Use letters, digits, spaces, hyphens and underscores only."
        )
    return name.strip()


def _safe_directory(path_str: str) -> Path:
    """
    Resolve and validate a user-supplied directory path.

    Raises ValueError for paths that contain null bytes.
    Returns a resolved Path; the caller is responsible for existence checks.
    """
    if "\x00" in path_str:
        raise ValueError("Path contains a null byte.")
    return Path(path_str.strip()).resolve()


def _validate_dataset(dataset_dir: str) -> str:
    if not dataset_dir.strip():
        return "⚠️ Please enter a dataset directory path."
    try:
        p = _safe_directory(dataset_dir)
    except ValueError as e:
        return f"❌ Invalid path: {e}"
    if not p.exists():
        return f"❌ Path does not exist: {p}"
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = [f for f in p.rglob("*") if f.suffix.lower() in audio_exts]
    if not files:
        return f"⚠️ No audio files found in {p}"
    total_s = sum(f.stat().st_size for f in files) / (1024 ** 2)
    return f"✅ Found {len(files)} audio files ({total_s:.1f} MB) in {p}"


def _run_preparation(
    model_name: str,
    dataset_dir: str,
    sample_rate: int,
    f0_method: str,
) -> tuple:
    """Dataset preparation — runs synchronously for simplicity."""
    logs: list[str] = []

    def _log(msg: str) -> None:
        logger.info(msg)
        logs.append(msg)

    try:
        model_name = _safe_model_name(model_name)
    except ValueError as e:
        return f"⚠️ {e}", ""

    try:
        safe_dir = _safe_directory(dataset_dir)
    except ValueError as e:
        return f"⚠️ {e}", ""

    if not safe_dir.exists():
        return "❌ Invalid dataset directory.", ""

    try:
        from modules.rvc.training import TrainingConfig, DatasetPrep

        cfg = TrainingConfig(
            model_name=model_name,
            dataset_dir=str(safe_dir),
            sample_rate=sample_rate,
            f0_method=f0_method,
            on_log=_log,
        )
        prep = DatasetPrep(cfg)
        prep.prepare()
        status = f"✅ Dataset preparation complete for '{model_name}'."
    except Exception as e:
        logger.exception("Dataset prep error")
        status = f"❌ Error: {e}"

    return status, "\n".join(logs)


def _run_training(
    model_name: str,
    dataset_dir: str,
    sample_rate: int,
    epochs: int,
    batch_size: int,
    save_every: int,
    f0_method: str,
    version: str,
    fp16: bool,
    gpu_id: str,
) -> tuple:
    logs: list[str] = []

    def _log(msg: str) -> None:
        logger.info(msg)
        logs.append(msg)

    try:
        model_name = _safe_model_name(model_name)
    except ValueError as e:
        return f"⚠️ {e}", ""

    try:
        safe_dir = _safe_directory(dataset_dir)
    except ValueError as e:
        return f"⚠️ {e}", ""

    if not dataset_dir.strip():
        return "⚠️ Enter a dataset directory.", ""

    try:
        from modules.rvc.training import TrainingConfig, RVCTrainer

        cfg = TrainingConfig(
            model_name=model_name,
            dataset_dir=str(safe_dir),
            sample_rate=int(sample_rate),
            epochs=int(epochs),
            batch_size=int(batch_size),
            save_every=int(save_every),
            f0_method=f0_method,
            version=version,
            fp16=fp16,
            gpu_id=gpu_id,
            on_log=_log,
        )
        trainer = RVCTrainer(cfg)
        trainer.prepare_dataset()
        out_pth = trainer.train()
        status  = f"✅ Training complete → {out_pth}"
    except Exception as e:
        logger.exception("Training error")
        status = f"❌ Error: {e}"

    return status, "\n".join(logs)


# ── Tab builder ───────────────────────────────────────────────────────────────

def build_tab(cfg: dict):
    import gradio as gr

    tr_cfg = cfg.get("training", {})

    with gr.Tab("🏋️ Train Model"):
        gr.Markdown(
            "## 🏋️ Train a Custom Voice Model\n"
            "Provide a directory of audio samples (ideally 10–60 min of clean speech) "
            "and configure the training pipeline."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Dataset")
                model_name_in = gr.Textbox(label="Model Name", placeholder="MyVoice")
                dataset_dir   = gr.Textbox(
                    label="Dataset Directory",
                    placeholder="/path/to/audio/samples",
                )
                validate_btn  = gr.Button("🔍 Validate Dataset", size="sm")
                validate_out  = gr.Textbox(label="Validation", interactive=False, lines=1)

                gr.Markdown("### ⚙️ Audio Settings")
                sr_dd = gr.Dropdown(
                    choices=[32000, 40000, 48000],
                    value=tr_cfg.get("default_sample_rate", 40000),
                    label="Target Sample Rate",
                )
                version_dd = gr.Dropdown(["v2", "v1"], value="v2", label="RVC Version")
                f0_dd = gr.Dropdown(
                    ["rmvpe", "harvest", "dio", "crepe", "pm"],
                    value="rmvpe",
                    label="F0 Extraction Method",
                )

                gr.Markdown("### 🔢 Training Hyper-parameters")
                epochs_sl     = gr.Slider(10, 1000, value=tr_cfg.get("default_epochs", 100),
                                          step=10, label="Epochs")
                batch_sl      = gr.Slider(1, 32, value=tr_cfg.get("default_batch_size", 4),
                                          step=1, label="Batch Size")
                save_every_sl = gr.Slider(5, 100, value=tr_cfg.get("default_save_every_epoch", 10),
                                          step=5, label="Save Checkpoint Every N Epochs")
                fp16_chk      = gr.Checkbox(value=True, label="Use FP16 (faster on GPU)")
                gpu_id        = gr.Textbox(value="0", label="GPU ID")

            with gr.Column(scale=2):
                gr.Markdown("### 🚀 Actions")

                with gr.Row():
                    prep_btn  = gr.Button("📊 Prepare Dataset Only", variant="secondary")
                    train_btn = gr.Button("🚀 Prepare + Train", variant="primary")

                train_status = gr.Textbox(label="Status", interactive=False, lines=1)

                gr.Markdown("### 📜 Training Log")
                log_box = gr.Textbox(
                    label="Log",
                    interactive=False,
                    lines=20,
                    max_lines=100,
                )

                gr.Markdown("### 📋 Tips")
                gr.Markdown(
                    "- **Minimum data**: ~10 min of clean, consistent-quality audio.\n"
                    "- **Best results**: 30–60 min, single speaker, no background noise.\n"
                    "- **Epochs**: 100–300 for most voices. Use more for complex voices.\n"
                    "- **GPU**: Training is much faster with a CUDA GPU.\n"
                    "- Models are saved to `models/<ModelName>.pth` and `.index`.\n"
                )

        # ── Events ────────────────────────────────────────────────────────────

        validate_btn.click(
            fn=_validate_dataset,
            inputs=dataset_dir,
            outputs=validate_out,
        )

        prep_btn.click(
            fn=_run_preparation,
            inputs=[model_name_in, dataset_dir, sr_dd, f0_dd],
            outputs=[train_status, log_box],
        )

        train_btn.click(
            fn=_run_training,
            inputs=[
                model_name_in, dataset_dir, sr_dd,
                epochs_sl, batch_sl, save_every_sl,
                f0_dd, version_dd, fp16_chk, gpu_id,
            ],
            outputs=[train_status, log_box],
        )

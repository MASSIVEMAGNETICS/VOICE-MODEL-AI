"""
Voice Conversion (Inference) UI Tab.

Provides the Gradio interface for:
  - Loading a voice model (.pth + optional .index)
  - Uploading / recording input audio
  - Adjusting conversion parameters
  - Running RVC inference and playing back / downloading the result
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"
OUTPUTS    = ROOT / "outputs"


# ── Lazy pipeline singleton ───────────────────────────────────────────────────

_pipeline: Optional[Any] = None   # RVCInferencePipeline


def _get_pipeline(gpu_id: str = "0"):
    global _pipeline
    if _pipeline is None:
        try:
            from modules.rvc.inference import RVCInferencePipeline
            _pipeline = RVCInferencePipeline(gpu_id=gpu_id)
        except Exception as e:
            logger.warning("Could not create RVC pipeline: %s", e)
            _pipeline = None
    return _pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scan_models() -> list[str]:
    try:
        from modules.rvc.model_manager import get_model_names
        names = get_model_names()
        return names if names else ["(no models found — add .pth files to models/)"]
    except Exception:
        return ["(no models found)"]


def _load_model(model_name: str, gpu_id: str = "0") -> str:
    if not model_name or model_name.startswith("("):
        return "⚠️ No model selected."
    pipe = _get_pipeline(gpu_id)
    if pipe is None:
        return "⚠️ RVC pipeline unavailable (check torch installation)."
    try:
        pipe.load_model(model_name)
        return f"✅ Model '{model_name}' loaded successfully."
    except FileNotFoundError as e:
        return f"❌ {e}"
    except Exception as e:
        logger.exception("Model load error")
        return f"❌ Error: {e}"


def _load_model_from_file(pth_file, index_file, gpu_id: str = "0") -> str:
    if pth_file is None:
        return "⚠️ Please upload a .pth model file."
    pth_path   = pth_file.name if hasattr(pth_file, "name") else str(pth_file)
    index_path = index_file.name if (index_file and hasattr(index_file, "name")) else None
    pipe = _get_pipeline(gpu_id)
    if pipe is None:
        return "⚠️ RVC pipeline unavailable."
    try:
        pipe.load_model_from_path(pth_path, index_path)
        return f"✅ Custom model loaded from {Path(pth_path).name}"
    except Exception as e:
        logger.exception("Custom model load error")
        return f"❌ Error: {e}"


def _run_inference(
    audio_input,
    model_name: str,
    f0_method: str,
    transpose: int,
    index_rate: float,
    filter_radius: int,
    resample_sr: int,
    rms_mix_rate: float,
    protect: float,
    auto_pitch: bool,
    output_format: str,
    gpu_id: str,
):
    """Core inference handler called by Gradio."""
    if audio_input is None:
        return None, "⚠️ Please upload or record input audio."

    pipe = _get_pipeline(gpu_id)
    if pipe is None:
        return None, "⚠️ RVC pipeline unavailable (install torch)."

    # If no model loaded yet, try to load selected
    if pipe.model is None:
        msg = _load_model(model_name, gpu_id)
        if "❌" in msg or "⚠️" in msg:
            return None, msg

    input_path = audio_input if isinstance(audio_input, str) else audio_input[1] if isinstance(audio_input, tuple) else str(audio_input)

    OUTPUTS.mkdir(exist_ok=True)
    out_name   = f"vc_{Path(str(input_path)).stem}_{int(time.time())}.{output_format}"
    out_path   = OUTPUTS / out_name

    try:
        result = pipe.convert(
            input_audio_path=input_path,
            output_path=out_path,
            f0_method=f0_method,
            transpose=int(transpose),
            index_rate=float(index_rate),
            filter_radius=int(filter_radius),
            resample_sr=int(resample_sr),
            rms_mix_rate=float(rms_mix_rate),
            protect=float(protect),
            auto_pitch=bool(auto_pitch),
        )
        return str(result), f"✅ Done → {result.name}"
    except Exception as e:
        logger.exception("Inference error")
        return None, f"❌ Error: {e}"


# ── Tab builder ───────────────────────────────────────────────────────────────

def build_tab(cfg: dict):
    """Build and return the Gradio Voice Conversion tab."""
    import gradio as gr

    inf_cfg = cfg.get("inference", {})

    with gr.Tab("🎙️ Voice Conversion"):
        gr.Markdown(
            "## 🎙️ Voice Conversion (RVC)\n"
            "Upload audio, select a voice model, tweak parameters, and convert."
        )

        with gr.Row():
            # ── Left column: Model ────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📦 Model")

                with gr.Tabs():
                    with gr.Tab("From Library"):
                        model_dd = gr.Dropdown(
                            choices=_scan_models(),
                            label="Voice Model",
                            info="Models in models/ directory",
                        )
                        refresh_btn = gr.Button("🔄 Refresh", size="sm")
                        load_status = gr.Textbox(label="Status", interactive=False, lines=1)
                        load_btn    = gr.Button("Load Model", variant="primary")

                    with gr.Tab("Upload Custom"):
                        pth_upload   = gr.File(label="Model (.pth)", file_types=[".pth"])
                        index_upload = gr.File(label="Index (.index) — optional", file_types=[".index"])
                        upload_status = gr.Textbox(label="Status", interactive=False, lines=1)
                        upload_btn  = gr.Button("Load Uploaded Model", variant="primary")

                gr.Markdown("### ⚙️ Parameters")
                f0_method = gr.Dropdown(
                    choices=["rmvpe", "crepe", "harvest", "dio", "pm"],
                    value=inf_cfg.get("default_f0_method", "rmvpe"),
                    label="Pitch (F0) Method",
                )
                transpose = gr.Slider(
                    -24, 24,
                    value=inf_cfg.get("default_transpose", 0),
                    step=1,
                    label="Transpose (semitones)",
                )
                auto_pitch = gr.Checkbox(
                    value=inf_cfg.get("auto_pitch", False),
                    label="Auto Pitch (detect gender & adjust)",
                )
                index_rate = gr.Slider(
                    0, 1,
                    value=inf_cfg.get("default_index_rate", 0.75),
                    step=0.01,
                    label="Index Rate (FAISS retrieval blend)",
                )
                with gr.Accordion("Advanced", open=False):
                    filter_radius = gr.Slider(
                        0, 7,
                        value=inf_cfg.get("default_filter_radius", 3),
                        step=1,
                        label="F0 Filter Radius",
                    )
                    resample_sr = gr.Dropdown(
                        choices=[0, 16000, 22050, 32000, 40000, 44100, 48000],
                        value=inf_cfg.get("default_resample_sr", 0),
                        label="Resample Output SR (0 = model default)",
                    )
                    rms_mix_rate = gr.Slider(
                        0, 1,
                        value=inf_cfg.get("default_rms_mix_rate", 0.25),
                        step=0.01,
                        label="RMS Mix Rate",
                    )
                    protect = gr.Slider(
                        0, 0.5,
                        value=inf_cfg.get("default_protect", 0.33),
                        step=0.01,
                        label="Protect Consonants",
                    )
                output_format = gr.Dropdown(
                    choices=["wav", "mp3", "flac", "ogg"],
                    value="wav",
                    label="Output Format",
                )
                gpu_id = gr.Textbox(value="0", label="GPU ID (0 = default, empty = CPU)")

            # ── Right column: Audio ───────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 🎵 Audio")
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Input Audio",
                )
                convert_btn = gr.Button("🚀 Convert", variant="primary", size="lg")
                inf_status  = gr.Textbox(label="Status", interactive=False, lines=1)
                audio_output = gr.Audio(label="Converted Audio", interactive=False)

        # ── Event handlers ────────────────────────────────────────────────────

        refresh_btn.click(
            fn=lambda: gr.update(choices=_scan_models()),
            outputs=model_dd,
        )

        load_btn.click(
            fn=lambda name, gpu: _load_model(name, gpu),
            inputs=[model_dd, gpu_id],
            outputs=load_status,
        )

        upload_btn.click(
            fn=lambda pth, idx, gpu: _load_model_from_file(pth, idx, gpu),
            inputs=[pth_upload, index_upload, gpu_id],
            outputs=upload_status,
        )

        convert_btn.click(
            fn=_run_inference,
            inputs=[
                audio_input, model_dd,
                f0_method, transpose, index_rate,
                filter_radius, resample_sr, rms_mix_rate,
                protect, auto_pitch, output_format, gpu_id,
            ],
            outputs=[audio_output, inf_status],
        )

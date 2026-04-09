"""
Audio Tools UI Tab.

Provides standalone audio utilities:
  - Format converter (any → any via ffmpeg)
  - Noise reduction
  - Pitch shifter / time stretcher
  - Audio info viewer
  - Batch processor
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[3]
OUTPUTS = ROOT / "outputs"


# ── Handlers ──────────────────────────────────────────────────────────────────

def _get_info(audio_file) -> str:
    if audio_file is None:
        return "Upload an audio file to see its information."
    path = audio_file if isinstance(audio_file, str) else getattr(audio_file, "name", str(audio_file))
    try:
        from modules.audio.processing import get_audio_info
        info = get_audio_info(path)
        lines = [
            f"**File:** {info['path']}",
            f"**Duration:** {info['duration_s']:.2f} s",
            f"**Sample Rate:** {info['sample_rate']} Hz",
            f"**Channels:** {info['channels']}",
            f"**Peak:** {info['peak_dBFS']} dBFS",
            f"**RMS:** {info['rms_dBFS']} dBFS",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {e}"


def _convert_format(audio_file, out_format: str, out_sr: int) -> tuple:
    if audio_file is None:
        return None, "⚠️ No file uploaded."
    path = audio_file if isinstance(audio_file, str) else getattr(audio_file, "name", str(audio_file))
    OUTPUTS.mkdir(exist_ok=True)
    out_path = OUTPUTS / f"converted_{int(time.time())}.{out_format}"
    try:
        from modules.audio.processing import convert_audio_format
        result = convert_audio_format(path, out_path, output_sr=out_sr if out_sr else None)
        return str(result), f"✅ Converted → {result.name}"
    except Exception as e:
        logger.exception("Format conversion error")
        return None, f"❌ Error: {e}"


def _denoise(audio_file, prop_decrease: float) -> tuple:
    if audio_file is None:
        return None, "⚠️ No file uploaded."
    path = audio_file if isinstance(audio_file, str) else getattr(audio_file, "name", str(audio_file))
    OUTPUTS.mkdir(exist_ok=True)
    out_path = OUTPUTS / f"denoised_{int(time.time())}.wav"
    try:
        from modules.audio.processing import load_audio, save_audio, denoise_audio
        audio, sr = load_audio(path, sr=None, mono=False)  # type: ignore[arg-type]
        audio_d = denoise_audio(audio, sr, prop_decrease=prop_decrease)
        save_audio(audio_d, out_path, sr=sr)
        return str(out_path), f"✅ Denoised → {out_path.name}"
    except Exception as e:
        logger.exception("Denoise error")
        return None, f"❌ Error: {e}"


def _pitch_and_stretch(audio_file, semitones: float, stretch_rate: float) -> tuple:
    if audio_file is None:
        return None, "⚠️ No file uploaded."
    path = audio_file if isinstance(audio_file, str) else getattr(audio_file, "name", str(audio_file))
    OUTPUTS.mkdir(exist_ok=True)
    out_path = OUTPUTS / f"pitched_{int(time.time())}.wav"
    try:
        from modules.audio.processing import load_audio, save_audio, pitch_shift, time_stretch
        audio, sr = load_audio(path, sr=None)  # type: ignore[arg-type]
        if semitones != 0:
            audio = pitch_shift(audio, sr, semitones)
        if stretch_rate != 1.0:
            audio = time_stretch(audio, stretch_rate)
        save_audio(audio, out_path, sr=sr)
        return str(out_path), f"✅ Done → {out_path.name}"
    except Exception as e:
        logger.exception("Pitch/stretch error")
        return None, f"❌ Error: {e}"


def _normalize(audio_file, target_peak: float) -> tuple:
    if audio_file is None:
        return None, "⚠️ No file uploaded."
    path = audio_file if isinstance(audio_file, str) else getattr(audio_file, "name", str(audio_file))
    OUTPUTS.mkdir(exist_ok=True)
    out_path = OUTPUTS / f"normalized_{int(time.time())}.wav"
    try:
        from modules.audio.processing import load_audio, save_audio, normalize_audio
        audio, sr = load_audio(path, sr=None)  # type: ignore[arg-type]
        audio = normalize_audio(audio, target_peak=target_peak)
        save_audio(audio, out_path, sr=sr)
        return str(out_path), f"✅ Normalized → {out_path.name}"
    except Exception as e:
        return None, f"❌ Error: {e}"


# ── Tab builder ───────────────────────────────────────────────────────────────

def build_tab(cfg: dict):
    import gradio as gr

    audio_cfg = cfg.get("audio", {})

    with gr.Tab("🛠️ Audio Tools"):
        gr.Markdown(
            "## 🛠️ Audio Tools\n"
            "Standalone audio processing utilities — "
            "format conversion, denoising, pitch shift, normalization."
        )

        # ── Audio Info ─────────────────────────────────────────────────────────
        with gr.Accordion("📊 Audio Information", open=True):
            with gr.Row():
                info_audio = gr.Audio(sources=["upload"], type="filepath", label="Audio File")
                info_out   = gr.Markdown("Upload a file to inspect.")
            info_audio.change(fn=_get_info, inputs=info_audio, outputs=info_out)

        # ── Format Converter ───────────────────────────────────────────────────
        with gr.Accordion("🔄 Format Converter", open=True):
            with gr.Row():
                conv_in   = gr.Audio(sources=["upload"], type="filepath", label="Input Audio")
                with gr.Column():
                    conv_fmt  = gr.Dropdown(
                        choices=["wav", "mp3", "flac", "ogg", "aac", "opus"],
                        value="wav", label="Output Format",
                    )
                    conv_sr   = gr.Dropdown(
                        choices=[0, 16000, 22050, 32000, 44100, 48000],
                        value=0, label="Output Sample Rate (0 = keep original)",
                    )
                    conv_btn  = gr.Button("Convert", variant="primary")
                    conv_status = gr.Textbox(label="Status", interactive=False)
                    conv_out    = gr.Audio(label="Converted", interactive=False)

            conv_btn.click(
                fn=_convert_format,
                inputs=[conv_in, conv_fmt, conv_sr],
                outputs=[conv_out, conv_status],
            )

        # ── Noise Reduction ────────────────────────────────────────────────────
        with gr.Accordion("🔇 Noise Reduction", open=False):
            with gr.Row():
                nr_in   = gr.Audio(sources=["upload"], type="filepath", label="Input Audio")
                with gr.Column():
                    nr_prop = gr.Slider(0.1, 1.0, value=0.75, step=0.05,
                                        label="Reduction Strength")
                    nr_btn  = gr.Button("Denoise", variant="primary")
                    nr_status = gr.Textbox(label="Status", interactive=False)
                    nr_out  = gr.Audio(label="Denoised", interactive=False)

            nr_btn.click(
                fn=_denoise,
                inputs=[nr_in, nr_prop],
                outputs=[nr_out, nr_status],
            )

        # ── Pitch Shift / Time Stretch ─────────────────────────────────────────
        with gr.Accordion("🎚️ Pitch Shift & Time Stretch", open=False):
            with gr.Row():
                ps_in     = gr.Audio(sources=["upload"], type="filepath", label="Input Audio")
                with gr.Column():
                    semitones  = gr.Slider(-24, 24, value=0, step=0.5, label="Pitch Shift (semitones)")
                    stretch_r  = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Time Stretch Rate")
                    ps_btn     = gr.Button("Apply", variant="primary")
                    ps_status  = gr.Textbox(label="Status", interactive=False)
                    ps_out     = gr.Audio(label="Output", interactive=False)

            ps_btn.click(
                fn=_pitch_and_stretch,
                inputs=[ps_in, semitones, stretch_r],
                outputs=[ps_out, ps_status],
            )

        # ── Normalize ──────────────────────────────────────────────────────────
        with gr.Accordion("📶 Normalize Audio", open=False):
            with gr.Row():
                norm_in    = gr.Audio(sources=["upload"], type="filepath", label="Input Audio")
                with gr.Column():
                    peak_sl    = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="Target Peak")
                    norm_btn   = gr.Button("Normalize", variant="primary")
                    norm_status = gr.Textbox(label="Status", interactive=False)
                    norm_out   = gr.Audio(label="Normalized", interactive=False)

            norm_btn.click(
                fn=_normalize,
                inputs=[norm_in, peak_sl],
                outputs=[norm_out, norm_status],
            )

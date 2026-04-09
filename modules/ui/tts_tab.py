"""
TTS (Text-to-Speech) UI Tab.

Features:
  - Engine selection: edge-tts / gTTS / pyttsx3
  - Voice and language selection
  - Rate, pitch and volume controls
  - One-click: TTS → RVC chaining
  - Audio playback & download
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[3]
OUTPUTS = ROOT / "outputs"

# ── Edge-TTS default voices ───────────────────────────────────────────────────

EDGE_VOICE_PRESETS = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-DavisNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
    "en-CA-ClaraNeural",
    "es-ES-ElviraNeural",
    "es-MX-DaliaNeural",
    "fr-FR-DeniseNeural",
    "de-DE-KatjaNeural",
    "it-IT-ElsaNeural",
    "pt-BR-FranciscaNeural",
    "ru-RU-SvetlanaNeural",
    "ja-JP-NanamiNeural",
    "ko-KR-SunHiNeural",
    "zh-CN-XiaoxiaoNeural",
    "ar-EG-SalmaNeural",
    "hi-IN-SwaraNeural",
]

GTTS_LANGUAGES = {
    "English (US)": "en",
    "English (UK)": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese (Mandarin)": "zh-CN",
    "Arabic": "ar",
    "Hindi": "hi",
}


# ── Handlers ──────────────────────────────────────────────────────────────────

def _fetch_edge_voices(lang_prefix: str) -> list[str]:
    """Try to fetch live voice list from edge-tts."""
    try:
        from modules.tts.synthesis import TTSSynthesizer
        synth  = TTSSynthesizer("edge-tts")
        voices = synth.list_voices(lang_prefix if lang_prefix.strip() else None)
        if voices:
            return [v.get("ShortName", "") for v in voices if v.get("ShortName")]
    except Exception as e:
        logger.warning("edge-tts voice fetch failed: %s", e)
    return EDGE_VOICE_PRESETS


def _run_tts(
    engine: str,
    text: str,
    edge_voice: str,
    gtts_lang: str,
    pyttsx3_voice: str,
    rate: float,
    pitch: int,
    volume: float,
    slow: bool,
    output_format: str,
    apply_rvc: bool,
    rvc_model: str,
    rvc_transpose: int,
    rvc_f0_method: str,
    gpu_id: str,
):
    if not text.strip():
        return None, None, "⚠️ Please enter some text."

    OUTPUTS.mkdir(exist_ok=True)
    tts_path = OUTPUTS / f"tts_{int(time.time())}.{output_format}"

    try:
        from modules.tts.synthesis import TTSSynthesizer
        synth = TTSSynthesizer(engine)

        if engine == "edge-tts":
            out = synth.synthesize(
                text, voice=edge_voice or "en-US-AriaNeural",
                output_path=tts_path,
                rate=float(rate), pitch=int(pitch), volume=float(volume),
            )
        elif engine == "gtts":
            out = synth.synthesize(
                text, language=gtts_lang or "en",
                output_path=tts_path, slow=bool(slow),
            )
        else:  # pyttsx3
            out = synth.synthesize(
                text, voice=pyttsx3_voice or "",
                output_path=tts_path,
                rate=float(rate), volume=float(volume),
            )

        status = f"✅ TTS done → {out.name}"
        rvc_audio = None

        if apply_rvc and rvc_model and not rvc_model.startswith("("):
            from modules.rvc.inference import RVCInferencePipeline
            rvc_out = OUTPUTS / f"tts_rvc_{int(time.time())}.wav"
            pipe = RVCInferencePipeline(gpu_id=gpu_id)
            pipe.load_model(rvc_model)
            rvc_result = pipe.convert(
                str(out),
                output_path=rvc_out,
                f0_method=rvc_f0_method,
                transpose=int(rvc_transpose),
            )
            rvc_audio = str(rvc_result)
            status += f" | RVC done → {rvc_result.name}"

        return str(out), rvc_audio, status

    except Exception as e:
        logger.exception("TTS error")
        return None, None, f"❌ Error: {e}"


def _scan_rvc_models() -> list[str]:
    try:
        from modules.rvc.model_manager import get_model_names
        names = get_model_names()
        return ["(none)"] + names
    except Exception:
        return ["(none)"]


# ── Tab builder ───────────────────────────────────────────────────────────────

def build_tab(cfg: dict):
    import gradio as gr

    tts_cfg = cfg.get("tts", {})
    default_engine = tts_cfg.get("default_engine", "edge-tts")
    default_voice  = tts_cfg.get("default_voice", "en-US-AriaNeural")

    with gr.Tab("🗣️ Text-to-Speech"):
        gr.Markdown(
            "## 🗣️ Text-to-Speech\n"
            "Convert text to natural-sounding speech using multiple engines. "
            "Optionally chain through an RVC voice model."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔧 Engine & Voice")

                engine_dd = gr.Dropdown(
                    choices=["edge-tts", "gtts", "pyttsx3"],
                    value=default_engine,
                    label="TTS Engine",
                    info="edge-tts: best quality (online) | pyttsx3: offline",
                )

                with gr.Group():
                    gr.Markdown("**Edge-TTS Voice**")
                    edge_voice = gr.Dropdown(
                        choices=EDGE_VOICE_PRESETS,
                        value=default_voice,
                        label="Voice (Edge-TTS)",
                        allow_custom_value=True,
                    )
                    lang_filter = gr.Textbox(
                        value="en",
                        label="Language Filter (e.g. en, es, fr)",
                        max_lines=1,
                    )
                    refresh_voices_btn = gr.Button("🔄 Refresh Voices", size="sm")

                with gr.Group():
                    gr.Markdown("**Google TTS**")
                    gtts_lang = gr.Dropdown(
                        choices=list(GTTS_LANGUAGES.values()),
                        value="en",
                        label="Language (gTTS)",
                    )
                    slow_chk = gr.Checkbox(value=False, label="Slow Mode")

                with gr.Group():
                    gr.Markdown("**pyttsx3 (Offline)**")
                    pyttsx3_voice = gr.Textbox(
                        value="",
                        label="Voice ID (leave blank for default)",
                        max_lines=1,
                    )

                gr.Markdown("### 🎛️ Audio Settings")
                rate_sl   = gr.Slider(0.5, 2.0, value=tts_cfg.get("default_speed", 1.0),
                                      step=0.05, label="Speed (rate)")
                pitch_sl  = gr.Slider(-50, 50, value=tts_cfg.get("default_pitch", 0),
                                      step=1, label="Pitch (Hz, edge-tts only)")
                volume_sl = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Volume")
                fmt_dd    = gr.Dropdown(["wav", "mp3", "flac", "ogg"], value="wav", label="Output Format")

                gr.Markdown("### 🎙️ RVC Chain (optional)")
                apply_rvc    = gr.Checkbox(value=False, label="Apply RVC After TTS")
                rvc_model_dd = gr.Dropdown(choices=_scan_rvc_models(), value="(none)", label="RVC Voice Model")
                rvc_transpose_sl = gr.Slider(-24, 24, value=0, step=1, label="RVC Transpose")
                rvc_f0_dd    = gr.Dropdown(["rmvpe","harvest","dio","crepe","pm"],
                                           value="rmvpe", label="RVC F0 Method")
                gpu_id       = gr.Textbox(value="0", label="GPU ID")

            with gr.Column(scale=2):
                gr.Markdown("### ✍️ Text Input")
                text_in = gr.Textbox(
                    label="Text to Speak",
                    placeholder="Enter your text here …",
                    lines=8,
                )
                char_counter = gr.Markdown("0 characters")
                tts_btn = gr.Button("🗣️ Synthesize", variant="primary", size="lg")
                status  = gr.Textbox(label="Status", interactive=False, lines=1)

                gr.Markdown("### 🎵 Output")
                tts_audio = gr.Audio(label="TTS Audio", interactive=False)
                rvc_audio = gr.Audio(label="RVC-Converted Audio", interactive=False)

        # ── Events ────────────────────────────────────────────────────────────

        text_in.change(
            fn=lambda t: f"{len(t)} characters",
            inputs=text_in,
            outputs=char_counter,
        )

        refresh_voices_btn.click(
            fn=lambda lf: gr.update(choices=_fetch_edge_voices(lf)),
            inputs=lang_filter,
            outputs=edge_voice,
        )

        tts_btn.click(
            fn=_run_tts,
            inputs=[
                engine_dd, text_in, edge_voice, gtts_lang, pyttsx3_voice,
                rate_sl, pitch_sl, volume_sl, slow_chk, fmt_dd,
                apply_rvc, rvc_model_dd, rvc_transpose_sl, rvc_f0_dd, gpu_id,
            ],
            outputs=[tts_audio, rvc_audio, status],
        )

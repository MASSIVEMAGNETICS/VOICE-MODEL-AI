"""
Settings UI Tab.

Lets users configure application-level settings:
  - Default paths
  - GPU settings
  - Audio defaults
  - Model auto-download
  - Application theme
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[3]
CONFIG_FILE = ROOT / "configs" / "default.json"


def _load_cfg() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cfg(cfg: dict) -> str:
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return "✅ Settings saved."
    except Exception as e:
        return f"❌ Save failed: {e}"


def _sys_info() -> str:
    lines = [
        f"**Platform:** {platform.platform()}",
        f"**Python:** {platform.python_version()}",
        f"**ffmpeg:** {'found' if shutil.which('ffmpeg') else 'NOT FOUND'}",
    ]
    try:
        import torch
        lines.append(f"**PyTorch:** {torch.__version__}")
        if torch.cuda.is_available():
            lines.append(f"**CUDA:** {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem   = props.total_memory // (1024 ** 3)
                lines.append(f"  GPU {i}: {props.name} ({mem} GB VRAM)")
        else:
            lines.append("**CUDA:** Not available (CPU mode)")
    except ImportError:
        lines.append("**PyTorch:** Not installed")

    try:
        import gradio as gr
        lines.append(f"**Gradio:** {gr.__version__}")
    except ImportError:
        lines.append("**Gradio:** Not installed")

    try:
        import edge_tts
        lines.append(f"**edge-tts:** installed")
    except ImportError:
        lines.append("**edge-tts:** Not installed")

    try:
        import faiss
        lines.append(f"**faiss:** installed")
    except ImportError:
        lines.append("**faiss:** Not installed")

    return "\n".join(lines)


def _download_pretrained() -> str:
    msgs = []
    try:
        from modules.rvc.model_manager import ensure_all_pretrained
        result = ensure_all_pretrained()
        for name, path in result.items():
            msgs.append(f"✅ {name} → {path}")
    except Exception as e:
        msgs.append(f"❌ Error: {e}")
    return "\n".join(msgs)


def build_tab(cfg: dict):
    import gradio as gr

    app_cfg  = cfg.get("app",  {})
    inf_cfg  = cfg.get("inference", {})
    tts_cfg  = cfg.get("tts", {})
    ui_cfg   = cfg.get("ui",  {})

    with gr.Tab("⚙️ Settings"):
        gr.Markdown("## ⚙️ Settings\nConfigure Voice Model Studio preferences.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🖥️ System Information")
                sys_info_md = gr.Markdown(_sys_info())
                refresh_si  = gr.Button("🔄 Refresh", size="sm")
                refresh_si.click(fn=_sys_info, outputs=sys_info_md)

                gr.Markdown("### 📦 Pre-trained Models")
                gr.Markdown(
                    "Download HuBERT, RMVPE and FCPE utility models "
                    "from HuggingFace (required for full inference)."
                )
                dl_btn  = gr.Button("⬇️ Download Pre-trained Models", variant="secondary")
                dl_out  = gr.Textbox(label="Download Status", interactive=False, lines=4)
                dl_btn.click(fn=_download_pretrained, outputs=dl_out)

            with gr.Column():
                gr.Markdown("### 🎛️ Default Inference Settings")
                def_f0    = gr.Dropdown(
                    ["rmvpe", "crepe", "harvest", "dio", "pm"],
                    value=inf_cfg.get("default_f0_method", "rmvpe"),
                    label="Default F0 Method",
                )
                def_tr    = gr.Slider(-24, 24, value=inf_cfg.get("default_transpose", 0),
                                      step=1, label="Default Transpose")
                def_ir    = gr.Slider(0, 1, value=inf_cfg.get("default_index_rate", 0.75),
                                      step=0.01, label="Default Index Rate")
                def_prot  = gr.Slider(0, 0.5, value=inf_cfg.get("default_protect", 0.33),
                                      step=0.01, label="Default Protect")

                gr.Markdown("### 🗣️ Default TTS Settings")
                def_engine = gr.Dropdown(
                    ["edge-tts", "gtts", "pyttsx3"],
                    value=tts_cfg.get("default_engine", "edge-tts"),
                    label="Default TTS Engine",
                )
                def_voice  = gr.Textbox(
                    value=tts_cfg.get("default_voice", "en-US-AriaNeural"),
                    label="Default Edge-TTS Voice",
                )

                gr.Markdown("### 🌐 Application")
                port_in  = gr.Number(value=app_cfg.get("port", 7860), label="Port", precision=0)
                share_in = gr.Checkbox(value=app_cfg.get("share", False), label="Enable Gradio Share Link")

                save_btn = gr.Button("💾 Save Settings", variant="primary")
                save_out = gr.Textbox(label="Status", interactive=False, lines=1)

                def _save(f0, tr, ir, prot, engine, voice, port, share):
                    c = _load_cfg()
                    c.setdefault("inference", {}).update({
                        "default_f0_method": f0,
                        "default_transpose": int(tr),
                        "default_index_rate": ir,
                        "default_protect": prot,
                    })
                    c.setdefault("tts", {}).update({
                        "default_engine": engine,
                        "default_voice": voice,
                    })
                    c.setdefault("app", {}).update({
                        "port": int(port),
                        "share": share,
                    })
                    return _save_cfg(c)

                save_btn.click(
                    fn=_save,
                    inputs=[def_f0, def_tr, def_ir, def_prot,
                            def_engine, def_voice, port_in, share_in],
                    outputs=save_out,
                )

"""
Voice Model Studio — Main Application Entry Point
=================================================

A production-grade, one-click, end-to-end voice model studio with:
  ✅ RVC voice conversion (all .pth + .index formats)
  ✅ Text-to-Speech synthesis (edge-tts / gTTS / pyttsx3)
  ✅ Custom voice model training pipeline
  ✅ Audio tools (format conversion, denoise, pitch-shift, normalize)
  ✅ Pre-trained model downloader (HuBERT, RMVPE, FCPE)
  ✅ GPU / CPU auto-detection
  ✅ Works on Windows 10, Linux, macOS — single launcher

Run
---
    python app.py              # default (port 7860)
    python app.py --port 8080
    python app.py --share      # expose via Gradio share link
    python app.py --download-models   # pre-download HuBERT / RMVPE / FCPE
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ── Configure logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

ROOT        = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "configs" / "default.json"


# ── Configuration helpers ─────────────────────────────────────────────────────

def load_config() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Config file not found — using defaults.")
        return {}
    except json.JSONDecodeError as e:
        logger.error("Config parse error: %s — using defaults.", e)
        return {}


def _apply_env_overrides(cfg: dict) -> dict:
    """Allow .env / environment variables to override config values."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass

    app = cfg.setdefault("app", {})
    if os.environ.get("HOST"):
        app["host"] = os.environ["HOST"]
    if os.environ.get("PORT"):
        try:
            app["port"] = int(os.environ["PORT"])
        except ValueError:
            pass
    if os.environ.get("SHARE", "").lower() in ("1", "true", "yes"):
        app["share"] = True
    if os.environ.get("GPU_ID"):
        cfg.setdefault("inference", {})["gpu_id"] = os.environ["GPU_ID"]
    return cfg


# ── Pre-trained model download ────────────────────────────────────────────────

def download_models() -> None:
    logger.info("Downloading pre-trained models (HuBERT, RMVPE, FCPE) …")
    try:
        from modules.rvc.model_manager import ensure_all_pretrained
        results = ensure_all_pretrained()
        for name, path in results.items():
            logger.info("  ✅ %s → %s", name, path)
    except Exception as e:
        logger.error("Download failed: %s", e)


# ── Gradio UI construction ────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global ──────────────────────────────────────────── */
:root {
    --accent: #6C63FF;
    --accent-light: #9b94ff;
    --bg-dark: #0f0f17;
    --bg-card: #1a1a2e;
    --text-muted: #888;
}

body, .gradio-container {
    background: var(--bg-dark) !important;
}

/* ── Header ─────────────────────────────────────────── */
.studio-header {
    background: linear-gradient(135deg, #6C63FF 0%, #3ec6e0 100%);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 16px;
}

.studio-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: white;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}

.studio-header p {
    color: rgba(255,255,255,0.85);
    margin: 0;
    font-size: 1rem;
}

/* ── Tabs ───────────────────────────────────────────── */
.tab-nav button {
    font-weight: 600;
    font-size: 0.95rem;
}

/* ── Buttons ────────────────────────────────────────── */
button.primary {
    background: linear-gradient(90deg, #6C63FF, #3ec6e0) !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
}

button.primary:hover {
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(108,99,255,0.4);
}

/* ── Cards ──────────────────────────────────────────── */
.gr-box, .gr-panel {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* ── Footer ─────────────────────────────────────────── */
.studio-footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.8rem;
    margin-top: 24px;
    padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
"""

HEADER_HTML = """
<div class="studio-header">
  <h1>🎵 Voice Model Studio</h1>
  <p>
    Production-grade AI voice conversion &amp; synthesis — RVC, TTS, Training, Audio Tools
  </p>
</div>
"""

FOOTER_HTML = """
<div class="studio-footer">
  Voice Model Studio v1.0.0 &nbsp;|&nbsp;
  Powered by PyTorch, RVC, Gradio &nbsp;|&nbsp;
  <a href="https://github.com/MASSIVEMAGNETICS/VOICE-MODEL-AI" style="color:#6C63FF;">GitHub</a>
</div>
"""


def build_ui(cfg: dict):
    """Construct and return the Gradio Blocks UI."""
    import gradio as gr

    from modules.ui import (
        inference_tab,
        training_tab,
        tts_tab,
        audio_tools_tab,
        settings_tab,
    )

    with gr.Blocks(title="Voice Model Studio") as demo:
        gr.HTML(HEADER_HTML)

        with gr.Tabs():
            inference_tab.build_tab(cfg)
            tts_tab.build_tab(cfg)
            training_tab.build_tab(cfg)
            audio_tools_tab.build_tab(cfg)
            settings_tab.build_tab(cfg)

        gr.HTML(FOOTER_HTML)

    return demo


# ── CLI / main ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Voice Model Studio")
    p.add_argument("--host",             default="",    help="Bind host (overrides config)")
    p.add_argument("--port",   type=int, default=0,     help="Port (overrides config)")
    p.add_argument("--share",  action="store_true",     help="Enable Gradio share link")
    p.add_argument("--download-models",  action="store_true", dest="download_models",
                   help="Download pre-trained utility models and exit")
    p.add_argument("--no-browser",       action="store_true", dest="no_browser",
                   help="Do not open a browser window automatically")
    p.add_argument("--debug",            action="store_true", help="Enable debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = load_config()
    cfg = _apply_env_overrides(cfg)

    if args.download_models:
        download_models()
        return

    app_cfg  = cfg.get("app", {})
    host     = args.host  or app_cfg.get("host", "0.0.0.0")
    port     = args.port  or app_cfg.get("port", 7860)
    share    = args.share or app_cfg.get("share", False)

    logger.info("━" * 56)
    logger.info("  Voice Model Studio v1.0.0")
    logger.info("  http://%s:%d", host if host != "0.0.0.0" else "127.0.0.1", port)
    if share:
        logger.info("  Gradio public share link enabled")
    logger.info("━" * 56)

    try:
        import gradio as gr
    except ImportError:
        logger.error(
            "Gradio is not installed. Run 'python install.py' or "
            "'pip install gradio' to install it."
        )
        sys.exit(1)

    demo = build_ui(cfg)

    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=not args.no_browser,
        show_error=True,
        quiet=False,
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()

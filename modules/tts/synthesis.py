"""
Text-to-Speech synthesis module.

Supports multiple TTS engines with a unified interface:
  - edge-tts   (Microsoft Azure Neural Voices — free, high quality, online)
  - gTTS        (Google Translate TTS — free, online)
  - pyttsx3     (OS TTS — offline, no internet required)

Usage
-----
    from modules.tts.synthesis import TTSSynthesizer

    synth = TTSSynthesizer(engine="edge-tts")
    out   = await synth.synthesize(
        text="Hello, this is Voice Model Studio!",
        voice="en-US-AriaNeural",
        output_path="outputs/hello.wav",
    )
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[2]
OUTPUTS    = ROOT / "outputs"

TTS_ENGINE = str  # "edge-tts" | "gtts" | "pyttsx3"


# ── Engine availability ───────────────────────────────────────────────────────

def _has(pkg: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(pkg) is not None


# ── Edge-TTS (Microsoft Neural Voices) ───────────────────────────────────────

class EdgeTTSEngine:
    """Microsoft Edge TTS — requires internet, free, very high quality."""

    @staticmethod
    async def list_voices(language: Optional[str] = None) -> List[Dict]:
        """Return list of available voices, optionally filtered by language prefix."""
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            if language:
                voices = [v for v in voices if v["Locale"].startswith(language)]
            return voices
        except Exception as e:
            logger.warning("edge-tts voice list failed: %s", e)
            return []

    @staticmethod
    async def synthesize(
        text: str,
        voice: str = "en-US-AriaNeural",
        output_path: str | Path = "",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ) -> Path:
        """
        Convert text to speech using Edge TTS.

        Parameters
        ----------
        text        : Input text (plain text or SSML)
        voice       : Edge voice name, e.g. "en-US-AriaNeural"
        output_path : Destination (WAV or MP3)
        rate        : Speaking rate, e.g. "+10%" or "-10%"
        pitch       : Pitch adjustment, e.g. "+10Hz"
        volume      : Volume, e.g. "+10%"
        """
        import edge_tts

        output_path = Path(output_path) if output_path else (
            OUTPUTS / f"tts_{int(time.time())}.wav"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Edge-TTS only writes MP3 natively; convert to WAV via soundfile if needed
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_mp3 = tmp.name

        try:
            communicate = edge_tts.Communicate(
                text, voice, rate=rate, pitch=pitch, volume=volume
            )
            await communicate.save(tmp_mp3)

            suffix = output_path.suffix.lower()
            if suffix in (".wav", ".flac", ".ogg"):
                from modules.audio.processing import convert_audio_format
                convert_audio_format(tmp_mp3, output_path)
            else:
                import shutil
                shutil.copy2(tmp_mp3, str(output_path))
        finally:
            try:
                os.unlink(tmp_mp3)
            except OSError:
                pass

        return output_path


# ── Google TTS ───────────────────────────────────────────────────────────────

class GTTSEngine:
    """Google Translate TTS — requires internet, free, decent quality."""

    LANG_MAP = {
        "en": "en", "es": "es", "fr": "fr", "de": "de",
        "it": "it", "pt": "pt", "ru": "ru", "ja": "ja",
        "ko": "ko", "zh": "zh-CN", "ar": "ar", "hi": "hi",
    }

    @staticmethod
    def synthesize(
        text: str,
        lang: str = "en",
        output_path: str | Path = "",
        slow: bool = False,
    ) -> Path:
        from gtts import gTTS

        output_path = Path(output_path) if output_path else (
            OUTPUTS / f"tts_{int(time.time())}.mp3"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tts = gTTS(text=text, lang=lang, slow=slow)

        if output_path.suffix.lower() == ".mp3":
            tts.save(str(output_path))
        else:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_mp3 = tmp.name
            try:
                tts.save(tmp_mp3)
                from modules.audio.processing import convert_audio_format
                convert_audio_format(tmp_mp3, output_path)
            finally:
                try:
                    os.unlink(tmp_mp3)
                except OSError:
                    pass

        return output_path


# ── pyttsx3 (offline) ─────────────────────────────────────────────────────────

class Pyttsx3Engine:
    """Offline TTS using the OS speech engine (SAPI5 on Windows, espeak on Linux)."""

    def __init__(self):
        import pyttsx3
        self._engine = pyttsx3.init()

    def list_voices(self) -> List[Dict]:
        voices = self._engine.getProperty("voices")
        return [{"id": v.id, "name": v.name, "lang": getattr(v, "languages", [])} for v in voices]

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        rate: int = 200,
        volume: float = 1.0,
        output_path: str | Path = "",
    ) -> Path:
        output_path = Path(output_path) if output_path else (
            OUTPUTS / f"tts_{int(time.time())}.wav"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if voice_id:
            self._engine.setProperty("voice", voice_id)
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", volume)
        self._engine.save_to_file(text, str(output_path))
        self._engine.runAndWait()
        return output_path


# ── Unified Synthesizer ───────────────────────────────────────────────────────

class TTSSynthesizer:
    """
    Unified TTS interface.

    Parameters
    ----------
    engine : "edge-tts" | "gtts" | "pyttsx3"
    """

    ENGINES = ("edge-tts", "gtts", "pyttsx3")

    def __init__(self, engine: TTS_ENGINE = "edge-tts"):
        self.engine_name = engine
        self._pyttsx3: Optional[Pyttsx3Engine] = None

    # ─── Sync wrapper for async engines ──────────────────────────────────────

    def _run_async(self, coro):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result()
        except RuntimeError:
            pass
        return asyncio.run(coro)

    # ─── Voice listing ────────────────────────────────────────────────────────

    def list_voices(self, language: Optional[str] = None) -> List[Dict]:
        """Return available voices for the current engine."""
        if self.engine_name == "edge-tts":
            return self._run_async(EdgeTTSEngine.list_voices(language))
        if self.engine_name == "pyttsx3":
            if self._pyttsx3 is None:
                self._pyttsx3 = Pyttsx3Engine()
            return self._pyttsx3.list_voices()
        return []

    def get_voice_names(self, language: Optional[str] = None) -> List[str]:
        """Return list of voice display names."""
        voices = self.list_voices(language)
        if self.engine_name == "edge-tts":
            return [v.get("ShortName", v.get("Name", "")) for v in voices]
        if self.engine_name == "pyttsx3":
            return [v["name"] for v in voices]
        return []

    # ─── Synthesis ────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        voice: str = "en-US-AriaNeural",
        language: str = "en",
        output_path: str | Path = "",
        rate: float = 1.0,
        pitch: int = 0,
        volume: float = 1.0,
        slow: bool = False,
    ) -> Path:
        """
        Synthesize speech from *text*.

        Parameters
        ----------
        text        : Input text
        voice       : Voice name (engine-specific)
        language    : Language code (for gTTS / edge-tts filtering)
        output_path : Output file path
        rate        : Speed multiplier (1.0 = normal)
        pitch       : Pitch adjustment in semitones (edge-tts: Hz, pyttsx3: ignored)
        volume      : Volume 0–1
        slow        : Slow mode (gTTS only)
        """
        if self.engine_name == "edge-tts":
            if not _has("edge_tts"):
                raise ImportError("edge-tts not installed: pip install edge-tts")
            rate_str   = self._rate_to_edge(rate)
            pitch_str  = f"{pitch:+d}Hz"
            volume_str = f"{int((volume - 1) * 100):+d}%"
            return self._run_async(
                EdgeTTSEngine.synthesize(
                    text, voice, output_path,
                    rate=rate_str, pitch=pitch_str, volume=volume_str,
                )
            )

        if self.engine_name == "gtts":
            if not _has("gtts"):
                raise ImportError("gTTS not installed: pip install gTTS")
            return GTTSEngine.synthesize(text, lang=language, output_path=output_path, slow=slow)

        if self.engine_name == "pyttsx3":
            if not _has("pyttsx3"):
                raise ImportError("pyttsx3 not installed: pip install pyttsx3")
            if self._pyttsx3 is None:
                self._pyttsx3 = Pyttsx3Engine()
            rate_wpm = int(rate * 175)  # 175 wpm baseline
            return self._pyttsx3.synthesize(
                text, voice_id=voice or None,
                rate=rate_wpm, volume=volume,
                output_path=output_path,
            )

        raise ValueError(f"Unknown engine: {self.engine_name!r}. Choose from {self.ENGINES}")

    @staticmethod
    def _rate_to_edge(rate: float) -> str:
        """Convert a speed multiplier to an Edge-TTS percentage string."""
        pct = int((rate - 1.0) * 100)
        return f"{pct:+d}%"

    # ─── TTS-to-RVC chaining ─────────────────────────────────────────────────

    def synthesize_then_convert(
        self,
        text: str,
        rvc_pipeline,           # RVCInferencePipeline
        voice: str = "en-US-AriaNeural",
        language: str = "en",
        output_path: str | Path = "",
        tts_output_path: str | Path = "",
        **rvc_kwargs,
    ) -> Tuple[Path, Path]:
        """
        Full pipeline: text → TTS audio → RVC voice conversion.

        Returns
        -------
        (tts_path, rvc_path)
        """
        tts_path = self.synthesize(
            text, voice=voice, language=language,
            output_path=tts_output_path,
        )
        rvc_path = rvc_pipeline.convert(
            str(tts_path),
            output_path=output_path or "",
            **rvc_kwargs,
        )
        return tts_path, rvc_path


# ── Convenience functions ─────────────────────────────────────────────────────

_default_synth: Optional[TTSSynthesizer] = None


def get_default_synthesizer(engine: str = "edge-tts") -> TTSSynthesizer:
    global _default_synth
    if _default_synth is None or _default_synth.engine_name != engine:
        _default_synth = TTSSynthesizer(engine)
    return _default_synth


def quick_tts(text: str, output_path: str | Path = "", engine: str = "edge-tts") -> Path:
    """One-liner TTS helper."""
    return get_default_synthesizer(engine).synthesize(text, output_path=output_path)

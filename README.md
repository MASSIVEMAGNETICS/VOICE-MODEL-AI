# 🎵 Voice Model Studio

**Production-grade, one-click, end-to-end AI voice studio.**  
Works on Windows 10, Linux and macOS — CPU *and* GPU.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

---

## ✨ Features

| Category | Details |
|---|---|
| **Voice Conversion** | RVC v1 / v2 — load any `.pth` + `.index` file and convert audio in seconds |
| **F0 Methods** | RMVPE ⭐, CREPE, Harvest, DIO, Parselmouth — choose the best for your use case |
| **Text-to-Speech** | Microsoft Edge-TTS (400+ voices), Google TTS, pyttsx3 (offline) |
| **TTS → RVC Chain** | Synthesise speech and pipe it directly through your voice model in one click |
| **Training Pipeline** | Dataset prep → F0 extraction → feature extraction → training → FAISS index |
| **Audio Tools** | Format converter, noise reduction, pitch shift, time stretch, normaliser |
| **File Compatibility** | WAV, MP3, FLAC, OGG, AAC, M4A, WMA, AIFF, OPUS, WEBM — all via FFmpeg |
| **Pre-trained Models** | Auto-download HuBERT, RMVPE, FCPE from HuggingFace |
| **GPU Acceleration** | CUDA (NVIDIA), Apple MPS, CPU — auto-detected |
| **Settings UI** | Live system info, default configuration editor, model downloader |

---

## 🚀 Quick Start

### Windows 10 / 11 — One Click

```batch
REM 1. Double-click setup.bat  (installs everything automatically)
REM 2. Double-click start.bat  (opens the studio in your browser)
```

> **Requires:** Python 3.10+ from [python.org](https://www.python.org/downloads/) and [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) in PATH.

### Linux / macOS — One Click

```bash
chmod +x setup.sh start.sh
./setup.sh          # install
./start.sh          # launch
```

### Manual (any platform)

```bash
python install.py           # creates .venv, installs deps
python app.py               # starts the web UI at http://localhost:7860
```

---

## 📁 Project Structure

```
VOICE-MODEL-AI/
├── app.py                      # Main entry point (Gradio WebUI)
├── install.py                  # Cross-platform installer
├── requirements.txt            # Python dependencies
├── setup.bat                   # Windows one-click installer
├── setup.sh                    # Linux/macOS one-click installer
├── start.bat / start.sh        # Launch scripts
│
├── configs/
│   └── default.json            # App configuration (editable in UI)
│
├── models/                     # Drop your .pth and .index files here
├── outputs/                    # Converted / synthesised audio
├── uploads/                    # Temporary upload area
├── logs/                       # Application logs
├── assets/
│   └── pretrained/             # HuBERT, RMVPE, FCPE checkpoints
│
└── modules/
    ├── audio/
    │   └── processing.py       # Load, save, normalize, denoise, pitch-shift
    ├── rvc/
    │   ├── inference.py        # RVC voice conversion pipeline
    │   ├── training.py         # Dataset prep + training orchestration
    │   ├── model_manager.py    # .pth / .index discovery & caching
    │   ├── feature_extract.py  # HuBERT / ContentVec feature extractor
    │   ├── synthesizer.py      # VITS synthesizer factory
    │   └── rmvpe.py            # RMVPE pitch tracker
    ├── tts/
    │   └── synthesis.py        # Unified TTS (edge-tts / gTTS / pyttsx3)
    └── ui/
        ├── inference_tab.py    # Voice Conversion tab
        ├── tts_tab.py          # Text-to-Speech tab
        ├── training_tab.py     # Train Model tab
        ├── audio_tools_tab.py  # Audio Tools tab
        └── settings_tab.py     # Settings tab
```

---

## 🎙️ Voice Conversion (RVC)

1. Drop your `.pth` (and optionally `.index`) file into the `models/` folder.
2. Open the **Voice Conversion** tab.
3. Select your model from the dropdown (or upload directly).
4. Upload or record audio, adjust parameters, click **Convert**.

### Parameters

| Parameter | Description |
|---|---|
| **F0 Method** | Pitch extraction algorithm. `rmvpe` is most accurate. |
| **Transpose** | Pitch shift in semitones (e.g. +12 = one octave up). |
| **Auto Pitch** | Auto-detect speaker gender and suggest transpose. |
| **Index Rate** | How much FAISS retrieval blends into the output (0–1). |
| **Filter Radius** | Median filter size for F0 smoothing (reduces pitch jitter). |
| **RMS Mix Rate** | Blend ratio of input/output RMS loudness. |
| **Protect** | Protects unvoiced consonants from pitch conversion (0–0.5). |

---

## 🗣️ Text-to-Speech

Switch between three engines:

| Engine | Quality | Internet | Voices |
|---|---|---|---|
| **edge-tts** | ⭐⭐⭐⭐⭐ | Required | 400+ neural voices |
| **gTTS** | ⭐⭐⭐ | Required | 30+ languages |
| **pyttsx3** | ⭐⭐ | None | OS voices (SAPI5 / espeak) |

Enable **Apply RVC After TTS** to pipe synthesised speech through your voice model.

---

## 🏋️ Training a Voice Model

1. Collect **10–60 minutes** of clean, single-speaker audio (WAV/MP3/FLAC).
2. Open the **Train Model** tab.
3. Enter a model name and path to your audio folder.
4. Click **Prepare + Train**.

Training stages:
1. **Slicing** — audio is split into ≤15-second segments.
2. **F0 extraction** — pitch contours saved as `.npy`.
3. **Feature extraction** — HuBERT content vectors saved as `.npy`.
4. **Model training** — VITS-based generator trained.
5. **FAISS index** — similarity index built from extracted features.

---

## 🛠️ Audio Tools

Standalone utilities accessible from the **Audio Tools** tab:

- **Format Converter** — convert between WAV, MP3, FLAC, OGG, AAC, OPUS
- **Noise Reduction** — powered by `noisereduce`
- **Pitch Shift** — shift pitch without affecting tempo
- **Time Stretch** — change tempo without affecting pitch
- **Normalize** — peak normalisation
- **Audio Info** — duration, sample rate, channels, peak/RMS levels

---

## ⬇️ Downloading Pre-trained Models

```bash
python app.py --download-models
```

Or use the **Settings → Download Pre-trained Models** button in the UI.

Downloads to `assets/pretrained/`:
- `hubert_base.pt` — content feature extractor
- `rmvpe.pt` — neural pitch tracker
- `fcpe.pt` — fast pitch estimator

---

## ⚙️ Configuration

Edit `configs/default.json` or use the **Settings** tab.

Key options:

```json
{
  "app":       { "host": "0.0.0.0", "port": 7860, "share": false },
  "inference": { "default_f0_method": "rmvpe", "default_transpose": 0 },
  "tts":       { "default_engine": "edge-tts", "default_voice": "en-US-AriaNeural" },
  "training":  { "default_sample_rate": 40000, "default_epochs": 100 }
}
```

Environment variables (`.env` file or shell):

```env
HOST=0.0.0.0
PORT=7860
SHARE=false
GPU_ID=0
```

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10, Ubuntu 20.04, macOS 11 | Windows 11, Ubuntu 22.04, macOS 13 |
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU mode) | NVIDIA RTX 3060+ (8 GB VRAM) |
| Storage | 5 GB | 20 GB |
| FFmpeg | Required | Required |

---

## 🪟 Windows 10 — Troubleshooting

| Issue | Fix |
|---|---|
| `python` not found | Install from [python.org](https://www.python.org/downloads/), tick **Add to PATH** |
| `ffmpeg` not found | Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), add `bin/` to PATH |
| CUDA not detected | Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) matching your driver |
| `faiss` error on Windows | Use `faiss-cpu` — `faiss-gpu` requires Linux/WSL |
| Port in use | Edit `configs/default.json` — set a different `port` |

---

## 📄 License

MIT License — free to use, modify and distribute.

---

## 🙏 Acknowledgements

- [RVC (Retrieval-based Voice Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)
- [RMVPE](https://github.com/Dream-High/RMVPE)
- [edge-tts](https://github.com/rany2/edge-tts)
- [Gradio](https://www.gradio.app/)
- [WORLD Vocoder / pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)

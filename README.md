# TUNDA — Affect-aware voice companion

TUNDA is a **local voice/chat companion** that labels affect (from text, and from audio on the desktop path), routes crisis language to help resources, and replies with supportive language via a local LLM when available.

It is **not** a therapist, not emergency care, and **not** a validated empathic intelligence system. Empathy here means *affect-labeled, prompt-guided or template-backed supportive replies* — not measured clinical empathy.

## Honest scope

| Fair to say | Do not claim |
|-------------|--------------|
| Affect-aware local companion system | “Empathic AI” as proven understanding |
| Emotion *labels* condition style, prompt, and TTS | Accurate clinical affect sensing |
| Local LLM (Ollama) when online; templates or offline care refusal when not | Always on-device deep empathy |
| Keyword crisis tiers + scripted grounding | Suicide-risk assessment or crisis intervention |
| Consent, encrypted memory, clinic profiles | Replacement for human care |

**Paths differ:** the **desktop** orchestrator can run audio emotion (e.g. wav2vec2 / fallback models). The **web** app currently uses speech-to-text then **text/keyword** emotion for replies. Vector memory RAG feeds the desktop path; the web path uses conversation continuity/recaps more than retrieval into every reply.

## Features

- **Speech recognition** — Whisper / Faster-Whisper (web upload or desktop mic)
- **Affect labels** — audio models on desktop when configured; text/keyword fusion and web text heuristics
- **Response generation** — Ollama local LLM when online; Companion profile may use empathy *templates*; care profiles refuse template improv if the LLM is offline
- **Safety rails** — tiered crisis keyword routing, regional help numbers, interruptible grounding scripts
- **Clinic profiles** — Companion / Between sessions / High-risk watch with locked prompts
- **Voice output** — optional Piper or system TTS; web may fall back to browser `speechSynthesis`
- **Privacy-oriented memory** — consent, wipe, encryption options; local processing by default

## Architecture (simplified)

```
Audio/Text → STT (Whisper) → Affect label → Mode / Safety / Grounding
                                    ↓
                         Local LLM (Ollama) or templates / offline care message
                                    ↓
                              TTS (Piper / system / browser)
```

Default config may use **system** TTS rather than Piper. Check `config.yaml`.

## Supported emotion labels

Happy, sad, angry, anxious, calm, neutral — used as **routing labels**, not diagnostic categories.

## Installation

### Prerequisites

- Python 3.8+ (3.9–3.11 recommended)
- FFmpeg (audio)
- ~4GB+ RAM if you run a local LLM
- [Ollama](https://ollama.com) recommended for care profiles (`between_sessions`, `high_risk_watch`)

### Quick setup

```bash
cd TUNDA
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
python install.py
```

Or manually:

```bash
pip install -r requirements-minimal.txt
python setup_models.py
```

## Usage

```bash
# Desktop / CLI
python main.py

# Web UI — http://localhost:8000
python app.py

# REST API — http://localhost:8001
python api_server.py
```

In the web UI, the honesty strip states that TUNDA is not a therapist and not emergency care. Prefer **Companion** without Ollama; start Ollama before care profiles.

## Configuration

Edit `config.yaml` for latency profile, STT/TTS providers, emotion model type, clinic default profile, and personality styles.

See also:

- [`docs/ROADMAP_FOR_CLINICIANS_AND_USERS.md`](docs/ROADMAP_FOR_CLINICIANS_AND_USERS.md) — user and clinician framing
- [`GETTING_STARTED.md`](GETTING_STARTED.md) — setup walkthrough

## Project layout (high level)

```
TUNDA/
├── src/
│   ├── speech/          # STT, TTS, enhance, stream
│   ├── emotion/         # detectors, fusion
│   ├── response/        # generator, empathy templates, safety, grounding, mode router
│   ├── memory/          # conversation, crypto, continuity
│   ├── clinic/          # care profiles
│   └── eval/            # small golden harness (routing/safety, not human-rated empathy)
├── web/                 # templates + static
├── tests/
├── main.py              # CLI
├── app.py               # web
└── config.yaml
```

## License

MIT License — see `LICENSE`.

## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Make your changes and add tests  
4. Open a pull request  

Avoid marketing copy that implies clinical empathy or emergency capability.

## Acknowledgments

- OpenAI Whisper / Faster-Whisper for speech recognition  
- Librosa and Hugging Face models for audio analysis where configured  
- Piper (optional) and system/browser TTS for speech synthesis  
- Ollama for local LLM responses  

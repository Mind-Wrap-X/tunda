"""Emotion-adaptive spoken-reply profiles and barge-in helpers.

Clinical intent: match the *patient's* affect without mirroring distress.
Anxious and angry replies slow down and soften; they never speed up or get louder.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SpeechProfile:
    """Prosody used for a spoken reply."""

    rate: float = 0.92
    pitch: float = 1.0
    volume: float = 0.84
    pause_ms: int = 220

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# Care-first defaults: slower, quieter, never agitated.
_PROFILES: Dict[str, SpeechProfile] = {
    "neutral": SpeechProfile(rate=0.92, pitch=1.00, volume=0.84, pause_ms=220),
    "calm": SpeechProfile(rate=0.86, pitch=0.97, volume=0.80, pause_ms=280),
    "sad": SpeechProfile(rate=0.82, pitch=0.93, volume=0.78, pause_ms=320),
    "anxious": SpeechProfile(rate=0.84, pitch=0.97, volume=0.76, pause_ms=300),
    "angry": SpeechProfile(rate=0.85, pitch=0.95, volume=0.78, pause_ms=280),
    "happy": SpeechProfile(rate=0.95, pitch=1.04, volume=0.86, pause_ms=180),
}

_ESCALATE = SpeechProfile(rate=0.78, pitch=0.94, volume=0.74, pause_ms=360)
_GROUNDING = SpeechProfile(rate=0.76, pitch=0.96, volume=0.76, pause_ms=1400)

_ALIASES = {
    "fear": "anxious",
    "fearful": "anxious",
    "worried": "anxious",
    "stressed": "anxious",
    "panic": "anxious",
    "melancholy": "sad",
    "down": "sad",
    "frustrated": "angry",
    "irritated": "angry",
    "joy": "happy",
    "excited": "happy",
    "peaceful": "calm",
    "relaxed": "calm",
    "content": "calm",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalize_emotion(emotion: Optional[str]) -> str:
    key = (emotion or "neutral").strip().lower()
    return _ALIASES.get(key, key if key in _PROFILES else "neutral")


def get_speech_profile(
    emotion: Optional[str] = None,
    interaction_mode: Optional[str] = None,
) -> SpeechProfile:
    """Return spoken-reply prosody for an emotion and interaction mode."""
    mode = (interaction_mode or "").strip().lower()
    if mode in {"escalate", "crisis"}:
        return _ESCALATE
    if mode == "grounding":
        return _GROUNDING
    return _PROFILES[normalize_emotion(emotion)]


def speech_settings_for(
    emotion: Optional[str] = None,
    interaction_mode: Optional[str] = None,
) -> Dict[str, float]:
    """JSON-safe profile for the web client."""
    return get_speech_profile(emotion, interaction_mode).to_dict()


def split_speech_units(text: str) -> List[str]:
    """Split a reply into barge-in-friendly spoken units."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(cleaned) if p.strip()]
    return parts or [cleaned]


def apply_speech_profile(
    audio: np.ndarray,
    sample_rate: int,
    profile: SpeechProfile,
) -> np.ndarray:
    """Apply rate, pitch, and volume to a waveform (numpy-only, testable)."""
    if audio is None or len(audio) == 0:
        return audio

    out = np.asarray(audio, dtype=np.float32)

    rate = float(np.clip(profile.rate, 0.5, 1.4))
    if abs(rate - 1.0) > 0.01:
        out = _resample(out, 1.0 / rate)

    pitch = float(np.clip(profile.pitch, 0.7, 1.3))
    if abs(pitch - 1.0) > 0.01:
        out = _shift_pitch(out, sample_rate, pitch)

    volume = float(np.clip(profile.volume, 0.2, 1.0))
    out = np.clip(out * volume, -1.0, 1.0)

    if profile.pause_ms > 0 and sample_rate > 0:
        pad = int(sample_rate * (profile.pause_ms / 1000.0))
        if pad > 0:
            out = np.concatenate([out, np.zeros(pad, dtype=np.float32)])

    return out.astype(np.float32)


def _resample(audio: np.ndarray, factor: float) -> np.ndarray:
    if len(audio) < 2:
        return audio
    new_len = max(2, int(round(len(audio) * factor)))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=True)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _shift_pitch(audio: np.ndarray, sample_rate: int, pitch_ratio: float) -> np.ndarray:
    try:
        import librosa

        n_steps = 12.0 * np.log2(pitch_ratio)
        return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=float(n_steps)).astype(
            np.float32
        )
    except Exception:
        return _resample(audio, pitch_ratio)

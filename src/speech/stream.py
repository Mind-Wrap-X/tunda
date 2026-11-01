"""Low-latency TTS chunking so the first sound can start under ~800ms."""

from __future__ import annotations

import io
import re
import wave
from typing import Iterable, List, Tuple

import numpy as np

_CLAUSE = re.compile(r"[,;:—–] |\. |\? |! ")


def split_for_streaming(text: str, first_max: int = 64, later_max: int = 140) -> List[str]:
    """First clause stays short for time-to-first-audio; later chunks can be longer."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []
    first, rest = _take_first_clause(cleaned, first_max)
    chunks = [first] if first else []
    if rest:
        chunks.extend(_pack_sentences(rest, later_max))
    return [c.strip() for c in chunks if c.strip()]


def _take_first_clause(text: str, limit: int) -> Tuple[str, str]:
    if len(text) <= limit:
        return text, ""
    window = text[: limit + 24]
    match = None
    for found in _CLAUSE.finditer(window):
        if found.end() <= limit + 8:
            match = found
    if match and match.end() >= 18:
        cut = match.end()
        return text[:cut].strip(), text[cut:].strip()
    space = text.rfind(" ", 24, limit)
    if space > 0:
        return text[:space].strip(), text[space:].strip()
    return text[:limit].strip(), text[limit:].strip()


def _pack_sentences(text: str, max_chars: int) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    out: List[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if len(current) + len(part) + 1 <= max_chars:
            current = f"{current} {part}".strip()
        else:
            if current:
                out.append(current)
            current = part
    if current:
        out.append(current)
    return out


def float_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm_i16.tobytes())
    return buf.getvalue()


def wav_bytes_to_float(data: bytes, target_rate: int = 16000) -> np.ndarray:
    """Decode a mono/stereo WAV into float32 PCM at target_rate."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if width == 2:
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        pcm = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        pcm = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if rate != target_rate and len(pcm) > 1:
        duration = len(pcm) / float(rate)
        new_len = max(1, int(round(duration * target_rate)))
        x_old = np.linspace(0.0, 1.0, num=len(pcm), endpoint=True)
        x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=True)
        pcm = np.interp(x_new, x_old, pcm).astype(np.float32)
    return pcm.astype(np.float32)

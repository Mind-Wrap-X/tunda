"""Noise-robust cleanup for distressed, phone, and room audio."""

from __future__ import annotations

import numpy as np

from ..utils.audio import AudioProcessor


DISTRESS_STT_PROMPT = (
    "Empathic conversation. The speaker may be crying, whispering, or on a phone. "
    "Transcribe the words faithfully. Do not invent extra sentences."
)


def enhance_for_stt(
    audio: np.ndarray,
    sample_rate: int = 16000,
    highpass_hz: float = 90.0,
) -> np.ndarray:
    """High-pass, spectral denoise, and lift quiet speech without clipping."""
    if audio is None or len(audio) == 0:
        return audio
    out = np.asarray(audio, dtype=np.float32)
    out = _highpass(out, sample_rate, highpass_hz)
    processor = AudioProcessor(sample_rate=sample_rate)
    try:
        noise_s = min(0.4, max(0.08, len(out) / sample_rate * 0.12))
        out = processor.apply_noise_reduction(out, noise_duration=noise_s)
    except Exception:
        pass
    out = _compress(out)
    return processor.normalize_audio(out).astype(np.float32)


def _highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if len(audio) < 8 or sample_rate <= 0:
        return audio
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = rc / (rc + dt)
    y = np.empty_like(audio)
    y[0] = audio[0]
    for i in range(1, len(audio)):
        y[i] = alpha * (y[i - 1] + audio[i] - audio[i - 1])
    return y


def _compress(audio: np.ndarray, threshold: float = 0.08, ratio: float = 2.4) -> np.ndarray:
    mag = np.abs(audio)
    gain = np.ones_like(mag)
    over = mag > threshold
    gain[over] = (threshold + (mag[over] - threshold) / ratio) / np.maximum(mag[over], 1e-8)
    # Lift very quiet voiced frames so crying/whisper survives Whisper VAD
    quiet = (mag > 0.004) & (mag < threshold)
    gain[quiet] = np.minimum(2.2, threshold / np.maximum(mag[quiet], 1e-8) * 0.45)
    return np.clip(audio * gain, -1.0, 1.0)

import numpy as np

from src.speech.enhance import enhance_for_stt
from src.speech.stream import split_for_streaming, float_to_wav_bytes, wav_bytes_to_float


def test_streaming_puts_a_short_clause_first():
    text = (
        "I hear how heavy this feels right now, and we can slow down together "
        "before we talk about what happens next."
    )
    chunks = split_for_streaming(text, first_max=64, later_max=140)
    assert chunks
    assert len(chunks[0]) <= 80
    assert " ".join(chunks).replace("  ", " ")


def test_wav_roundtrip_preserves_length():
    tone = np.sin(np.linspace(0, 8 * np.pi, 1600)).astype(np.float32)
    wav = float_to_wav_bytes(tone, 16000)
    back = wav_bytes_to_float(wav, target_rate=16000)
    assert abs(len(back) - len(tone)) < 8


def test_enhance_keeps_speech_energy():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    voice = 0.25 * np.sin(2 * np.pi * 180 * t)
    noisy = (voice + 0.04 * rng.standard_normal(len(t))).astype(np.float32)
    cleaned = enhance_for_stt(noisy, sample_rate=16000)
    assert cleaned.shape[0] > 1000
    assert float(np.max(np.abs(cleaned))) > 0.05

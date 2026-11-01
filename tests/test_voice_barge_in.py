"""Tests for emotion-adaptive speech profiles used by barge-in replies."""

from src.speech.voice import (
    SpeechProfile,
    apply_speech_profile,
    get_speech_profile,
    speech_settings_for,
    split_speech_units,
)
import numpy as np


def test_anxious_and_sad_are_slower_and_softer_than_happy():
    happy = get_speech_profile("happy")
    sad = get_speech_profile("sad")
    anxious = get_speech_profile("anxious")

    assert sad.rate < happy.rate
    assert anxious.rate < happy.rate
    assert sad.volume <= happy.volume
    assert anxious.volume < happy.volume


def test_crisis_mode_is_slowest():
    calm = get_speech_profile("calm")
    crisis = get_speech_profile("sad", "escalate")
    assert crisis.rate < calm.rate
    assert crisis.pause_ms >= calm.pause_ms


def test_speech_settings_are_json_safe():
    settings = speech_settings_for("anxious", "listen")
    assert set(settings) == {"rate", "pitch", "volume", "pause_ms"}
    assert 0.5 < settings["rate"] < 1.0


def test_split_speech_units_lets_barge_in_between_sentences():
    units = split_speech_units("I hear you. We can go slowly. You are not alone.")
    assert len(units) == 3
    assert units[0].startswith("I hear you")


def test_slower_profile_makes_audio_longer():
    tone = np.sin(np.linspace(0, 8 * np.pi, 1600)).astype(np.float32)
    slow = SpeechProfile(rate=0.8, pitch=1.0, volume=1.0, pause_ms=0)
    stretched = apply_speech_profile(tone, 16000, slow)
    assert len(stretched) > len(tone)

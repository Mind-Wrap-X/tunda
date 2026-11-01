"""Fuse audio emotion with text sentiment."""

from dataclasses import dataclass
from typing import Dict, Optional

from .types import EmotionPrediction


@dataclass
class TextSentiment:
    label: str
    score: float


class EmotionFusion:
    def __init__(self):
        self.positive = {
            "happy", "joy", "excited", "great", "wonderful", "amazing", "fantastic", "love",
            "grateful", "relieved", "peaceful", "calm"
        }
        self.negative = {
            "sad", "depressed", "down", "upset", "crying", "hurt", "disappointed", "angry",
            "mad", "furious", "annoyed", "frustrated", "anxious", "worried", "nervous",
            "scared", "afraid", "stress", "tired"
        }

    def analyze_text(self, text: str) -> TextSentiment:
        if not text:
            return TextSentiment(label="neutral", score=0.0)
        words = [w.strip(".,!?;:") for w in text.lower().split()]
        pos = sum(1 for w in words if w in self.positive)
        neg = sum(1 for w in words if w in self.negative)
        score = 0.0
        if pos > neg:
            score = min(1.0, pos / max(1, len(words)))
            return TextSentiment(label="positive", score=score)
        if neg > pos:
            score = min(1.0, neg / max(1, len(words)))
            return TextSentiment(label="negative", score=score)
        return TextSentiment(label="neutral", score=0.0)

    def fuse(
        self,
        audio_pred: EmotionPrediction,
        text: str,
        asr_confidence: float,
        text_boost: float = 0.0,
    ) -> EmotionPrediction:
        """Fuse audio-based emotion with text; text_boost (0–~0.5) trusts words more (user calibration)."""
        sentiment = self.analyze_text(text)
        text_emotion = self._map_sentiment_to_emotion(sentiment, text)
        audio_weight = 1.0 - min(0.8, max(0.0, asr_confidence))
        text_weight = 1.0 - audio_weight
        if sentiment.label == "neutral":
            text_weight *= 0.4
            audio_weight = 1.0 - text_weight

        # Calibration from user feedback: shift weight toward lexical/text path
        boost = max(0.0, min(0.45, float(text_boost)))
        audio_weight *= 1.0 - boost
        text_weight = max(0.0, min(1.0, 1.0 - audio_weight))

        fused_emotion = audio_pred.emotion
        fused_confidence = audio_pred.confidence * audio_weight
        if text_emotion:
            fused_emotion = text_emotion
            fused_confidence += sentiment.score * text_weight

        fused_confidence = min(1.0, max(0.0, fused_confidence))
        probabilities = dict(audio_pred.probabilities)
        if text_emotion:
            probabilities[text_emotion] = max(probabilities.get(text_emotion, 0.0), fused_confidence)

        return EmotionPrediction(
            emotion=fused_emotion,
            confidence=fused_confidence,
            probabilities=probabilities,
            features_used=audio_pred.features_used
        )

    def _map_sentiment_to_emotion(self, sentiment: TextSentiment, text: str) -> Optional[str]:
        if sentiment.label == "positive":
            return "happy"
        if sentiment.label == "negative":
            if any(term in text.lower() for term in ["anxious", "worried", "nervous", "scared", "afraid"]):
                return "anxious"
            if any(term in text.lower() for term in ["angry", "mad", "furious", "annoyed", "frustrated"]):
                return "angry"
            return "sad"
        return None

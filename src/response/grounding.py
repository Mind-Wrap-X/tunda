"""Spoken, in-the-moment grounding — paced for voice, not a sidebar card."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class GroundingStep:
    text: str
    pause_ms: int = 900


@dataclass
class GroundingScript:
    script_id: str
    title: str
    steps: List[GroundingStep] = field(default_factory=list)

    def spoken_text(self) -> str:
        return " ".join(step.text.strip() for step in self.steps if step.text.strip())

    def to_payload(self) -> Dict:
        return {
            "id": self.script_id,
            "title": self.title,
            "steps": [{"text": s.text, "pause_ms": s.pause_ms} for s in self.steps],
        }


_EXPLICIT = re.compile(
    r"\b("
    r"ground(ing| me)?|help me (calm|ground|settle)|can't (calm|breathe)|"
    r"panic( attack)?|hyperventilat|box breath|5[- ]?4[- ]?3[- ]?2[- ]?1|"
    r"breathing exercise|guide me( through)?|i('m| am) (freaking|spiraling)|"
    r"racing (heart|thoughts)|overwhelmed"
    r")\b",
    re.IGNORECASE,
)

_PANIC = re.compile(
    r"\b(panic attack|can't breathe|cannot breathe|hyperventilat|"
    r"i('m| am) panicking|about to pass out)\b",
    re.IGNORECASE,
)

_SCRIPTS: Dict[str, GroundingScript] = {
    "orient": GroundingScript(
        script_id="orient",
        title="Here and now",
        steps=[
            GroundingStep("I'm right here with you. You don't have to do this perfectly.", 800),
            GroundingStep("Feel your feet on the floor.", 1600),
            GroundingStep("Let your shoulders drop, just a little.", 1400),
            GroundingStep("Look around and name one color you can see.", 2000),
            GroundingStep("When you're ready, tell me one word for how that feels.", 600),
        ],
    ),
    "box": GroundingScript(
        script_id="box",
        title="Box breath",
        steps=[
            GroundingStep("Let's take one slow box breath together. You can stop anytime.", 700),
            GroundingStep("Breathe in, gently, for four.", 4000),
            GroundingStep("Hold, softly, for four.", 4000),
            GroundingStep("Breathe out, slowly, for four.", 4000),
            GroundingStep("Hold empty, for four.", 4000),
            GroundingStep("That's one round. We can do another if you'd like, or just sit here.", 500),
        ],
    ),
    "54321": GroundingScript(
        script_id="54321",
        title="Five senses",
        steps=[
            GroundingStep("We'll use your senses, slowly. Skip anything that doesn't fit.", 800),
            GroundingStep("Name five things you can see.", 2800),
            GroundingStep("Four things you can feel, like fabric, air, or the chair.", 2800),
            GroundingStep("Three things you can hear.", 2400),
            GroundingStep("Two things you can smell, even faintly.", 2200),
            GroundingStep("One thing you can taste, or just the feeling in your mouth.", 1800),
            GroundingStep("You are here, in this moment. I'm still with you.", 500),
        ],
    ),
    "name": GroundingScript(
        script_id="name",
        title="Name the feeling",
        steps=[
            GroundingStep("You don't have to fix this second.", 900),
            GroundingStep("Silently name the feeling. Anxious. Tight. Heavy. Whatever fits.", 2200),
            GroundingStep("Now add, this is a feeling, and feelings move.", 1600),
            GroundingStep("One slow breath out through the mouth.", 2500),
            GroundingStep("I'm here. What feels one percent more true than a moment ago?", 500),
        ],
    ),
}


def should_offer_grounding(
    user_text: str,
    emotion: Optional[str] = None,
    interaction_mode: Optional[str] = None,
) -> bool:
    text = user_text or ""
    mode = (interaction_mode or "").lower()
    if mode in {"escalate", "crisis"}:
        return False
    if _EXPLICIT.search(text) or _PANIC.search(text):
        return True
    if mode == "coach" and (emotion or "").lower() in {"anxious", "angry", "sad"}:
        return True
    return False


def select_grounding(
    user_text: str,
    emotion: Optional[str] = None,
    interaction_mode: Optional[str] = None,
) -> Optional[GroundingScript]:
    if not should_offer_grounding(user_text, emotion, interaction_mode):
        return None
    text = (user_text or "").lower()
    if re.search(r"\b(box breath|breathe with me|breathing exercise)\b", text):
        return _SCRIPTS["box"]
    if re.search(r"\b(5[- ]?4[- ]?3[- ]?2[- ]?1|senses|look around)\b", text):
        return _SCRIPTS["54321"]
    if re.search(r"\b(name (the |this )?feeling|what am i feeling)\b", text):
        return _SCRIPTS["name"]
    if _PANIC.search(user_text or ""):
        return _SCRIPTS["box"]
    keys = ["orient", "54321", "name", "box"]
    digest = hashlib.sha256((user_text or "ground").encode("utf-8")).digest()
    return _SCRIPTS[keys[digest[0] % len(keys)]]

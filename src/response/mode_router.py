"""Route user turns to interaction modes: listen, coach, or escalate (escalate handled by safety first)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ModeDecision:
    """Which conversational stance to take."""

    mode: str  # listen | coach
    reason: str


class InteractionModeRouter:
    """
    Chooses between reflective listening and gentle coaching.

    Crisis / escalation is handled separately by SafetyGuard before generation.
    """

    COACH_HINTS = re.compile(
        r"\b("
        r"how do i|what should i do|help me (calm|relax|sleep|focus|stop)|"
        r"give me (a |an |some )?(tip|idea|exercise|technique|steps|tool)|"
        r"coping skill|grounding|breathe|breathing exercise|box breath|"
        r"panic attack|can't calm down|need (advice|strategies)|"
        r"something (that )?helps|quick (fix|help)"
        r")\b",
        re.IGNORECASE,
    )

    LISTEN_HINTS = re.compile(
        r"\b("
        r"just listen|don't (fix|advise)|no advice|hear me out|"
        r"i (just )?need to vent|only listen|don't tell me what to do"
        r")\b",
        re.IGNORECASE,
    )

    def route(self, user_text: str, user_preferences: Dict[str, Any] | None = None) -> ModeDecision:
        prefs = user_preferences or {}
        explicit = (prefs.get("response_mode") or "").strip().lower()
        if explicit in {"listen", "coach"}:
            return ModeDecision(mode=explicit, reason="user_preference")

        text = (user_text or "").strip()
        if not text:
            return ModeDecision(mode="listen", reason="empty")

        if self.LISTEN_HINTS.search(text):
            return ModeDecision(mode="listen", reason="explicit_listen")

        if self.COACH_HINTS.search(text):
            return ModeDecision(mode="coach", reason="explicit_skill_request")

        return ModeDecision(mode="listen", reason="default")


def mode_instructions(mode: str) -> str:
    """Prompt fragment for LLM."""
    if mode == "coach":
        return (
            "Interaction mode: COACH — The user may want practical support. "
            "Offer at most ONE small, concrete step (e.g. grounding, one slow breath cue, or a tiny next action). "
            "Keep it optional ('if you'd like'). Stay warm; do not lecture or diagnose."
        )
    return (
        "Interaction mode: LISTEN — Prioritize reflective listening. "
        "Reflect feelings briefly, validate, and ask at most ONE gentle open question. "
        "Do not give advice or exercises unless the user clearly asks."
    )

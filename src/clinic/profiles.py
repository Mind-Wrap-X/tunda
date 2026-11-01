"""Clinic-trust profiles: locked prompts, no improvised care when the LLM is down."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

LOCKED_BOUNDARY = (
    "You are Tunda, an adjunctive voice companion. You are not a therapist, not a clinician, "
    "and not emergency services. Do not diagnose, prescribe, or claim to replace human care. "
    "Do not give instructions for self-harm. If the user is in danger, they should use emergency "
    "services; the safety layer may already have handled that."
)

OFFLINE_CARE_MESSAGE = (
    "I'm not in care mode right now because the local language model (Ollama) is offline. "
    "I will not improvise a therapy-style reply from templates. Crisis routing and grounding "
    "scripts still work. Start Ollama, or switch to Companion profile for informal chat."
)

PROFILES: Dict[str, "ClinicProfile"] = {}


@dataclass(frozen=True)
class ClinicProfile:
    profile_id: str
    label: str
    care_mode: bool
    allow_template_fallback: bool
    default_interaction: str
    temperature: float
    max_tokens: int
    locked_prompt: str


PROFILES["companion"] = ClinicProfile(
    profile_id="companion",
    label="Companion",
    care_mode=False,
    allow_template_fallback=True,
    default_interaction="listen",
    temperature=0.7,
    max_tokens=160,
    locked_prompt=(
        f"{LOCKED_BOUNDARY} "
        "Everyday companion mode: warm, brief, and human. Listen first. "
        "Offer at most one optional suggestion if asked."
    ),
)

PROFILES["between_sessions"] = ClinicProfile(
    profile_id="between_sessions",
    label="Between sessions",
    care_mode=True,
    allow_template_fallback=False,
    default_interaction="listen",
    temperature=0.5,
    max_tokens=140,
    locked_prompt=(
        f"{LOCKED_BOUNDARY} "
        "Between-sessions mode: you support someone between visits with their clinician. "
        "Do not run a therapy session. Reflect, validate, and keep the door open to their "
        "human provider. Skills only if they ask, and keep them tiny and optional."
    ),
)

PROFILES["high_risk_watch"] = ClinicProfile(
    profile_id="high_risk_watch",
    label="High-risk watch",
    care_mode=True,
    allow_template_fallback=False,
    default_interaction="listen",
    temperature=0.35,
    max_tokens=120,
    locked_prompt=(
        f"{LOCKED_BOUNDARY} "
        "High-risk watch: prioritize safety and presence over advice. Stay calm and slow. "
        "Do not debate whether they should seek help. Do not set heavy homework. "
        "If they ask for coping, one grounding cue is enough. Never delay emergency care."
    ),
)


def normalize_profile_id(value: Optional[str]) -> str:
    key = (value or "companion").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "companion": "companion",
        "between_sessions": "between_sessions",
        "between_session": "between_sessions",
        "high_risk_watch": "high_risk_watch",
        "high_risk": "high_risk_watch",
        "watch": "high_risk_watch",
    }
    return aliases.get(key, "companion") if key not in PROFILES else key


def get_clinic_profile(profile_id: Optional[str] = None) -> ClinicProfile:
    return PROFILES[normalize_profile_id(profile_id)]


def list_clinic_profiles() -> list:
    return [
        {
            "id": p.profile_id,
            "label": p.label,
            "care_mode": p.care_mode,
            "allow_template_fallback": p.allow_template_fallback,
        }
        for p in PROFILES.values()
    ]

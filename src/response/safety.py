"""Tiered crisis routing with region-specific help numbers.

TUNDA is a companion, not emergency services. This module:
- classifies risk as distress, self-harm ideation, or imminent danger
- returns local help numbers
- escalates if crisis language repeats in the same session
- writes a minimal audit log (no full transcript by default)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.config import config

logger = logging.getLogger(__name__)

# Public, well-known services. Clinics can override via config.
REGION_RESOURCES: Dict[str, Dict[str, str]] = {
    "US": {
        "emergency": "911",
        "crisis_line": "988",
        "crisis_label": "988 Suicide & Crisis Lifeline",
        "crisis_url": "https://988lifeline.org",
    },
    "CA": {
        "emergency": "911",
        "crisis_line": "9-8-8",
        "crisis_label": "Talk Suicide Canada (9-8-8)",
        "crisis_url": "https://988.ca",
    },
    "UK": {
        "emergency": "999",
        "crisis_line": "116 123",
        "crisis_label": "Samaritans",
        "crisis_url": "https://www.samaritans.org",
    },
    "IE": {
        "emergency": "112",
        "crisis_line": "1800 247 247",
        "crisis_label": "Pieta",
        "crisis_url": "https://www.pieta.ie",
    },
    "AU": {
        "emergency": "000",
        "crisis_line": "13 11 14",
        "crisis_label": "Lifeline Australia",
        "crisis_url": "https://www.lifeline.org.au",
    },
    "NZ": {
        "emergency": "111",
        "crisis_line": "1737",
        "crisis_label": "Need to Talk 1737",
        "crisis_url": "https://1737.org.nz",
    },
    "IN": {
        "emergency": "112",
        "crisis_line": "9152987821",
        "crisis_label": "AASRA",
        "crisis_url": "https://www.aasra.info",
    },
}

DEFAULT_RESOURCES = {
    "emergency": "your local emergency number",
    "crisis_line": "https://www.iasp.info/suicidalthoughts/",
    "crisis_label": "a local crisis line (IASP directory)",
    "crisis_url": "https://www.iasp.info/suicidalthoughts/",
}

REGION_LABELS: Dict[str, str] = {
    "US": "United States",
    "CA": "Canada",
    "UK": "United Kingdom",
    "IE": "Ireland",
    "AU": "Australia",
    "NZ": "New Zealand",
    "IN": "India",
    "INTL": "Other / international",
}


def list_crisis_regions() -> List[Dict[str, str]]:
    """UI-ready region list with local emergency and crisis numbers."""
    rows = []
    for code, label in REGION_LABELS.items():
        resources = dict(DEFAULT_RESOURCES)
        resources.update(REGION_RESOURCES.get(code, {}))
        rows.append(
            {
                "code": code,
                "label": label,
                "emergency": resources["emergency"],
                "crisis_line": resources["crisis_line"],
                "crisis_label": resources["crisis_label"],
                "crisis_url": resources["crisis_url"],
            }
        )
    return rows

IMMINENT_PATTERNS = [
    r"\b(right now|tonight|today|this (morning|afternoon|evening|hour))\b.*\b(kill myself|end it|suicide|overdose)\b",
    r"\b(kill myself|end it|suicide|overdose)\b.*\b(right now|tonight|today)\b",
    r"\b(i (have|made) (a |the )?plan)\b",
    r"\b(going to|gonna|about to) (kill myself|end my life|do it tonight)\b",
    r"\b(goodbye forever|this is goodbye|final goodbye)\b",
    r"\b(i have the )(pills|rope|gun|knife|razor)\b",
    r"\b(wrote|writing) (a |my )?suicide note\b",
]

SELF_HARM_PATTERNS = [
    r"\bi want to die\b",
    r"\bi don't want to live\b",
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\bend it all\b",
    r"\bsuicidal\b",
    r"\bsuicide\b",
    r"\bself[- ]harm\b",
    r"\bhurt myself\b",
    r"\bcut myself\b",
    r"\bwant to die\b",
    r"\bbetter off dead\b",
    r"\bno reason to live\b",
    r"\bdon't want to be (here|alive)\b",
]

DISTRESS_PATTERNS = [
    r"\bcan't go on\b",
    r"\bcan'?t take (this|it) anymore\b",
    r"\bgive up on (life|everything)\b",
    r"\bhopeless\b",
    r"\bworthless\b",
    r"\bnobody (cares|would miss me)\b",
    r"\bi feel empty\b",
    r"\bnothing matters\b",
]


@dataclass
class SafetyResult:
    is_crisis: bool
    confidence: float
    reason: Optional[str] = None
    response: Optional[str] = None
    tier: str = "none"  # none | distress | self_harm | imminent
    resources: Dict[str, str] = field(default_factory=dict)
    repeated: bool = False
    support_note: Optional[str] = None


class SafetyGuard:
    def __init__(self):
        rg = config.response_generation
        self.enabled = getattr(rg, "safety_enabled", True)
        self.threshold = getattr(rg, "safety_confidence_threshold", 0.6)
        self.crisis_message = getattr(rg, "crisis_message", "")
        self.region = str(getattr(rg, "safety_region", "US") or "US").upper()
        self.log_events = getattr(rg, "safety_log_events", True)
        self.log_user_text = getattr(rg, "safety_log_user_text", False)
        self.log_path = Path(getattr(rg, "safety_log_path", "logs/crisis_events.jsonl"))
        self._imminent = [re.compile(p, re.IGNORECASE) for p in IMMINENT_PATTERNS]
        self._self_harm = [re.compile(p, re.IGNORECASE) for p in SELF_HARM_PATTERNS]
        self._distress = [re.compile(p, re.IGNORECASE) for p in DISTRESS_PATTERNS]
        self._session_hits: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"distress": 0, "self_harm": 0, "imminent": 0}
        )
        self._lock = threading.Lock()

    def resources_for(self, region: Optional[str] = None) -> Dict[str, str]:
        key = str(region or self.region or "US").upper()
        base = dict(DEFAULT_RESOURCES)
        base.update(REGION_RESOURCES.get(key, {}))
        override = getattr(config.response_generation, "crisis_resources", None) or {}
        if isinstance(override, dict):
            base.update({k: str(v) for k, v in override.items() if v})
        return base

    def assess(self, text: str, session_id: Optional[str] = None) -> SafetyResult:
        if not self.enabled or not text:
            return SafetyResult(is_crisis=False, confidence=0.0)

        region = self.region
        resources = self.resources_for(region)
        raw_tier, reason, confidence = self._classify(text)
        session_key = session_id or "default"

        with self._lock:
            hits = self._session_hits[session_key]
            prior_crisis = hits["self_harm"] + hits["imminent"]
            prior_same = hits.get(raw_tier, 0) if raw_tier != "none" else 0
            if raw_tier != "none":
                hits[raw_tier] += 1

        repeated = prior_same >= 1
        escalate_repeat = raw_tier == "self_harm" and prior_crisis >= 1

        if raw_tier == "none":
            return SafetyResult(is_crisis=False, confidence=0.0, resources=resources)

        if raw_tier == "distress" and not repeated:
            result = SafetyResult(
                is_crisis=False,
                confidence=confidence,
                reason=reason,
                tier="distress",
                resources=resources,
                repeated=False,
            )
            self._audit(session_key, result)
            return result

        if raw_tier == "distress" and repeated:
            note = (
                f" If this feels too heavy to hold alone, {resources['crisis_label']} "
                f"is {resources['crisis_line']}."
            )
            result = SafetyResult(
                is_crisis=False,
                confidence=min(1.0, confidence + 0.15),
                reason=reason,
                tier="distress",
                resources=resources,
                repeated=True,
                support_note=note,
            )
            self._audit(session_key, result)
            return result

        # self_harm or imminent — do not continue as a normal chat
        if raw_tier == "imminent" or escalate_repeat:
            response = self._message_imminent(resources)
            tier = "imminent"
            reason = reason if raw_tier == "imminent" else "repeated_self_harm"
            confidence = max(confidence, 0.92)
        else:
            response = self._message_self_harm(resources)
            tier = "self_harm"

        result = SafetyResult(
            is_crisis=True,
            confidence=confidence,
            reason=reason,
            response=response,
            tier=tier,
            resources=resources,
            repeated=repeated or escalate_repeat,
        )
        self._audit(session_key, result, text=text)
        return result

    def _classify(self, text: str) -> Tuple[str, Optional[str], float]:
        if any(p.search(text) for p in self._imminent):
            return "imminent", "imminent_danger", 0.95
        if any(p.search(text) for p in self._self_harm):
            return "self_harm", "self_harm_detected", 0.88
        if any(p.search(text) for p in self._distress):
            return "distress", "acute_distress", 0.62
        return "none", None, 0.0

    def _message_self_harm(self, resources: Dict[str, str]) -> str:
        return (
            "I'm really glad you told me. I care about your safety, and I'm not an emergency service. "
            f"Please reach {resources['crisis_label']} at {resources['crisis_line']}. "
            f"If you might act on these thoughts, call {resources['emergency']} or go to the nearest emergency department. "
            "You don't have to face this alone."
        )

    def _message_imminent(self, resources: Dict[str, str]) -> str:
        return (
            "If you are in danger right now, please call "
            f"{resources['emergency']} or go to the nearest emergency department. "
            "I'm not a crisis service, and I want you to get real-time human help. "
            f"You can also contact {resources['crisis_label']} at {resources['crisis_line']}. "
            "Stay with someone if you can."
        )

    def _audit(self, session_id: str, result: SafetyResult, text: str = "") -> None:
        if not self.log_events:
            return
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12],
            "tier": result.tier,
            "reason": result.reason,
            "repeated": result.repeated,
            "region": self.region,
        }
        if self.log_user_text and text:
            payload["text"] = text[:240]
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except OSError as exc:
            logger.warning("Could not write crisis audit log: %s", exc)

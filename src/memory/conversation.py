"""Conversation memory and context management."""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import threading

from ..utils.config import config
from .crypto import EncryptedJsonStore, crypto_available
from .continuity import build_shareable_recap, clinician_trends

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    timestamp: str
    user_text: str
    user_emotion: str
    user_confidence: float
    assistant_response: str
    empathy_style: str
    response_confidence: float


@dataclass
class EmotionHistory:
    """History of detected emotions."""
    emotion: str
    confidence: float
    timestamp: str
    context: str


@dataclass
class ConversationSession:
    """Complete conversation session."""
    session_id: str
    start_time: str
    end_time: Optional[str]
    turns: List[ConversationTurn]
    emotion_history: List[EmotionHistory]
    user_preferences: Dict[str, Any]
    session_summary: str
    user_name: Optional[str] = None
    persist_consent: Optional[bool] = None  # None = not yet chosen; True = save; False = discard
    shareable_recap: str = ""
    safety_flags: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    clinician_share: bool = False


_REMEMBER_CUES = re.compile(
    r"\b(remember this|please remember|save this (session|conversation|chat)|keep this (session|conversation))\b",
    re.IGNORECASE,
)
_FORGET_CUES = re.compile(
    r"\b(forget this|don't (save|remember)|do not (save|remember)|keep this private|delete this session)\b",
    re.IGNORECASE,
)


class ConversationMemory:
    """Manages conversation history and context."""
    
    def __init__(self, conversation_file: Optional[str] = None, memory_key: Optional[bytes] = None):
        self.config = config.memory
        self.current_session: Optional[ConversationSession] = None
        self.conversation_file = Path(conversation_file or self.config.conversation_file)
        self.lock = threading.Lock()
        encrypt = bool(getattr(self.config, "encrypt", True))
        require_consent = bool(getattr(self.config, "require_consent", True))
        self.require_consent = require_consent
        self.encrypt_enabled = encrypt
        key_file = getattr(self.config, "key_file", "data/.memory_key")
        self._store = EncryptedJsonStore(
            self.conversation_file,
            encrypt=encrypt,
            key=memory_key,
            key_file=key_file,
        )
        
        # Ensure data directory exists
        self.conversation_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing conversations
        self.conversations: List[ConversationSession] = self._load_conversations()
    
    def start_new_session(self, session_id: Optional[str] = None) -> str:
        """Start a new conversation session."""
        with self.lock:
            if session_id is None:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.current_session = ConversationSession(
                session_id=session_id,
                start_time=datetime.now().isoformat(),
                end_time=None,
                turns=[],
                emotion_history=[],
                user_preferences={},
                session_summary="",
                user_name=None
            )
            
            logger.info(f"Started new conversation session: {session_id}")
            return session_id
    
    def end_current_session(self):
        """End the current conversation session."""
        with self.lock:
            if self.current_session:
                self.current_session.end_time = datetime.now().isoformat()
                self.current_session.session_summary = self._generate_session_summary()
                self.current_session.shareable_recap = build_shareable_recap(self.current_session)
                if self._should_persist_unlocked():
                    self._upsert_session_unlocked(self.current_session)
                    self._save_conversations()
                    logger.info(f"Ended and stored conversation session: {self.current_session.session_id}")
                else:
                    logger.info(
                        "Ended session %s without storing it (no persist consent)",
                        self.current_session.session_id,
                    )
                self.current_session = None
    
    def add_conversation_turn(self, 
                            user_text: str,
                            user_emotion: str,
                            user_confidence: float,
                            assistant_response: str,
                            empathy_style: str,
                            response_confidence: float):
        """Add a conversation turn to current session."""
        if not self.current_session:
            self.start_new_session()
        
        with self.lock:
            turn = ConversationTurn(
                timestamp=datetime.now().isoformat(),
                user_text=user_text,
                user_emotion=user_emotion,
                user_confidence=user_confidence,
                assistant_response=assistant_response,
                empathy_style=empathy_style,
                response_confidence=response_confidence
            )
            
            self.current_session.turns.append(turn)
            
            # Add to emotion history
            emotion_entry = EmotionHistory(
                emotion=user_emotion,
                confidence=user_confidence,
                timestamp=turn.timestamp,
                context=user_text[:100]  # First 100 chars as context
            )
            self.current_session.emotion_history.append(emotion_entry)
            
            # Limit history length
            if len(self.current_session.turns) > self.config.max_history_length:
                self.current_session.turns = self.current_session.turns[-self.config.max_history_length:]
            
            if len(self.current_session.emotion_history) > self.config.emotion_history_window:
                self.current_session.emotion_history = self.current_session.emotion_history[-self.config.emotion_history_window:]
            
            # Try to extract user name if not already known
            if not self.current_session.user_name:
                extracted_name = self._extract_user_name(user_text)
                if extracted_name:
                    self.current_session.user_name = extracted_name
                    logger.info(f"Learned user name: {extracted_name}")

            # Also try to extract name even if we have one (in case of correction)
            else:
                extracted_name = self._extract_user_name(user_text)
                if extracted_name and extracted_name != self.current_session.user_name:
                    logger.info(f"User name updated from {self.current_session.user_name} to {extracted_name}")
                    self.current_session.user_name = extracted_name

            # Extract user preferences
            preferences = self._extract_user_preferences(user_text)
            if preferences:
                self.current_session.user_preferences.update(preferences)

            self._apply_spoken_consent(user_text)

            # Auto-save only after explicit persist consent
            if self._should_persist_unlocked() and len(self.current_session.turns) % 5 == 0:
                self._upsert_session_unlocked(self.current_session)
                self._save_conversations()

    def get_fusion_text_boost(self) -> float:
        """Calibration 0–~0.45: higher means trust words over audio more."""
        if not self.current_session:
            return 0.0
        return float(self.current_session.user_preferences.get("fusion_text_boost", 0.0))

    def record_emotion_feedback(self, detected_match: bool, felt_emotion: Optional[str] = None):
        """User calibration for emotion detection (e.g. thumbs on detected mood)."""
        if not self.current_session:
            self.start_new_session()
        with self.lock:
            prefs = self.current_session.user_preferences
            boost = float(prefs.get("fusion_text_boost", 0.0))
            if not detected_match:
                boost = min(0.45, boost + 0.06)
            else:
                boost = max(0.0, boost - 0.03)
            prefs["fusion_text_boost"] = boost
            if felt_emotion:
                prefs["last_stated_emotion"] = felt_emotion

    def record_response_feedback(self, helpful: bool):
        """Soft calibration from whether the last reply felt helpful."""
        if not self.current_session:
            self.start_new_session()
        with self.lock:
            prefs = self.current_session.user_preferences
            boost = float(prefs.get("fusion_text_boost", 0.0))
            if helpful:
                prefs["fusion_text_boost"] = max(0.0, boost - 0.02)
            else:
                prefs["fusion_text_boost"] = min(0.45, boost + 0.03)

    def set_response_mode_preference(self, mode: Optional[str]):
        """Persist listen vs coach stance when user selects it in UI."""
        if mode not in {None, "", "listen", "coach"}:
            return
        if not self.current_session:
            self.start_new_session()
        with self.lock:
            if mode:
                self.current_session.user_preferences["response_mode"] = mode
            else:
                self.current_session.user_preferences.pop("response_mode", None)
    
    def get_conversation_context(self, window_size: Optional[int] = None) -> List[Dict[str, str]]:
        """Get recent conversation context."""
        if not self.current_session:
            return []
        
        window_size = window_size or self.config.context_window
        recent_turns = self.current_session.turns[-window_size:]
        
        context = []
        recap = self.current_session.user_preferences.get("resume_recap")
        if recap:
            context.append({
                "user": "(last visit recap)",
                "assistant": recap,
                "emotion": "neutral",
                "timestamp": self.current_session.start_time,
            })
        for turn in recent_turns:
            context.append({
                'user': turn.user_text,
                'assistant': turn.assistant_response,
                'emotion': turn.user_emotion,
                'timestamp': turn.timestamp
            })
        
        return context
    
    def get_emotion_patterns(self) -> Dict[str, Any]:
        """Analyze emotion patterns from history."""
        if not self.current_session or not self.current_session.emotion_history:
            return {}
        
        emotions = [entry.emotion for entry in self.current_session.emotion_history]
        confidences = [entry.confidence for entry in self.current_session.emotion_history]
        
        # Calculate emotion frequencies
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Most common emotion
        most_common_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Recent emotion trend
        recent_emotions = emotions[-5:] if len(emotions) >= 5 else emotions
        recent_trend = self._analyze_emotion_trend(recent_emotions)
        
        return {
            'most_common_emotion': most_common_emotion,
            'emotion_distribution': emotion_counts,
            'average_confidence': avg_confidence,
            'recent_trend': recent_trend,
            'total_interactions': len(emotions)
        }
    
    def _analyze_emotion_trend(self, emotions: List[str]) -> str:
        """Analyze trend in recent emotions."""
        if len(emotions) < 2:
            return 'stable'
        
        # Simple trend analysis
        positive_emotions = ['happy', 'calm']
        negative_emotions = ['sad', 'angry', 'anxious']
        
        positive_count = sum(1 for e in emotions if e in positive_emotions)
        negative_count = sum(1 for e in emotions if e in negative_emotions)
        
        if positive_count > negative_count:
            return 'improving'
        elif negative_count > positive_count:
            return 'declining'
        else:
            return 'stable'
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences from current session."""
        if not self.current_session:
            return {}
        
        return self.current_session.user_preferences.copy()
    
    def update_user_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences."""
        if not self.current_session:
            self.start_new_session()
        
        with self.lock:
            self.current_session.user_preferences.update(preferences)
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics for current session."""
        if not self.current_session:
            return {}
        
        turns = self.current_session.turns
        emotions = self.current_session.emotion_history
        
        if not turns:
            return {}
        
        # Calculate statistics
        session_duration = None
        if self.current_session.end_time:
            start = datetime.fromisoformat(self.current_session.start_time)
            end = datetime.fromisoformat(self.current_session.end_time)
            session_duration = (end - start).total_seconds() / 60  # minutes
        
        avg_response_confidence = sum(turn.response_confidence for turn in turns) / len(turns)
        avg_emotion_confidence = sum(turn.user_confidence for turn in turns) / len(turns)
        
        empathy_styles_used = {}
        for turn in turns:
            style = turn.empathy_style
            empathy_styles_used[style] = empathy_styles_used.get(style, 0) + 1
        
        return {
            'session_id': self.current_session.session_id,
            'total_turns': len(turns),
            'session_duration_minutes': session_duration,
            'average_response_confidence': avg_response_confidence,
            'average_emotion_confidence': avg_emotion_confidence,
            'empathy_styles_used': empathy_styles_used,
            'emotion_patterns': self.get_emotion_patterns()
        }
    
    def _generate_session_summary(self) -> str:
        """Generate a summary of the conversation session."""
        if not self.current_session or not self.current_session.turns:
            return "No conversation data available."
        
        stats = self.get_session_statistics()
        emotion_patterns = stats.get('emotion_patterns', {})
        
        summary_parts = []
        summary_parts.append(f"Session with {stats['total_turns']} interactions.")
        
        if emotion_patterns.get('most_common_emotion'):
            summary_parts.append(f"Primary emotion: {emotion_patterns['most_common_emotion']}.")
        
        if emotion_patterns.get('recent_trend'):
            summary_parts.append(f"Emotional trend: {emotion_patterns['recent_trend']}.")
        
        return " ".join(summary_parts)
    
    def may_persist(self) -> bool:
        """True when this session may be written to disk / long-term memory."""
        with self.lock:
            return self._should_persist_unlocked()

    def set_persist_consent(self, persist: bool) -> Dict[str, Any]:
        """Record remember / forget for the live session."""
        with self.lock:
            if not self.current_session:
                return {
                    "require_consent": self.require_consent,
                    "persist_consent": None,
                    "encrypted": self.encrypt_enabled and crypto_available(),
                    "stored": False,
                    "error": "no_active_session",
                }
            self.current_session.persist_consent = bool(persist)
            if persist:
                self.current_session.shareable_recap = build_shareable_recap(self.current_session)
                self._upsert_session_unlocked(self.current_session)
                self._save_conversations()
            else:
                sid = self.current_session.session_id
                self.conversations = [s for s in self.conversations if s.session_id != sid]
                self._save_conversations()
            return self.consent_status_unlocked()

    def consent_status(self) -> Dict[str, Any]:
        with self.lock:
            return self.consent_status_unlocked()

    def consent_status_unlocked(self) -> Dict[str, Any]:
        persist = None
        if self.current_session:
            persist = self.current_session.persist_consent
        return {
            "require_consent": self.require_consent,
            "persist_consent": persist,
            "encrypted": self.encrypt_enabled and crypto_available(),
            "stored": persist is True,
            "clinician_share": bool(self.current_session.clinician_share) if self.current_session else False,
        }

    def record_safety_flag(self, tier: str) -> None:
        if not self.current_session or not tier or tier == "none":
            return
        with self.lock:
            if tier not in self.current_session.safety_flags:
                self.current_session.safety_flags.append(tier)

    def record_tool(self, name: str) -> None:
        if not self.current_session or not name:
            return
        with self.lock:
            if name not in self.current_session.tools_used:
                self.current_session.tools_used.append(name)

    def set_clinician_share(self, enabled: bool) -> Dict[str, Any]:
        with self.lock:
            if not self.current_session:
                return {"clinician_share": False, "error": "no_active_session"}
            self.current_session.clinician_share = bool(enabled)
            if self._should_persist_unlocked():
                self._upsert_session_unlocked(self.current_session)
                self._save_conversations()
            return {"clinician_share": self.current_session.clinician_share}

    def shareable_recap(self, session: Optional[ConversationSession] = None) -> str:
        target = session or self.current_session
        if not target:
            return ""
        recap = build_shareable_recap(target)
        target.shareable_recap = recap
        return recap

    def last_persisted_preview(self) -> Optional[Dict[str, Any]]:
        stored = [s for s in self.conversations if s.persist_consent is True]
        if not stored:
            return None
        last = stored[-1]
        recap = last.shareable_recap or build_shareable_recap(last)
        return {
            "session_id": last.session_id,
            "start_time": last.start_time,
            "recap": recap,
            "turns": len(last.turns),
        }

    def resume_last_session(self) -> Dict[str, Any]:
        preview = self.last_persisted_preview()
        if not preview or not self.current_session:
            return {"resumed": False}
        last = self.get_session_by_id(preview["session_id"])
        with self.lock:
            if last:
                if last.user_name:
                    self.current_session.user_name = last.user_name
                self.current_session.user_preferences.update(last.user_preferences or {})
                self.current_session.user_preferences["resume_recap"] = preview["recap"]
                self.current_session.clinician_share = bool(last.clinician_share)
            return {"resumed": True, "recap": preview["recap"]}

    def clinician_trend_view(self) -> Dict[str, Any]:
        opted = any(s.clinician_share and s.persist_consent is True for s in self.conversations)
        if self.current_session and self.current_session.clinician_share:
            opted = True
        if not opted:
            return {"enabled": False, "includes_transcripts": False, "days": []}
        sessions = [s for s in self.conversations if s.persist_consent is True]
        if self.current_session and self.current_session.clinician_share:
            sessions = sessions + [self.current_session]
        payload = clinician_trends(sessions)
        payload["enabled"] = True
        return payload

    def _should_persist_unlocked(self) -> bool:
        if not self.config.save_conversations:
            return False
        if not self.current_session:
            return False
        if not self.require_consent:
            return self.current_session.persist_consent is not False
        return self.current_session.persist_consent is True

    def _upsert_session_unlocked(self, session: ConversationSession) -> None:
        self.conversations = [s for s in self.conversations if s.session_id != session.session_id]
        self.conversations.append(session)

    def _apply_spoken_consent(self, text: str) -> None:
        if not self.current_session or not text:
            return
        if _FORGET_CUES.search(text):
            self.current_session.persist_consent = False
            sid = self.current_session.session_id
            self.conversations = [s for s in self.conversations if s.session_id != sid]
            self._save_conversations()
            return
        if _REMEMBER_CUES.search(text):
            self.current_session.persist_consent = True
            self._upsert_session_unlocked(self.current_session)
            self._save_conversations()

    def _load_conversations(self) -> List[ConversationSession]:
        """Load conversations from file."""
        if not self.conversation_file.exists():
            return []
        
        try:
            data = self._store.read()
            if not isinstance(data, list):
                return []
            
            conversations = []
            for session_data in data:
                turns = [ConversationTurn(**turn) for turn in session_data.get('turns', [])]
                emotions = [EmotionHistory(**emotion) for emotion in session_data.get('emotion_history', [])]
                
                session = ConversationSession(
                    session_id=session_data['session_id'],
                    start_time=session_data['start_time'],
                    end_time=session_data.get('end_time'),
                    turns=turns,
                    emotion_history=emotions,
                    user_preferences=session_data.get('user_preferences', {}),
                    session_summary=session_data.get('session_summary', ''),
                    user_name=session_data.get('user_name'),
                    persist_consent=session_data.get('persist_consent', True),
                    shareable_recap=session_data.get('shareable_recap', ''),
                    safety_flags=session_data.get('safety_flags', []) or [],
                    tools_used=session_data.get('tools_used', []) or [],
                    clinician_share=bool(session_data.get('clinician_share', False)),
                )
                conversations.append(session)
            
            logger.info(f"Loaded {len(conversations)} conversation sessions")
            return conversations
            
        except Exception as e:
            logger.error(f"Failed to load conversations: {e}")
            return []
    
    def _save_conversations(self):
        """Save consented conversations to encrypted file."""
        if not self.config.save_conversations:
            return
        
        try:
            data = []
            for session in self.conversations:
                if session.persist_consent is False:
                    continue
                if self.require_consent and session.persist_consent is not True:
                    continue
                data.append(asdict(session))
            self._store.write(data)
            logger.debug(f"Saved {len(data)} conversation sessions")
            
        except Exception as e:
            logger.error(f"Failed to save conversations: {e}")
    
    def get_all_sessions(self) -> List[ConversationSession]:
        """Get all conversation sessions."""
        return self.conversations.copy()
    
    def get_session_by_id(self, session_id: str) -> Optional[ConversationSession]:
        """Get specific session by ID."""
        for session in self.conversations:
            if session.session_id == session_id:
                return session
        
        if self.current_session and self.current_session.session_id == session_id:
            return self.current_session
        
        return None
    
    def clear_all_conversations(self):
        """Clear all conversation history."""
        with self.lock:
            self.conversations.clear()
            self.current_session = None

            if self.conversation_file.exists():
                self.conversation_file.unlink()

            logger.info("Cleared all conversation history")

    def _extract_user_name(self, text: str) -> Optional[str]:
        """Extract user name from text using simple patterns."""
        import re

        text_lower = text.lower()

        # Common patterns for name introduction
        patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
            r"call me (\w+)",
            r"this is (\w+)",
            r"it's (\w+)",
            r"(\w+) here",
            r"hi,? i'm (\w+)",
            r"hello,? i'm (\w+)",
            r"name's (\w+)",
            r"i go by (\w+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1).capitalize()
                # Filter out common words that aren't names
                excluded_words = [
                    'good', 'fine', 'okay', 'well', 'here', 'there', 'doing', 'feeling',
                    'great', 'bad', 'sad', 'happy', 'tired', 'busy', 'free', 'ready',
                    'sorry', 'thanks', 'hello', 'hi', 'hey', 'yes', 'no', 'sure'
                ]
                if name.lower() not in excluded_words and len(name) > 1:
                    logger.info(f"Extracted name: {name} from text: {text}")
                    return name

        return None

    def _extract_user_preferences(self, text: str) -> Dict[str, Any]:
        """Extract user preferences like tone, verbosity, and language."""
        import re
        preferences: Dict[str, Any] = {}
        text_lower = text.lower()

        if re.search(r"\b(be|sound|respond)\b.*\bbrief|short\b", text_lower):
            preferences["verbosity"] = "brief"
        elif re.search(r"\b(be|sound|respond)\b.*\bverbose|detailed|long\b", text_lower):
            preferences["verbosity"] = "detailed"

        if "gentle tone" in text_lower or "soft tone" in text_lower:
            preferences["tone"] = "gentle"
        elif "direct tone" in text_lower or "straightforward" in text_lower:
            preferences["tone"] = "direct"
        elif "supportive tone" in text_lower or "encouraging" in text_lower:
            preferences["tone"] = "supportive"

        language_match = re.search(r"(speak|respond) in ([a-z\s]+)", text_lower)
        if language_match:
            preferences["language"] = language_match.group(2).strip()

        if re.search(r"\b(listen-?only|just reflect|don'?t (give )?advice)\b", text_lower):
            preferences["response_mode"] = "listen"
        elif re.search(r"\b(coaching mode|coping tips|practical (tips|steps))\b", text_lower):
            preferences["response_mode"] = "coach"

        return preferences

    def get_user_name(self) -> Optional[str]:
        """Get the current user's name if known."""
        if self.current_session:
            return self.current_session.user_name
        return None

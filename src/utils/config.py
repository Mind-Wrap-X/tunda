"""Configuration management for Empathic Voice Companion."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PerformanceConfig:
    profile: str = "balanced"
    latency_budget_ms: int = 1200
    latency_budget_min_ms: int = 600
    latency_budget_max_ms: int = 2000
    latency_ewma_alpha: float = 0.2
    adaptive_latency_enabled: bool = True
    prewarm_models: bool = True
    lazy_load: bool = False
    gpu_mem_min_gb_for_large_stt: float = 10.0
    gpu_mem_min_gb_for_large_llm: float = 12.0
    llm_models: Dict[str, str] = None

    def __post_init__(self):
        if self.llm_models is None:
            self.llm_models = {
                "fast": "llama3.1:8b",
                "balanced": "llama3.1:8b",
                "quality": "llama3.1:70b"
            }


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    silence_threshold: float = 0.01
    silence_duration: float = 2.0
    min_speech_seconds: float = 0.3
    max_segment_seconds: float = 20.0
    queue_maxsize: int = 10
    noise_reduction: bool = False


@dataclass
class SpeechRecognitionConfig:
    model: str = "base"
    language: str = "en"
    temperature: float = 0.0
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    device: Optional[str] = None
    compute_type: str = "auto"
    cpu_threads: Optional[int] = None
    num_workers: int = 1
    plugin: Optional[str] = None
    dynamic_model_switching: bool = False
    short_model: str = "small"
    long_model: str = "large-v3"
    short_utterance_seconds: float = 4.0
    vad_filter: bool = True
    vad_threshold: float = 0.5
    vad_min_silence_duration_ms: int = 300
    vad_speech_pad_ms: int = 200
    vad_adaptive: bool = True
    vad_adaptive_min: float = 0.3
    vad_adaptive_max: float = 0.8
    vad_adaptive_scale: float = 1.8
    vad_noise_percentile: float = 0.1
    initial_prompt: Optional[str] = None
    condition_on_previous_text: bool = True
    denoise: bool = True
    distress_prompt: str = (
        "Empathic conversation. The speaker may be crying, whispering, or on a phone. "
        "Transcribe the words faithfully."
    )


@dataclass
class EmotionDetectionConfig:
    enabled: bool = True
    model_type: str = "wav2vec2"
    confidence_threshold: float = 0.6
    dl_model_name: str = "superb/wav2vec2-base-superb-er"
    device: Optional[str] = None
    plugin: Optional[str] = None
    lazy_load: bool = False
    supported_emotions: list = None
    mfcc_coefficients: int = 13
    chroma_features: int = 12
    spectral_features: bool = True
    prosodic_features: bool = True
    window_size: int = 2048
    hop_length: int = 512

    def __post_init__(self):
        if self.supported_emotions is None:
            self.supported_emotions = ["happy", "sad", "angry", "anxious", "calm", "neutral"]


@dataclass
class ResponseGenerationConfig:
    llm_provider: str = "ollama"
    model_name: str = "llama3.1:8b"
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    default_style: str = "supportive"
    empathy_styles: Dict[str, str] = None
    max_memories: int = 3
    plugin: Optional[str] = None
    lazy_load: bool = False
    safety_enabled: bool = True
    safety_confidence_threshold: float = 0.6
    safety_region: str = "US"
    safety_log_events: bool = True
    safety_log_user_text: bool = False
    safety_log_path: str = "logs/crisis_events.jsonl"
    crisis_resources: Optional[Dict[str, str]] = None
    crisis_message: str = (
        "I'm really sorry you're feeling this way. You deserve support. "
        "If you are in immediate danger or thinking about harming yourself, "
        "please contact your local emergency number or a trusted person right now."
    )

    def __post_init__(self):
        if self.empathy_styles is None:
            self.empathy_styles = {
                "supportive": "Provide warm, encouraging responses that validate emotions",
                "reflective": "Mirror back emotions and help process feelings",
                "solution_focused": "Acknowledge emotions while gently guiding toward solutions",
                "therapeutic": "Use CBT-inspired techniques for emotional support"
            }


@dataclass
class TextToSpeechConfig:
    provider: str = "piper"
    voice_model: str = "en_US-lessac-medium"
    speaking_rate: float = 1.0
    pitch_adjustment: float = 0.0
    volume: float = 0.8
    emotion_adaptive: bool = True
    barge_in: bool = True
    barge_in_min_chunks: int = 4
    barge_in_energy_multiplier: float = 2.4
    barge_in_grace_ms: int = 350
    voice_mapping: Dict[str, str] = None
    lazy_load: bool = False
    streaming: bool = True
    stream_chunk_chars: int = 140
    first_chunk_chars: int = 64
    target_first_audio_ms: int = 800

    def __post_init__(self):
        if self.voice_mapping is None:
            self.voice_mapping = {
                "happy": "en_US-amy-medium",
                "sad": "en_US-lessac-low",
                "angry": "en_US-ryan-high",
                "anxious": "en_US-lessac-medium",
                "calm": "en_US-lessac-low",
                "neutral": "en_US-lessac-medium"
            }


@dataclass
class MemoryConfig:
    enabled: bool = True
    max_history_length: int = 50
    emotion_history_window: int = 10
    context_window: int = 5
    retrieval_limit: int = 3
    save_conversations: bool = True
    conversation_file: str = "data/conversations.json"
    encrypt: bool = True
    require_consent: bool = True
    key_file: str = "data/.memory_key"
    max_items: int = 2000
    prune_to: int = 1500
    prune_strategy: str = "global"  # global, per_topic
    importance_threshold: float = 0.2
    batch_embeddings: bool = True
    batch_size: int = 8
    batch_flush_seconds: float = 1.5


@dataclass
class ClinicConfig:
    profile: str = "companion"
    allow_web_profile_switch: bool = True


class Config:
    """Main configuration class for the Empathic Voice Companion."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config_data = {}
        self.load_config()
        self._apply_performance_profile()
        
        # Initialize configuration sections
        self.performance = self._create_performance_config()
        self.audio = self._create_audio_config()
        self.speech_recognition = self._create_speech_recognition_config()
        self.emotion_detection = self._create_emotion_detection_config()
        self.response_generation = self._create_response_generation_config()
        self.text_to_speech = self._create_text_to_speech_config()
        self.memory = self._create_memory_config()
        self.clinic = self._create_clinic_config()
    
    def load_config(self):
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f) or {}
        else:
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            self._config_data = {}
    
    def save_config(self):
        """Save current configuration to YAML file."""
        config_dict = {
            'performance': self.performance.__dict__,
            'audio': self.audio.__dict__,
            'speech_recognition': self.speech_recognition.__dict__,
            'emotion_detection': self.emotion_detection.__dict__,
            'response_generation': self.response_generation.__dict__,
            'text_to_speech': self.text_to_speech.__dict__,
            'memory': self.memory.__dict__
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _create_performance_config(self) -> PerformanceConfig:
        performance_data = self._config_data.get('performance', {})
        return PerformanceConfig(**performance_data)

    def _create_audio_config(self) -> AudioConfig:
        audio_data = self._config_data.get('audio', {})
        return AudioConfig(**audio_data)
    
    def _create_speech_recognition_config(self) -> SpeechRecognitionConfig:
        sr_data = dict(self._config_data.get('speech_recognition', {}))
        allowed = SpeechRecognitionConfig.__dataclass_fields__.keys()
        sr_data = {k: v for k, v in sr_data.items() if k in allowed}
        return SpeechRecognitionConfig(**sr_data)
    
    def _create_emotion_detection_config(self) -> EmotionDetectionConfig:
        ed_data = self._config_data.get('emotion_detection', {})
        feature_data = ed_data.get('feature_extraction', {})
        
        # Merge feature extraction settings
        merged_data = {**ed_data, **feature_data}
        merged_data.pop('feature_extraction', None)
        
        return EmotionDetectionConfig(**merged_data)
    
    def _create_response_generation_config(self) -> ResponseGenerationConfig:
        rg_data = dict(self._config_data.get('response_generation', {}))
        allowed = ResponseGenerationConfig.__dataclass_fields__.keys()
        rg_data = {k: v for k, v in rg_data.items() if k in allowed}
        return ResponseGenerationConfig(**rg_data)
    
    def _create_text_to_speech_config(self) -> TextToSpeechConfig:
        tts_data = dict(self._config_data.get('text_to_speech', {}))
        aliases = {
            "speaking_rate": "speaking_rate",
            "pitch_adjustment": "pitch_adjustment",
            "emotion_adaptive": "emotion_adaptive",
            "voice_model": "voice_model",
        }
        for src, dest in aliases.items():
            if src in tts_data and dest not in tts_data:
                tts_data[dest] = tts_data.pop(src)
            elif src in tts_data and src != dest:
                tts_data.pop(src, None)
        allowed = TextToSpeechConfig.__dataclass_fields__.keys()
        tts_data = {k: v for k, v in tts_data.items() if k in allowed}
        return TextToSpeechConfig(**tts_data)
    
    def _create_clinic_config(self) -> ClinicConfig:
        data = dict(self._config_data.get("clinic", {}))
        allowed = ClinicConfig.__dataclass_fields__.keys()
        data = {k: v for k, v in data.items() if k in allowed}
        return ClinicConfig(**data)

    def _create_memory_config(self) -> MemoryConfig:
        memory_data = dict(self._config_data.get('memory', {}))
        allowed = MemoryConfig.__dataclass_fields__.keys()
        memory_data = {k: v for k, v in memory_data.items() if k in allowed}
        return MemoryConfig(**memory_data)

    def _apply_performance_profile(self):
        """Apply performance profile overrides before creating configs."""
        performance = self._config_data.get("performance", {})
        profile = (performance.get("profile") or "balanced").lower()
        if profile not in {"fast", "balanced", "quality"}:
            profile = "balanced"
        performance["profile"] = profile
        self._config_data["performance"] = performance

        if profile == "balanced":
            return

        speech = self._config_data.setdefault("speech_recognition", {})
        emotion = self._config_data.setdefault("emotion_detection", {})
        response = self._config_data.setdefault("response_generation", {})
        tts = self._config_data.setdefault("text_to_speech", {})

        if profile == "fast":
            speech.setdefault("model", "small")
            speech.setdefault("beam_size", 1)
            speech.setdefault("best_of", 1)
            speech.setdefault("compute_type", "int8")
            emotion.setdefault("model_type", "random_forest")
            response.setdefault("max_tokens", 120)
            response.setdefault("temperature", 0.5)
            response.setdefault("top_p", 0.9)
            response.setdefault("repeat_penalty", 1.15)
            tts.setdefault("emotion_adaptive", False)
        elif profile == "quality":
            speech.setdefault("model", "large-v3")
            speech.setdefault("beam_size", 5)
            speech.setdefault("best_of", 5)
            speech.setdefault("compute_type", "float16")
            emotion.setdefault("model_type", "wav2vec2")
            response.setdefault("max_tokens", 220)
            response.setdefault("temperature", 0.7)
            response.setdefault("top_p", 0.9)
            response.setdefault("repeat_penalty", 1.1)
            tts.setdefault("emotion_adaptive", True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key path (e.g., 'web.host')."""
        keys = key.split('.')
        value = self._config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_model_path(self, model_type: str) -> Path:
        """Get the path for a specific model type."""
        models_config = self._config_data.get('models', {})
        base_path = Path(models_config.get(model_type, f"models/{model_type}"))
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path


# Global configuration instance
config = Config()

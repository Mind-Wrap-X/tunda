"""Speech recognition module using OpenAI Whisper."""

import whisper
import os
import time
import numpy as np
import torch
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import tempfile
import soundfile as sf
from dataclasses import dataclass

from ..utils.config import config
from ..utils.performance import latency_manager
from ..core.plugins import get_stt
from ..utils.audio import AudioProcessor

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result of speech transcription."""
    text: str
    language: str
    confidence: float
    segments: List[Dict[str, Any]]
    processing_time: float


class WhisperRecognizer:
    """Speech recognition using OpenAI Whisper."""
    
    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.audio_processor = AudioProcessor(sample_rate=16000)
        
        logger.info(f"Initializing Whisper with model '{model_name}' on device '{self.device}'")
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info(f"Whisper model '{self.model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(self, 
                  audio: np.ndarray, 
                  language: Optional[str] = None,
                  temperature: float = 0.0,
                  beam_size: int = 5,
                  best_of: int = 5,
                  patience: float = 1.0,
                  vad_filter: bool = True,
                  vad_threshold: float = 0.5,
                  vad_min_silence_duration_ms: int = 300,
                  vad_speech_pad_ms: int = 200,
                  initial_prompt: Optional[str] = None,
                  condition_on_previous_text: bool = True) -> TranscriptionResult:
        """Transcribe audio to text."""
        import time
        start_time = time.time()
        
        try:
            # Ensure audio is in the correct format
            audio = self.audio_processor.normalize_audio(audio)
            
            # Whisper expects audio to be float32
            audio = audio.astype(np.float32)
            
            # Transcribe using Whisper
            _ = vad_filter, vad_threshold, vad_min_silence_duration_ms, vad_speech_pad_ms
            result = self.model.transcribe(
                audio,
                language=language,
                temperature=temperature,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                initial_prompt=initial_prompt,
                condition_on_previous_text=condition_on_previous_text,
                verbose=False
            )
            
            processing_time = time.time() - start_time
            
            # Calculate average confidence from segments
            confidence = 0.0
            if result.get('segments'):
                confidences = [seg.get('avg_logprob', 0.0) for seg in result['segments']]
                confidence = np.mean(confidences) if confidences else 0.0
                # Convert log probability to confidence (approximate)
                confidence = max(0.0, min(1.0, (confidence + 1.0) / 2.0))
            
            return TranscriptionResult(
                text=result['text'].strip(),
                language=result.get('language', 'unknown'),
                confidence=confidence,
                segments=result.get('segments', []),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptionResult(
                text="",
                language="unknown",
                confidence=0.0,
                segments=[],
                processing_time=time.time() - start_time
            )
    
    def transcribe_file(self, file_path: str, **kwargs) -> TranscriptionResult:
        """Transcribe audio file."""
        try:
            audio, _ = self.audio_processor.load_audio(file_path)
            return self.transcribe(audio, **kwargs)
        except Exception as e:
            logger.error(f"Failed to transcribe file {file_path}: {e}")
            raise
    
    def is_speech_detected(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """Check if speech is detected in audio."""
        if len(audio) == 0:
            return False
        
        # Simple energy-based detection
        rms = self.audio_processor.calculate_rms(audio)
        return rms > threshold
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(whisper.tokenizer.LANGUAGES.keys())


class FasterWhisperRecognizer:
    """Speech recognition using Faster-Whisper for better performance."""
    
    def __init__(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        compute_type: str = "auto",
        cpu_threads: Optional[int] = None,
        num_workers: int = 1
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if compute_type == "auto":
            compute_type = "float16" if self.device == "cuda" else "int8"
        self.compute_type = compute_type
        if cpu_threads is None and self.device == "cpu":
            cpu_count = os.cpu_count() or 2
            cpu_threads = max(1, cpu_count - 1)
        self.cpu_threads = cpu_threads
        self.num_workers = max(1, num_workers)
        self.model = None
        self.audio_processor = AudioProcessor(sample_rate=16000)
        
        try:
            from faster_whisper import WhisperModel
            self.WhisperModel = WhisperModel
            logger.info(f"Using Faster-Whisper with model '{model_name}' on device '{self.device}'")
            self._load_model()
        except ImportError:
            logger.warning("Faster-Whisper not available, falling back to standard Whisper")
            self.model = None
    
    def _load_model(self):
        """Load the Faster-Whisper model."""
        try:
            self.model = self.WhisperModel(
                self.model_name, 
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=self.num_workers
            )
            logger.info(f"Faster-Whisper model '{self.model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper model: {e}")
            self.model = None
    
    def transcribe(self, 
                  audio: np.ndarray, 
                  language: Optional[str] = None,
                  temperature: float = 0.0,
                  beam_size: int = 5,
                  best_of: int = 5,
                  patience: float = 1.0,
                  vad_filter: bool = True,
                  vad_threshold: float = 0.5,
                  vad_min_silence_duration_ms: int = 300,
                  vad_speech_pad_ms: int = 200,
                  initial_prompt: Optional[str] = None,
                  condition_on_previous_text: bool = True) -> TranscriptionResult:
        """Transcribe audio to text using Faster-Whisper."""
        if self.model is None:
            raise RuntimeError("Faster-Whisper model not loaded")
        
        import time
        start_time = time.time()
        
        try:
            # Ensure audio is in the correct format
            audio = self.audio_processor.normalize_audio(audio)
            audio = audio.astype(np.float32)
            
            # Transcribe using Faster-Whisper
            vad_parameters = {
                "threshold": vad_threshold,
                "min_silence_duration_ms": vad_min_silence_duration_ms,
                "speech_pad_ms": vad_speech_pad_ms
            }
            segments, info = self.model.transcribe(
                audio,
                language=language,
                temperature=temperature,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters,
                initial_prompt=initial_prompt,
                condition_on_previous_text=condition_on_previous_text
            )
            
            # Collect segments
            segment_list = []
            full_text = ""
            confidences = []
            
            for segment in segments:
                segment_dict = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text,
                    'avg_logprob': segment.avg_logprob
                }
                segment_list.append(segment_dict)
                full_text += segment.text
                confidences.append(segment.avg_logprob)
            
            processing_time = time.time() - start_time
            
            # Calculate average confidence
            confidence = 0.0
            if confidences:
                confidence = np.mean(confidences)
                # Convert log probability to confidence (approximate)
                confidence = max(0.0, min(1.0, (confidence + 1.0) / 2.0))
            
            return TranscriptionResult(
                text=full_text.strip(),
                language=info.language,
                confidence=confidence,
                segments=segment_list,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Faster-Whisper transcription failed: {e}")
            return TranscriptionResult(
                text="",
                language="unknown",
                confidence=0.0,
                segments=[],
                processing_time=time.time() - start_time
            )

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # Faster-Whisper supports the same languages as Whisper
        try:
            import whisper
            return list(whisper.tokenizer.LANGUAGES.keys())
        except ImportError:
            # Fallback list of common languages
            return ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']

    def transcribe_file(self, file_path: str, **kwargs) -> TranscriptionResult:
        """Transcribe audio file."""
        try:
            from ..utils.audio import AudioProcessor
            audio_processor = AudioProcessor(sample_rate=16000)
            audio, _ = audio_processor.load_audio(file_path)
            return self.transcribe(audio, **kwargs)
        except Exception as e:
            logger.error(f"Failed to transcribe file {file_path}: {e}")
            raise

    def is_speech_detected(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """Check if speech is detected in audio."""
        if len(audio) == 0:
            return False

        # Simple energy-based detection
        from ..utils.audio import AudioProcessor
        audio_processor = AudioProcessor()
        rms = audio_processor.calculate_rms(audio)
        return rms > threshold


class SpeechRecognitionPipeline:
    """Main speech recognition pipeline."""
    
    def __init__(self):
        self.config = config.speech_recognition
        self.performance = config.performance
        self.recognizer = None
        self.short_recognizer = None
        self.long_recognizer = None
        self.device = None
        self.lazy_load = self.performance.lazy_load
        if not self.lazy_load:
            self._initialize_recognizer()
    
    def _resolve_device(self) -> str:
        if self.config.device:
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def _resolve_model_name(self, device: str) -> str:
        model_name = self.config.model
        if model_name == "auto":
            return "large-v3" if device == "cuda" else "small"
        return model_name
    
    def _initialize_recognizer(self):
        """Initialize the appropriate recognizer."""
        self.device = self._resolve_device()
        model_name = self._resolve_model_name(self.device)
        plugin_name = self.config.plugin
        if plugin_name:
            factory = get_stt(plugin_name)
            if factory:
                self.recognizer = factory()
                return
        if self.config.dynamic_model_switching and self.device == "cuda":
            self.short_recognizer = self._create_recognizer(self.config.short_model, self.device)
            if self._has_large_stt_budget():
                self.long_recognizer = self._create_recognizer(self.config.long_model, self.device)
            else:
                self.long_recognizer = self.short_recognizer
            if self.short_recognizer and self.long_recognizer:
                self.recognizer = self.short_recognizer
                return
        try:
            # Try Faster-Whisper first for better performance
            self.recognizer = self._create_recognizer(model_name, self.device)
            if self.recognizer.model is None:
                raise RuntimeError("Faster-Whisper not available")
            logger.info("Using Faster-Whisper for speech recognition")
        except:
            # Fall back to standard Whisper
            self.recognizer = WhisperRecognizer(
                model_name=model_name,
                device=self.device
            )
            logger.info("Using standard Whisper for speech recognition")
    
    def _create_recognizer(self, model_name: str, device: str):
        try:
            recognizer = FasterWhisperRecognizer(
                model_name=model_name,
                device=device,
                compute_type=self.config.compute_type,
                cpu_threads=self.config.cpu_threads,
                num_workers=self.config.num_workers
            )
            if recognizer.model is None:
                raise RuntimeError("Faster-Whisper not available")
            return recognizer
        except Exception:
            return WhisperRecognizer(model_name=model_name, device=device)

    def _choose_recognizer(self, audio: np.ndarray):
        if self.recognizer is None:
            self._initialize_recognizer()
        if self.short_recognizer and self.long_recognizer:
            duration = len(audio) / 16000
            if self._is_latency_sensitive():
                return self.short_recognizer
            if duration <= self.config.short_utterance_seconds:
                return self.short_recognizer
            return self.long_recognizer
        return self.recognizer
    
    def _adaptive_vad_threshold(self, audio: np.ndarray) -> float:
        if not self.config.vad_adaptive:
            return self.config.vad_threshold
        if len(audio) == 0:
            return self.config.vad_threshold
        abs_audio = np.abs(audio.astype(np.float32))
        noise_floor = np.percentile(abs_audio, max(1.0, self.config.vad_noise_percentile * 100))
        threshold = max(self.config.vad_threshold, noise_floor * self.config.vad_adaptive_scale)
        return float(np.clip(threshold, self.config.vad_adaptive_min, self.config.vad_adaptive_max))
    
    def _is_latency_sensitive(self) -> bool:
        budget_ms = latency_manager.get_budget_ms()
        return budget_ms <= 800
    
    def _has_large_stt_budget(self) -> bool:
        if self.device != "cuda":
            return False
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            return free_gb >= getattr(self.performance, "gpu_mem_min_gb_for_large_stt", 10.0)
        except Exception:
            return False
    
    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe audio using configured settings."""
        start = time.time()
        if getattr(self.config, "denoise", True):
            try:
                from .enhance import enhance_for_stt
                audio = enhance_for_stt(audio, sample_rate=16000)
            except Exception as exc:
                logger.warning("STT denoise skipped: %s", exc)
        recognizer = self._choose_recognizer(audio)
        vad_threshold = self._adaptive_vad_threshold(audio)
        prompt = self.config.initial_prompt or getattr(self.config, "distress_prompt", None)
        result = recognizer.transcribe(
            audio,
            language=self.config.language if self.config.language != "auto" else None,
            temperature=self.config.temperature,
            beam_size=self.config.beam_size,
            best_of=self.config.best_of,
            patience=self.config.patience,
            vad_filter=self.config.vad_filter,
            vad_threshold=vad_threshold,
            vad_min_silence_duration_ms=self.config.vad_min_silence_duration_ms,
            vad_speech_pad_ms=self.config.vad_speech_pad_ms,
            initial_prompt=prompt,
            condition_on_previous_text=self.config.condition_on_previous_text
        )
        latency_manager.observe("stt", (time.time() - start) * 1000.0)
        return result

    def warm_up(self):
        """Warm up STT models."""
        dummy_audio = np.zeros(16000, dtype=np.float32)
        try:
            self.transcribe(dummy_audio)
        except Exception:
            pass

    def health_check(self) -> Dict[str, Any]:
        recognizer = self.recognizer or self.short_recognizer or self.long_recognizer
        if recognizer is None:
            return {"available": False, "engine": "none"}
        engine = recognizer.__class__.__name__
        return {"available": True, "engine": engine}
    
    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """Transcribe audio file."""
        return self.recognizer.transcribe_file(file_path)
    
    def is_speech_detected(self, audio: np.ndarray) -> bool:
        """Check if speech is detected in audio."""
        return self.recognizer.is_speech_detected(audio)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.recognizer.get_supported_languages()

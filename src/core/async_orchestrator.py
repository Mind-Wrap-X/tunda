"""
Async streaming orchestrator using asyncio.
"""

import asyncio
import logging
import threading
import time
import numpy as np
import pyaudio
import collections

from ..utils.audio import AudioProcessor
from ..emotion.detector import EmotionDetector
from ..emotion.fusion import EmotionFusion
from ..speech.recognition import SpeechRecognitionPipeline
from ..response.generator import EmpathicResponseGenerator, ResponseContext
from ..speech.synthesis import TextToSpeechPipeline
from ..memory.vector import VectorMemory
from ..memory.conversation import ConversationMemory
from ..utils.config import config
from ..utils.tracing import new_trace_id, log_event

logger = logging.getLogger(__name__)


class AsyncStreamOrchestrator:
    def __init__(self):
        self.config = config
        self.running = False
        self.loop = None

        self.audio_processor = AudioProcessor(sample_rate=self.config.audio.sample_rate)
        self.emotion_detector = EmotionDetector()
        self.emotion_fusion = EmotionFusion()
        self.stt = SpeechRecognitionPipeline()
        self.llm = EmpathicResponseGenerator()
        self.tts = TextToSpeechPipeline()
        self.memory = VectorMemory()
        self.conversation_memory = ConversationMemory() if self.config.memory.enabled else None

        self.pyaudio_instance = pyaudio.PyAudio()
        self.input_stream = None
        self.sample_rate = self.config.audio.sample_rate
        self.chunk_size = self.config.audio.chunk_size

        self.audio_buffer = collections.deque(maxlen=int(self.sample_rate * 30 / self.chunk_size))
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_counter = 0
        self.silence_threshold = self.config.audio.silence_threshold
        self.silence_threshold_chunks = int(self.config.audio.silence_duration * self.sample_rate / self.chunk_size)
        self.min_speech_seconds = self.config.get("audio.min_speech_seconds", 0.3)
        self.max_segment_seconds = self.config.get("audio.max_segment_seconds", 20.0)

        self.transcription_queue = asyncio.Queue(maxsize=self.config.get("audio.queue_maxsize", 10))
        self.synthesis_queue = asyncio.Queue(maxsize=self.config.get("audio.queue_maxsize", 10))
        self.tts_playing = False
        self.tts_started_at = 0.0
        self.barge_in_event = threading.Event()
        self._barge_in_hits = 0
        tts_cfg = self.config.text_to_speech
        self.barge_in_enabled = getattr(tts_cfg, "barge_in", True)
        self.barge_in_min_chunks = getattr(tts_cfg, "barge_in_min_chunks", 4)
        self.barge_in_energy_multiplier = getattr(tts_cfg, "barge_in_energy_multiplier", 2.4)
        self.barge_in_grace_s = getattr(tts_cfg, "barge_in_grace_ms", 350) / 1000.0

        self.process_task = None
        self.speak_task = None

    async def start(self):
        self.running = True
        self.loop = asyncio.get_running_loop()
        if self.conversation_memory:
            self.conversation_memory.start_new_session()
        if self.config.performance.prewarm_models:
            self._prewarm_models()

        self.input_stream = self.pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=self.config.audio.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.config.audio.input_device,
            stream_callback=self._audio_callback
        )
        self.input_stream.start_stream()

        self.process_task = asyncio.create_task(self._process_loop())
        self.speak_task = asyncio.create_task(self._speak_loop())
        logger.info("Async orchestrator started.")

    async def stop(self):
        self.running = False
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.conversation_memory:
            self.conversation_memory.end_current_session()
        for task in [self.process_task, self.speak_task]:
            if task:
                task.cancel()
        self.pyaudio_instance.terminate()
        logger.info("Async orchestrator stopped.")

    def stop_sync(self):
        try:
            import asyncio
            asyncio.run(self.stop())
        except RuntimeError:
            pass

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if not self.running:
            return (None, pyaudio.paComplete)
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)
        rms = self.audio_processor.calculate_rms(audio_chunk)
        self._maybe_barge_in(rms)
        if rms > self.silence_threshold:
            self.is_speaking = True
            self.silence_counter = 0
            self.speech_buffer.append(audio_chunk)
        else:
            if self.is_speaking:
                self.silence_counter += 1
                self.speech_buffer.append(audio_chunk)
                if self.silence_counter > self.silence_threshold_chunks:
                    full_audio = np.concatenate(self.speech_buffer)
                    if len(full_audio) / self.sample_rate >= self.min_speech_seconds:
                        if self.loop:
                            self.loop.call_soon_threadsafe(self._enqueue_audio, full_audio)
                    self.speech_buffer = []
                    self.is_speaking = False
                    self.silence_counter = 0
        if self.is_speaking:
            if len(self.speech_buffer) * self.chunk_size / self.sample_rate > self.max_segment_seconds:
                full_audio = np.concatenate(self.speech_buffer)
                if self.loop:
                    self.loop.call_soon_threadsafe(self._enqueue_audio, full_audio)
                self.speech_buffer = []
                self.is_speaking = False
                self.silence_counter = 0

        return (in_data, pyaudio.paContinue)

    def _maybe_barge_in(self, rms: float) -> None:
        if not self.barge_in_enabled or not self.tts_playing:
            self._barge_in_hits = 0
            return
        if (time.monotonic() - self.tts_started_at) < self.barge_in_grace_s:
            return
        if rms > self.silence_threshold * self.barge_in_energy_multiplier:
            self._barge_in_hits += 1
            if self._barge_in_hits >= self.barge_in_min_chunks:
                self._request_barge_in()
        else:
            self._barge_in_hits = 0

    def _request_barge_in(self) -> None:
        if self.barge_in_event.is_set():
            return
        self.barge_in_event.set()
        self.tts_playing = False
        logger.info("Barge-in: patient interrupted spoken reply")
        if self.loop:
            self.loop.call_soon_threadsafe(self._drain_synthesis_queue)

    def _drain_synthesis_queue(self) -> None:
        while not self.synthesis_queue.empty():
            try:
                self.synthesis_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _enqueue_audio(self, audio):
        if self.transcription_queue.full():
            return
        self.transcription_queue.put_nowait(audio)

    async def _process_loop(self):
        while self.running:
            audio = await self.transcription_queue.get()
            trace_id = new_trace_id()
            log_event(logger, "segment_received", trace_id, duration_s=len(audio)/self.sample_rate)
            audio = self.audio_processor.normalize_audio(audio)
            audio = self.audio_processor.remove_silence(audio, threshold=self.silence_threshold)
            if len(audio) / self.sample_rate < self.min_speech_seconds:
                continue
            transcription = self.stt.transcribe(audio)
            text = transcription.text.strip()
            if not text:
                continue
            emotion_pred = self.emotion_detector.predict_emotion(audio)
            fusion_boost = self.conversation_memory.get_fusion_text_boost() if self.conversation_memory else 0.0
            emotion_pred = self.emotion_fusion.fuse(
                emotion_pred, text, transcription.confidence, text_boost=fusion_boost
            )

            relevant_memories = self.memory.retrieve_relevant(text, limit=self.config.get("memory.retrieval_limit", 3))
            conversation_history = []
            user_name = "Friend"
            user_preferences = {}
            if self.conversation_memory:
                conversation_history = self.conversation_memory.get_conversation_context()
                user_name = self.conversation_memory.get_user_name() or user_name
                user_preferences = self.conversation_memory.get_user_preferences()
            user_preferences = {**user_preferences, "user_name": user_name}

            context = ResponseContext(
                user_text=text,
                emotion=emotion_pred.emotion,
                confidence=emotion_pred.confidence,
                conversation_history=conversation_history,
                empathy_style=config.response_generation.default_style,
                user_preferences=user_preferences,
                relevant_memories=relevant_memories
            )
            token_stream = self.llm.generate_response_stream(context)

            current_sentence = ""
            full_response = ""
            for token in token_stream:
                full_response += token
                current_sentence += token
                if any(punct in token for punct in ['.', '!', '?']):
                    if not self.synthesis_queue.full():
                        await self.synthesis_queue.put((current_sentence, emotion_pred.emotion))
                    current_sentence = ""
            if current_sentence.strip() and not self.synthesis_queue.full():
                await self.synthesis_queue.put((current_sentence, emotion_pred.emotion))

            full_response = full_response.strip()
            if full_response:
                if not self.conversation_memory or self.conversation_memory.may_persist():
                    self.memory.add_memory(
                        text=f"User: {text}\nAssistant: {full_response}",
                        metadata={"emotion": emotion_pred.emotion}
                    )
                if self.conversation_memory:
                    self.conversation_memory.add_conversation_turn(
                        user_text=text,
                        user_emotion=emotion_pred.emotion,
                        user_confidence=emotion_pred.confidence,
                        assistant_response=full_response,
                        empathy_style=context.empathy_style,
                        response_confidence=0.8
                    )
            log_event(logger, "response_complete", trace_id, response=full_response[:200])

    async def _speak_loop(self):
        while self.running:
            text, emotion = await self.synthesis_queue.get()
            if self.barge_in_event.is_set():
                self.barge_in_event.clear()
                continue
            self.tts_playing = True
            self.tts_started_at = time.monotonic()
            self._barge_in_hits = 0
            try:
                if self.config.text_to_speech.streaming:
                    for chunk_result in self.tts.synthesize_stream(text, emotion=emotion):
                        if self.barge_in_event.is_set() or not self.running:
                            break
                        if chunk_result.success and len(chunk_result.audio) > 0:
                            await asyncio.to_thread(
                                self._play_audio, chunk_result.audio, chunk_result.sample_rate
                            )
                else:
                    result = self.tts.synthesize(text, emotion=emotion)
                    if result.success and len(result.audio) > 0:
                        await asyncio.to_thread(self._play_audio, result.audio, result.sample_rate)
            finally:
                self.tts_playing = False
                if self.barge_in_event.is_set():
                    self._drain_synthesis_queue()
                    self.barge_in_event.clear()

    def _play_audio(self, audio: np.ndarray, sample_rate: int):
        if self.barge_in_event.is_set() or not self.running:
            return
        output_stream = self.pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            output=True,
            output_device_index=self.config.audio.output_device
        )
        chunk = 1024
        samples = audio.astype(np.float32)
        try:
            for i in range(0, len(samples), chunk):
                if self.barge_in_event.is_set() or not self.running:
                    break
                output_stream.write(samples[i:i + chunk].tobytes())
        finally:
            output_stream.stop_stream()
            output_stream.close()

    def _prewarm_models(self):
        try:
            self.stt.warm_up()
        except Exception:
            pass
        try:
            self.emotion_detector.warm_up()
        except Exception:
            pass
        try:
            self.llm.warm_up()
        except Exception:
            pass
        try:
            self.tts.warm_up()
        except Exception:
            pass

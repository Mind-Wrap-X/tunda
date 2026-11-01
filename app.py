#!/usr/bin/env python3
"""
Web interface for the Empathic Voice Companion.
"""

import sys
import logging
from pathlib import Path
import json
import asyncio
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import uvicorn
import base64
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import config
from src.utils.performance import latency_manager, get_gpu_stats
from src.utils.tracing import new_trace_id, log_event
from src.emotion.fusion import EmotionFusion
from src.emotion.types import EmotionPrediction
from src.speech.recognition import SpeechRecognitionPipeline
from src.emotion.detector import EmotionDetector
from src.response.generator import EmpathicResponseGenerator, ResponseContext
from src.speech.synthesis import TextToSpeechPipeline
from src.speech.voice import speech_settings_for
from src.speech.stream import wav_bytes_to_float
from src.memory.conversation import ConversationMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Empathic Voice Companion", version="1.0.0")

# Health check endpoint for Render
@app.post("/api/tts/stream")
async def tts_stream(request: Request):
    """Stream WAV chunks; first clause is synthesized first for low latency."""
    if not tts_pipeline:
        return JSONResponse({"error": "tts_unavailable"}, status_code=503)
    body = await request.json()
    text = (body.get("text") or "").strip()
    emotion = body.get("emotion")
    if not text:
        return JSONResponse({"error": "empty_text"}, status_code=400)

    def generate():
        for event in tts_pipeline.iter_stream_events(text, emotion=emotion):
            yield json.dumps(event) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/api/stt")
async def stt_from_wav(request: Request):
    """Transcribe a 16-bit WAV body (base64) with denoise + distress prompt."""
    if not speech_recognizer:
        return JSONResponse({"error": "stt_unavailable"}, status_code=503)
    body = await request.json()
    b64 = body.get("audio_b64") or ""
    try:
        wav = base64.b64decode(b64)
        audio = wav_bytes_to_float(wav, target_rate=16000)
    except Exception as exc:
        return JSONResponse({"error": f"bad_wav: {exc}"}, status_code=400)
    result = speech_recognizer.transcribe(audio)
    return {
        "text": result.text,
        "confidence": result.confidence,
        "language": result.language,
        "processing_time": result.processing_time,
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Tunda Voice Companion"}

# Setup templates and static files
import os

# Create directories if they don't exist
templates_dir = "web/templates"
static_dir = "web/static"

if not os.path.exists(templates_dir):
    os.makedirs(templates_dir, exist_ok=True)
    logger.info(f"Created templates directory: {templates_dir}")

if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
    logger.info(f"Created static directory: {static_dir}")

templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize components
try:
    speech_recognizer = SpeechRecognitionPipeline()
    emotion_detector = EmotionDetector()
    response_generator = EmpathicResponseGenerator()
    tts_pipeline = TextToSpeechPipeline()
    conversation_memory = ConversationMemory()
    emotion_fusion = EmotionFusion()
    if config.performance.prewarm_models:
        speech_recognizer.warm_up()
        emotion_detector.warm_up()
        response_generator.warm_up()
        tts_pipeline.warm_up()
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    speech_recognizer = None
    emotion_detector = None
    response_generator = None
    tts_pipeline = None
    conversation_memory = None
    emotion_fusion = None


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(json.dumps(message))


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Landing page."""
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/app", response_class=HTMLResponse)
async def app_home(request: Request):
    """Main chat application."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def get_status():
    """Get system status."""
    return {
        "status": "running",
        "components": {
            "speech_recognition": speech_recognizer is not None,
            "emotion_detection": emotion_detector is not None,
            "response_generation": response_generator is not None,
            "text_to_speech": tts_pipeline is not None and tts_pipeline.is_available(),
            "conversation_memory": conversation_memory is not None
        },
        "llm_available": response_generator.is_llm_available() if response_generator else False
    }


@app.get("/api/health")
async def get_health_details():
    """Get component health details."""
    return {
        "status": "running",
        "components": {
            "speech_recognition": speech_recognizer.health_check() if speech_recognizer else {"available": False},
            "emotion_detection": {"available": emotion_detector is not None},
            "response_generation": response_generator.health_check() if response_generator else {"available": False},
            "text_to_speech": tts_pipeline.health_check() if tts_pipeline else {"available": False},
            "conversation_memory": conversation_memory is not None
        },
        "llm_available": response_generator.is_llm_available() if response_generator else False
    }


@app.get("/api/clinic/status")
async def get_clinic_status():
    from src.clinic.profiles import get_clinic_profile, list_clinic_profiles

    profile = get_clinic_profile(getattr(getattr(config, "clinic", None), "profile", "companion"))
    llm_online = response_generator.is_llm_available() if response_generator else False
    return {
        "profile": profile.profile_id,
        "label": profile.label,
        "care_mode": profile.care_mode,
        "allow_template_fallback": profile.allow_template_fallback,
        "allow_web_profile_switch": getattr(getattr(config, "clinic", None), "allow_web_profile_switch", True),
        "llm_online": llm_online,
        "care_ready": (not profile.care_mode) or llm_online,
        "profiles": list_clinic_profiles(),
        "offline_message": None if ((not profile.care_mode) or llm_online) else "Ollama is offline. Not in care mode.",
    }


@app.get("/api/recap")
async def get_recap():
    if not conversation_memory:
        return {"recap": ""}
    return {"recap": conversation_memory.shareable_recap(), "includes_transcripts": False}


@app.get("/api/clinician/trends")
async def get_clinician_trends():
    if not conversation_memory:
        return {"enabled": False, "days": [], "includes_transcripts": False}
    return conversation_memory.clinician_trend_view()


@app.post("/api/memory/wipe")
async def wipe_memory():
    if not conversation_memory:
        return {"wiped": False}
    conversation_memory.clear_all_conversations()
    return {"wiped": True, "includes_transcripts": False}


@app.get("/api/memory/status")
async def get_memory_status():
    """Consent and encryption status for stored conversations."""
    if not conversation_memory:
        return {"available": False}
    status = conversation_memory.consent_status()
    status["available"] = True
    return status


@app.get("/api/crisis-regions")
async def get_crisis_regions():
    """Local emergency and crisis numbers by region."""
    from src.response.safety import list_crisis_regions

    default_region = getattr(config.response_generation, "safety_region", "US")
    return {
        "default": default_region,
        "regions": list_crisis_regions(),
    }


@app.get("/api/performance")
async def get_performance():
    """Get live performance stats."""
    return {
        "latency": latency_manager.snapshot(),
        "gpu": get_gpu_stats()
    }


@app.get("/api/conversations")
async def get_conversations():
    """Get conversation history."""
    if not conversation_memory:
        return {"error": "Conversation memory not available"}
    
    sessions = conversation_memory.get_all_sessions()
    return {
        "sessions": [
            {
                "session_id": session.session_id,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "turns": len(session.turns),
                "summary": session.session_summary
            }
            for session in sessions
        ]
    }


@app.get("/api/statistics")
async def get_statistics():
    """Get conversation statistics."""
    if not conversation_memory:
        return {"error": "Conversation memory not available"}
    
    stats = conversation_memory.get_session_statistics()
    return stats


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time communication."""
    await manager.connect(websocket, client_id)
    
    # Start new conversation session
    if conversation_memory:
        session_id = conversation_memory.start_new_session(f"web_{client_id}")
        await manager.send_message(client_id, {
            "type": "session_started",
            "session_id": session_id,
            "memory": conversation_memory.consent_status(),
            "resume": conversation_memory.last_persisted_preview(),
        })
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            await handle_websocket_message(client_id, message)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        
        # End conversation session
        if conversation_memory:
            conversation_memory.end_current_session()


async def handle_websocket_message(client_id: str, message: dict):
    """Handle incoming WebSocket message."""
    message_type = message.get("type")
    
    if message_type == "text_input":
        await handle_text_input(client_id, message)
    elif message_type == "audio_input":
        await handle_audio_input(client_id, message)
    elif message_type == "get_emotion_patterns":
        await handle_get_emotion_patterns(client_id)
    elif message_type == "emotion_feedback":
        await handle_emotion_feedback(client_id, message)
    elif message_type == "response_feedback":
        await handle_response_feedback(client_id, message)
    elif message_type == "memory_consent":
        await handle_memory_consent(client_id, message)
    elif message_type == "resume_session":
        await handle_resume_session(client_id, message)
    elif message_type == "clinician_share":
        await handle_clinician_share(client_id, message)
    elif message_type == "get_recap":
        await handle_get_recap(client_id)
    elif message_type == "get_clinician_trends":
        await handle_get_clinician_trends(client_id)
    elif message_type == "memory_wipe":
        await handle_memory_wipe(client_id)
    else:
        await manager.send_message(client_id, {
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        })


async def handle_text_input(client_id: str, message: dict):
    """Handle text input from user."""
    try:
        user_text = message.get("text", "").strip()
        if not user_text:
            return
        trace_id = new_trace_id()
        log_event(logger, "web_text_received", trace_id, client_id=client_id, text=user_text)

        rm = message.get("response_mode")
        if rm in {"listen", "coach"} and conversation_memory:
            conversation_memory.set_response_mode_preference(rm)

        # Send processing status
        await manager.send_message(client_id, {
            "type": "processing",
            "stage": "emotion_detection"
        })

        # For text input, we'll use a simple emotion detection based on keywords
        emotion_result = detect_emotion_from_text(user_text)
        if emotion_fusion:
            audio_pred = EmotionPrediction(
                emotion=emotion_result["emotion"],
                confidence=emotion_result["confidence"],
                probabilities={emotion_result["emotion"]: emotion_result["confidence"]},
                features_used=0
            )
            boost = conversation_memory.get_fusion_text_boost() if conversation_memory else 0.0
            fused = emotion_fusion.fuse(
                audio_pred, user_text, 1.0, text_boost=boost
            )
            emotion_result = {"emotion": fused.emotion, "confidence": fused.confidence}

        # Save to conversation memory first to extract name
        if conversation_memory:
            conversation_memory.add_conversation_turn(
                user_text=user_text,
                user_emotion=emotion_result["emotion"],
                user_confidence=emotion_result["confidence"],
                assistant_response="",  # Will be updated after generation
                empathy_style=config.response_generation.default_style,
                response_confidence=0.0  # Will be updated after generation
            )

        # Generate empathic response
        await manager.send_message(client_id, {
            "type": "processing",
            "stage": "response_generation"
        })

        # Get user name after potential extraction
        user_name = conversation_memory.get_user_name() if conversation_memory else None
        user_preferences = conversation_memory.get_user_preferences() if conversation_memory else {}

        # Add user name to preferences if known
        if user_name:
            user_preferences['user_name'] = user_name
        user_preferences['session_id'] = client_id
        region = (message.get("region") or "").strip().upper()
        if region:
            user_preferences["region"] = region
            if conversation_memory:
                conversation_memory.update_user_preferences({"region": region})
        clinic_profile = (message.get("clinic_profile") or "").strip()
        if clinic_profile:
            user_preferences["clinic_profile"] = clinic_profile
            if conversation_memory:
                conversation_memory.update_user_preferences({"clinic_profile": clinic_profile})

        response_context = ResponseContext(
            user_text=user_text,
            emotion=emotion_result["emotion"],
            confidence=emotion_result["confidence"],
            conversation_history=conversation_memory.get_conversation_context() if conversation_memory else [],
            empathy_style=config.response_generation.default_style,
            user_preferences=user_preferences
        )

        empathic_response = response_generator.generate_response(response_context)
        log_event(logger, "web_response", trace_id, response=empathic_response.text[:200])

        # Update the last conversation turn with the response
        if conversation_memory and conversation_memory.current_session and conversation_memory.current_session.turns:
            last_turn = conversation_memory.current_session.turns[-1]
            last_turn.assistant_response = empathic_response.text
            last_turn.empathy_style = empathic_response.empathy_style
            last_turn.response_confidence = empathic_response.confidence
            conversation_memory.record_safety_flag(empathic_response.safety_tier)
            if empathic_response.grounding:
                conversation_memory.record_tool("grounding")

        # Send response
        await manager.send_message(client_id, {
            "type": "response",
            "user_text": user_text,
            "emotion": emotion_result["emotion"],
            "emotion_confidence": emotion_result["confidence"],
            "response_text": empathic_response.text,
            "empathy_style": empathic_response.empathy_style,
            "response_confidence": empathic_response.confidence,
            "user_name": user_name,
            "interaction_mode": empathic_response.interaction_mode,
            "fusion_text_boost": conversation_memory.get_fusion_text_boost() if conversation_memory else 0.0,
            "speech": speech_settings_for(
                emotion_result["emotion"],
                "grounding" if empathic_response.grounding else empathic_response.interaction_mode,
            ),
            "grounding": empathic_response.grounding,
            "safety_tier": empathic_response.safety_tier,
            "crisis_resources": empathic_response.crisis_resources,
            "memory": conversation_memory.consent_status() if conversation_memory else None,
            "recap": conversation_memory.shareable_recap() if conversation_memory else "",
            "clinic_profile": empathic_response.clinic_profile,
            "care_mode": empathic_response.care_mode,
            "llm_online": empathic_response.llm_online,
        })
        
    except Exception as e:
        logger.error(f"Error handling text input: {e}")
        await manager.send_message(client_id, {
            "type": "error",
            "message": f"Error processing text: {str(e)}"
        })


async def handle_audio_input(client_id: str, message: dict):
    """Handle audio input from user."""
    try:
        await manager.send_message(client_id, {
            "type": "processing",
            "stage": "audio_processing"
        })

        user_text = (message.get("transcript") or "").strip()
        b64 = message.get("audio_b64") or message.get("audio")
        if (not user_text) and b64 and speech_recognizer:
            wav = base64.b64decode(b64)
            audio = wav_bytes_to_float(wav, target_rate=16000)
            result = speech_recognizer.transcribe(audio)
            user_text = (result.text or "").strip()
        if not user_text:
            await manager.send_message(client_id, {
                "type": "error",
                "message": "I couldn't catch that. Try speaking a little closer, or type instead.",
            })
            return

        extra = {k: message.get(k) for k in ("response_mode", "region") if message.get(k)}
        extra["text"] = user_text
        await handle_text_input(client_id, extra)

    except Exception as e:
        logger.error(f"Error handling audio input: {e}")
        await manager.send_message(client_id, {
            "type": "error",
            "message": f"Error processing audio: {str(e)}"
        })


async def handle_emotion_feedback(client_id: str, message: dict):
    """Calibration: user confirms or corrects detected emotion."""
    if not conversation_memory:
        await manager.send_message(client_id, {
            "type": "error",
            "message": "Conversation memory not available"
        })
        return
    match = message.get("match", True)
    felt = message.get("felt_emotion")
    conversation_memory.record_emotion_feedback(bool(match), felt)
    await manager.send_message(client_id, {
        "type": "emotion_feedback_ack",
        "fusion_text_boost": conversation_memory.get_fusion_text_boost(),
    })


async def handle_response_feedback(client_id: str, message: dict):
    """Calibration: whether the last reply felt helpful."""
    if not conversation_memory:
        await manager.send_message(client_id, {
            "type": "error",
            "message": "Conversation memory not available"
        })
        return
    helpful = message.get("helpful", True)
    conversation_memory.record_response_feedback(bool(helpful))
    await manager.send_message(client_id, {
        "type": "response_feedback_ack",
        "fusion_text_boost": conversation_memory.get_fusion_text_boost(),
    })


async def handle_memory_wipe(client_id: str):
    """Erase encrypted saved visits and start a clean session."""
    if not conversation_memory:
        await manager.send_message(client_id, {
            "type": "error",
            "message": "Conversation memory not available",
        })
        return
    conversation_memory.clear_all_conversations()
    session_id = conversation_memory.start_new_session(f"web_{client_id}")
    await manager.send_message(client_id, {
        "type": "memory_wiped",
        "session_id": session_id,
        "memory": conversation_memory.consent_status(),
        "recap": "",
    })


async def handle_memory_consent(client_id: str, message: dict):
    """Remember or discard the current session."""
    if not conversation_memory:
        await manager.send_message(client_id, {
            "type": "error",
            "message": "Conversation memory not available",
        })
        return
    persist = bool(message.get("persist"))
    status = conversation_memory.set_persist_consent(persist)
    if persist:
        status["recap"] = conversation_memory.shareable_recap()
    await manager.send_message(client_id, {
        "type": "memory_consent_ack",
        **status,
    })


async def handle_resume_session(client_id: str, message: dict):
    if not conversation_memory:
        return
    if message.get("fresh"):
        conversation_memory.current_session.user_preferences.pop("resume_recap", None) if conversation_memory.current_session else None
        await manager.send_message(client_id, {"type": "resume_ack", "resumed": False, "fresh": True})
        return
    result = conversation_memory.resume_last_session()
    await manager.send_message(client_id, {"type": "resume_ack", **result})


async def handle_clinician_share(client_id: str, message: dict):
    if not conversation_memory:
        return
    status = conversation_memory.set_clinician_share(bool(message.get("enabled")))
    await manager.send_message(client_id, {"type": "clinician_share_ack", **status})


async def handle_get_recap(client_id: str):
    if not conversation_memory:
        await manager.send_message(client_id, {"type": "recap", "recap": ""})
        return
    await manager.send_message(client_id, {
        "type": "recap",
        "recap": conversation_memory.shareable_recap(),
    })


async def handle_get_clinician_trends(client_id: str):
    if not conversation_memory:
        await manager.send_message(client_id, {"type": "clinician_trends", "enabled": False, "days": []})
        return
    await manager.send_message(client_id, {
        "type": "clinician_trends",
        **conversation_memory.clinician_trend_view(),
    })


async def handle_get_emotion_patterns(client_id: str):
    """Send emotion patterns to client."""
    if not conversation_memory:
        await manager.send_message(client_id, {
            "type": "emotion_patterns",
            "patterns": {}
        })
        return
    
    patterns = conversation_memory.get_emotion_patterns()
    await manager.send_message(client_id, {
        "type": "emotion_patterns",
        "patterns": patterns
    })


def detect_emotion_from_text(text: str) -> dict:
    """Simple text-based emotion detection for web interface."""
    text_lower = text.lower()
    
    # Simple keyword-based emotion detection
    emotion_keywords = {
        'happy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'love'],
        'sad': ['sad', 'depressed', 'down', 'upset', 'crying', 'hurt', 'disappointed'],
        'angry': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'hate'],
        'anxious': ['anxious', 'worried', 'nervous', 'scared', 'afraid', 'stress'],
        'calm': ['calm', 'peaceful', 'relaxed', 'serene', 'tranquil']
    }
    
    emotion_scores = {}
    
    for emotion, keywords in emotion_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score
    
    if emotion_scores:
        detected_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = min(0.8, emotion_scores[detected_emotion] * 0.3)  # Simple confidence calculation
    else:
        detected_emotion = 'neutral'
        confidence = 0.5
    
    return {
        "emotion": detected_emotion,
        "confidence": confidence
    }


# Create web directory structure
def create_web_files():
    """Create web interface files."""
    web_dir = Path("web")
    templates_dir = web_dir / "templates"
    static_dir = web_dir / "static"
    
    # Create directories
    templates_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = templates_dir / "index.html"
    if index_path.exists():
        logger.info("Keeping existing %s", index_path)
        return

    # Create index.html (bootstrap only if missing — edit web/templates/index.html in repo)
    index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tunda - Your Empathic Voice Companion</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
            height: 100vh;
        }

        .main-panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            display: flex;
            flex-direction: column;
        }

        .side-panel {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            position: relative;
        }

        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        .header p {
            color: #666;
            font-size: 1.1em;
        }

        .tunda-avatar {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: linear-gradient(45deg, #667eea, #764ba2);
            margin: 0 auto 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }

        .tunda-avatar:hover {
            transform: scale(1.05);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        }

        .tunda-avatar.listening {
            animation: voicePulse 2s infinite;
        }

        .tunda-avatar.listening .voice-wave-bar {
            animation: voiceWaveActive 0.8s ease-in-out infinite;
        }

        .tunda-avatar.speaking {
            animation: voiceSpeaking 1s infinite;
        }

        .tunda-avatar.speaking .voice-wave-bar {
            animation: voiceWaveSpeaking 0.6s ease-in-out infinite;
        }

        .tunda-avatar.processing {
            animation: voiceProcessing 3s infinite;
        }

        .tunda-avatar.processing .voice-wave-bar {
            animation: voiceWaveProcessing 1.2s ease-in-out infinite;
        }

        .voice-node {
            position: relative;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .voice-core {
            width: 60px;
            height: 60px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            position: relative;
            z-index: 3;
            display: flex;
            align-items: center;
            justify-content: center;
            backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 255, 255, 0.3);
        }

        .voice-waves {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 2px;
            width: 100%;
            height: 100%;
        }

        .voice-wave-bar {
            width: 3px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 2px;
            animation: voiceWave 1.5s ease-in-out infinite;
        }

        .voice-wave-bar:nth-child(1) {
            height: 8px;
            animation-delay: 0s;
        }

        .voice-wave-bar:nth-child(2) {
            height: 16px;
            animation-delay: 0.1s;
        }

        .voice-wave-bar:nth-child(3) {
            height: 24px;
            animation-delay: 0.2s;
        }

        .voice-wave-bar:nth-child(4) {
            height: 20px;
            animation-delay: 0.3s;
        }

        .voice-wave-bar:nth-child(5) {
            height: 12px;
            animation-delay: 0.4s;
        }

        @keyframes voiceWave {
            0%, 100% {
                transform: scaleY(0.5);
                opacity: 0.7;
            }
            50% {
                transform: scaleY(1.2);
                opacity: 1;
            }
        }

        @keyframes voiceWaveActive {
            0%, 100% {
                transform: scaleY(0.3);
                opacity: 0.8;
            }
            50% {
                transform: scaleY(1.8);
                opacity: 1;
            }
        }

        @keyframes voiceWaveSpeaking {
            0%, 100% {
                transform: scaleY(0.4);
                opacity: 0.9;
            }
            25% {
                transform: scaleY(1.6);
                opacity: 1;
            }
            75% {
                transform: scaleY(1.2);
                opacity: 0.95;
            }
        }

        @keyframes voiceWaveProcessing {
            0% {
                transform: scaleY(0.2);
                opacity: 0.6;
            }
            33% {
                transform: scaleY(1.4);
                opacity: 0.8;
            }
            66% {
                transform: scaleY(0.8);
                opacity: 1;
            }
            100% {
                transform: scaleY(0.2);
                opacity: 0.6;
            }
        }

        .voice-ring {
            position: absolute;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            animation: voiceRing 3s linear infinite;
        }

        .voice-ring:nth-child(1) {
            width: 60px;
            height: 60px;
            animation-delay: 0s;
        }

        .voice-ring:nth-child(2) {
            width: 80px;
            height: 80px;
            animation-delay: 0.5s;
        }

        .voice-ring:nth-child(3) {
            width: 100px;
            height: 100px;
            animation-delay: 1s;
        }

        .voice-particles {
            position: absolute;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }

        .voice-particle {
            position: absolute;
            width: 4px;
            height: 4px;
            background: rgba(255, 255, 255, 0.6);
            border-radius: 50%;
            animation: voiceParticle 4s linear infinite;
        }

        .voice-particle:nth-child(1) {
            top: 20%;
            left: 50%;
            animation-delay: 0s;
        }

        .voice-particle:nth-child(2) {
            top: 50%;
            left: 80%;
            animation-delay: 1s;
        }

        .voice-particle:nth-child(3) {
            top: 80%;
            left: 50%;
            animation-delay: 2s;
        }

        .voice-particle:nth-child(4) {
            top: 50%;
            left: 20%;
            animation-delay: 3s;
        }

        @keyframes voicePulse {
            0% {
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
                transform: scale(1);
            }
            50% {
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6), 0 0 0 15px rgba(102, 126, 234, 0.1), 0 0 0 30px rgba(102, 126, 234, 0.05);
                transform: scale(1.02);
            }
            100% {
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
                transform: scale(1);
            }
        }

        @keyframes voiceRing {
            0% {
                transform: scale(0.8);
                opacity: 1;
            }
            100% {
                transform: scale(1.2);
                opacity: 0;
            }
        }

        @keyframes voiceParticle {
            0% {
                transform: scale(0) rotate(0deg);
                opacity: 1;
            }
            50% {
                transform: scale(1) rotate(180deg);
                opacity: 0.8;
            }
            100% {
                transform: scale(0) rotate(360deg);
                opacity: 0;
            }
        }

        @keyframes voiceSpeaking {
            0% {
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
                background: linear-gradient(45deg, #667eea, #764ba2);
            }
            25% {
                box-shadow: 0 10px 30px rgba(118, 75, 162, 0.5);
                background: linear-gradient(45deg, #764ba2, #667eea);
            }
            50% {
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.6);
                background: linear-gradient(45deg, #667eea, #764ba2);
            }
            75% {
                box-shadow: 0 10px 30px rgba(118, 75, 162, 0.5);
                background: linear-gradient(45deg, #764ba2, #667eea);
            }
            100% {
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
                background: linear-gradient(45deg, #667eea, #764ba2);
            }
        }

        @keyframes voiceProcessing {
            0% {
                transform: rotate(0deg);
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            }
            50% {
                transform: rotate(180deg);
                box-shadow: 0 10px 30px rgba(155, 89, 182, 0.5);
            }
            100% {
                transform: rotate(360deg);
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            }
        }

        .chat-container {
            flex: 1;
            border: none;
            padding: 20px;
            overflow-y: auto;
            margin-bottom: 20px;
            background: rgba(248, 250, 252, 0.8);
            border-radius: 15px;
            max-height: 400px;
        }

        .message {
            margin: 15px 0;
            padding: 15px 20px;
            border-radius: 20px;
            max-width: 80%;
            position: relative;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .user-message {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 5px;
        }

        .assistant-message {
            background: white;
            border: 2px solid #e2e8f0;
            margin-right: auto;
            border-bottom-left-radius: 5px;
        }

        .message-header {
            font-weight: bold;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .emotion-info {
            font-size: 0.85em;
            opacity: 0.8;
            margin-top: 8px;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .emotion-badge {
            background: rgba(255, 255, 255, 0.2);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
        }

        .input-container {
            display: flex;
            gap: 15px;
            align-items: center;
            background: white;
            padding: 15px;
            border-radius: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }

        .text-input {
            flex: 1;
            padding: 12px 20px;
            border: none;
            border-radius: 20px;
            background: #f8fafc;
            font-size: 1em;
            outline: none;
        }

        .text-input:focus {
            background: #e2e8f0;
        }

        .voice-button, .send-button {
            width: 50px;
            height: 50px;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2em;
            transition: all 0.3s ease;
        }

        .voice-button {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
        }

        .voice-button:hover {
            transform: scale(1.1);
        }

        .voice-button.recording {
            animation: recordPulse 1s infinite;
        }

        @keyframes recordPulse {
            0% { background: linear-gradient(45deg, #ff6b6b, #ee5a24); }
            50% { background: linear-gradient(45deg, #ff4757, #c44569); }
            100% { background: linear-gradient(45deg, #ff6b6b, #ee5a24); }
        }

        .send-button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
        }

        .send-button:hover {
            transform: scale(1.1);
        }

        .status {
            text-align: center;
            margin: 15px 0;
            padding: 10px;
            border-radius: 10px;
            background: rgba(102, 126, 234, 0.1);
            color: #667eea;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .status.success {
            background: rgba(39, 174, 96, 0.1);
            color: #27ae60;
        }

        .status.error {
            background: rgba(231, 76, 60, 0.1);
            color: #e74c3c;
        }

        .status.processing {
            background: rgba(155, 89, 182, 0.1);
            color: #9b59b6;
            animation: processingPulse 2s infinite;
        }

        .status.recording {
            background: rgba(255, 107, 107, 0.1);
            color: #ff6b6b;
            animation: recordingPulse 1s infinite;
        }

        .status.greeting {
            background: rgba(241, 196, 15, 0.1);
            color: #f1c40f;
        }

        @keyframes processingPulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        @keyframes recordingPulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.8; transform: scale(1.02); }
        }

        .emotion-patterns {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .emotion-patterns h3 {
            color: #667eea;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .pattern-item {
            margin: 10px 0;
            padding: 10px;
            background: #f8fafc;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .care-plan-section {
            margin-top: 20px;
            padding: 20px;
            background: linear-gradient(45deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            border-radius: 15px;
        }

        .care-plan-section h3 {
            color: #667eea;
            margin-bottom: 15px;
        }

        .activity-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }

        .activity-card h4 {
            color: #333;
            margin-bottom: 8px;
        }

        .activity-meta {
            font-size: 0.9em;
            color: #666;
            margin-top: 8px;
        }

        .waveform {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 2px;
            height: 30px;
            margin: 10px 0;
        }

        .waveform-bar {
            width: 3px;
            background: #667eea;
            border-radius: 2px;
            animation: waveform 1.5s ease-in-out infinite;
        }

        .waveform-bar:nth-child(2) { animation-delay: 0.1s; }
        .waveform-bar:nth-child(3) { animation-delay: 0.2s; }
        .waveform-bar:nth-child(4) { animation-delay: 0.3s; }
        .waveform-bar:nth-child(5) { animation-delay: 0.4s; }

        @keyframes waveform {
            0%, 100% { height: 5px; }
            50% { height: 25px; }
        }

        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
                padding: 10px;
            }

            .header h1 {
                font-size: 2em;
            }

            .tunda-avatar {
                width: 80px;
                height: 80px;
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-panel">
            <div class="header">
                <div class="tunda-avatar" id="tunda-avatar">
                    <div class="voice-node">
                        <div class="voice-ring"></div>
                        <div class="voice-ring"></div>
                        <div class="voice-ring"></div>
                        <div class="voice-core">
                            <div class="voice-waves">
                                <div class="voice-wave-bar"></div>
                                <div class="voice-wave-bar"></div>
                                <div class="voice-wave-bar"></div>
                                <div class="voice-wave-bar"></div>
                                <div class="voice-wave-bar"></div>
                            </div>
                        </div>
                        <div class="voice-particles">
                            <div class="voice-particle"></div>
                            <div class="voice-particle"></div>
                            <div class="voice-particle"></div>
                            <div class="voice-particle"></div>
                        </div>
                    </div>
                </div>
                <h1>Tunda</h1>
                <p>Your Empathic Voice Companion</p>
            </div>

            <div id="chat-container" class="chat-container"></div>

            <div class="status" id="status">
                <i class="fas fa-heart"></i> Ready to listen and support you...
            </div>

            <div class="input-container">
                <button id="voice-button" class="voice-button" title="Click to speak">
                    <i class="fas fa-microphone"></i>
                </button>
                <input type="text" id="text-input" class="text-input" placeholder="Type your message or click the microphone to speak..." />
                <button id="send-button" class="send-button" title="Send message">
                    <i class="fas fa-paper-plane"></i>
                </button>
            </div>
        </div>

        <div class="side-panel">
            <div id="emotion-patterns" class="emotion-patterns" style="display: none;">
                <h3><i class="fas fa-chart-line"></i> Emotion Patterns</h3>
                <div id="patterns-content"></div>
            </div>

            <div id="care-plan-section" class="care-plan-section" style="display: none;">
                <h3><i class="fas fa-heart-pulse"></i> Personalized Care</h3>
                <div id="care-plan-content"></div>
            </div>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const textInput = document.getElementById('text-input');
        const sendButton = document.getElementById('send-button');
        const voiceButton = document.getElementById('voice-button');
        const status = document.getElementById('status');
        const emotionPatterns = document.getElementById('emotion-patterns');
        const patternsContent = document.getElementById('patterns-content');
        const carePlanSection = document.getElementById('care-plan-section');
        const carePlanContent = document.getElementById('care-plan-content');
        const tundaAvatar = document.getElementById('tunda-avatar');

        const clientId = 'web_' + Math.random().toString(36).substr(2, 9);
        const ws = new WebSocket(`ws://localhost:8000/ws/${clientId}`);

        let isRecording = false;
        let mediaRecorder = null;
        let audioChunks = [];
        let recognition = null;

        // Initialize speech recognition if available
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
        }

        ws.onopen = function(event) {
            updateStatus('<i class="fas fa-heart"></i> Connected to Tunda - Ready to listen!', 'success');
            tundaAvatar.style.animation = 'pulse 2s ease-in-out infinite';
        };

        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            handleWebSocketMessage(message);
        };

        ws.onclose = function(event) {
            updateStatus('<i class="fas fa-exclamation-triangle"></i> Disconnected from server', 'error');
            tundaAvatar.style.animation = 'none';
        };
        
        function handleWebSocketMessage(message) {
            switch(message.type) {
                case 'session_started':
                    updateStatus(`<i class="fas fa-play"></i> Session started with Tunda`, 'success');
                    break;
                case 'processing':
                    const stage = message.stage.replace('_', ' ');
                    updateStatus(`<i class="fas fa-brain"></i> ${stage}...`, 'processing');
                    tundaAvatar.classList.remove('listening', 'speaking');
                    tundaAvatar.classList.add('processing');
                    break;
                case 'response':
                    addMessage(message.user_text, 'user');
                    addMessage(message.response_text, 'assistant', message.emotion, message.emotion_confidence);

                    // Update status with personalized message if name is known
                    if (message.user_name) {
                        updateStatus(`<i class="fas fa-heart"></i> Ready to listen and support you, ${message.user_name}...`, 'ready');
                    } else {
                        updateStatus('<i class="fas fa-heart"></i> Ready to listen and support you...', 'ready');
                    }

                    tundaAvatar.classList.remove('listening', 'processing');
                    tundaAvatar.classList.add('speaking');
                    requestEmotionPatterns();
                    speakResponse(message.response_text);
                    break;
                case 'emotion_patterns':
                    updateEmotionPatterns(message.patterns);
                    break;
                case 'error':
                    updateStatus(`<i class="fas fa-exclamation-circle"></i> ${message.message}`, 'error');
                    tundaAvatar.classList.remove('listening', 'processing', 'speaking');
                    break;
            }
        }

        function updateStatus(text, type = 'default') {
            status.innerHTML = text;
            status.className = 'status';
            if (type !== 'default') {
                status.classList.add(type);
            }
        }

        function speakResponse(text) {
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 0.9;
                utterance.pitch = 1.1;
                utterance.volume = 0.8;

                // Try to use a female voice
                const voices = speechSynthesis.getVoices();
                const femaleVoice = voices.find(voice =>
                    voice.name.toLowerCase().includes('female') ||
                    voice.name.toLowerCase().includes('zira') ||
                    voice.name.toLowerCase().includes('hazel')
                );
                if (femaleVoice) {
                    utterance.voice = femaleVoice;
                }

                // Remove speaking animation when done
                utterance.onend = function() {
                    tundaAvatar.classList.remove('speaking');
                };

                speechSynthesis.speak(utterance);
            } else {
                // If no speech synthesis, remove speaking animation after a delay
                setTimeout(() => {
                    tundaAvatar.classList.remove('speaking');
                }, 3000);
            }
        }
        
        function addMessage(text, sender, emotion = null, confidence = null) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;

            let content = '';
            if (sender === 'user') {
                content = `
                    <div class="message-header">
                        <i class="fas fa-user"></i> You
                    </div>
                    <div>${text}</div>
                `;
            } else {
                content = `
                    <div class="message-header">
                        <i class="fas fa-microphone-alt"></i> Tunda
                    </div>
                    <div>${text}</div>
                `;
                if (emotion) {
                    const emotionIcon = getEmotionIcon(emotion);
                    content += `
                        <div class="emotion-info">
                            ${emotionIcon} <span class="emotion-badge">${emotion}</span>
                            <span>${(confidence * 100).toFixed(0)}% confidence</span>
                        </div>
                    `;
                }
            }

            messageDiv.innerHTML = content;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            // Add typing animation effect
            if (sender === 'assistant') {
                messageDiv.style.opacity = '0';
                messageDiv.style.transform = 'translateY(20px)';
                setTimeout(() => {
                    messageDiv.style.transition = 'all 0.3s ease';
                    messageDiv.style.opacity = '1';
                    messageDiv.style.transform = 'translateY(0)';
                }, 100);
            }
        }

        function getEmotionIcon(emotion) {
            const icons = {
                'happy': '<i class="fas fa-smile" style="color: #f39c12;"></i>',
                'sad': '<i class="fas fa-frown" style="color: #3498db;"></i>',
                'angry': '<i class="fas fa-angry" style="color: #e74c3c;"></i>',
                'anxious': '<i class="fas fa-dizzy" style="color: #9b59b6;"></i>',
                'calm': '<i class="fas fa-leaf" style="color: #27ae60;"></i>',
                'neutral': '<i class="fas fa-meh" style="color: #95a5a6;"></i>'
            };
            return icons[emotion] || '<i class="fas fa-circle" style="color: #bdc3c7;"></i>';
        }
        
        function sendMessage() {
            const text = textInput.value.trim();
            if (!text) return;

            ws.send(JSON.stringify({
                type: 'text_input',
                text: text
            }));

            textInput.value = '';
        }

        function startVoiceRecording() {
            if (!recognition) {
                updateStatus('<i class="fas fa-exclamation-triangle"></i> Speech recognition not supported in this browser', 'error');
                return;
            }

            if (isRecording) {
                stopVoiceRecording();
                return;
            }

            isRecording = true;
            voiceButton.classList.add('recording');
            voiceButton.innerHTML = '<i class="fas fa-stop"></i>';
            updateStatus('<i class="fas fa-microphone"></i> Listening... Speak now!', 'recording');
            tundaAvatar.classList.add('listening');

            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                textInput.value = transcript;
                sendMessage();
            };

            recognition.onerror = function(event) {
                updateStatus('<i class="fas fa-exclamation-circle"></i> Speech recognition error', 'error');
                stopVoiceRecording();
            };

            recognition.onend = function() {
                stopVoiceRecording();
            };

            recognition.start();
        }

        function stopVoiceRecording() {
            if (!isRecording) return;

            isRecording = false;
            voiceButton.classList.remove('recording');
            voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
            updateStatus('<i class="fas fa-heart"></i> Ready to listen and support you...', 'ready');
            tundaAvatar.classList.remove('listening');

            if (recognition) {
                recognition.stop();
            }
        }

        function requestEmotionPatterns() {
            ws.send(JSON.stringify({
                type: 'get_emotion_patterns'
            }));
        }

        function updateEmotionPatterns(patterns) {
            if (Object.keys(patterns).length === 0) {
                emotionPatterns.style.display = 'none';
                return;
            }

            let content = '';

            if (patterns.most_common_emotion) {
                const icon = getEmotionIcon(patterns.most_common_emotion);
                content += `
                    <div class="pattern-item">
                        <strong>${icon} Most Common Emotion:</strong>
                        <span style="text-transform: capitalize;">${patterns.most_common_emotion}</span>
                    </div>
                `;
            }

            if (patterns.recent_trend) {
                const trendIcon = patterns.recent_trend === 'improving' ?
                    '<i class="fas fa-arrow-up" style="color: #27ae60;"></i>' :
                    patterns.recent_trend === 'declining' ?
                    '<i class="fas fa-arrow-down" style="color: #e74c3c;"></i>' :
                    '<i class="fas fa-minus" style="color: #f39c12;"></i>';

                content += `
                    <div class="pattern-item">
                        <strong>${trendIcon} Recent Trend:</strong>
                        <span style="text-transform: capitalize;">${patterns.recent_trend}</span>
                    </div>
                `;
            }

            if (patterns.total_interactions) {
                content += `
                    <div class="pattern-item">
                        <strong><i class="fas fa-comments"></i> Total Interactions:</strong>
                        ${patterns.total_interactions}
                    </div>
                `;
            }

            patternsContent.innerHTML = content;
            emotionPatterns.style.display = 'block';

            // Show care plan section
            showCarePlan(patterns.most_common_emotion);
        }

        function showCarePlan(emotion) {
            if (!emotion) return;

            const carePlans = {
                'happy': {
                    title: 'Joy Amplification',
                    activities: ['Gratitude practice', 'Share positivity', 'Creative expression'],
                    color: '#f39c12'
                },
                'sad': {
                    title: 'Gentle Healing',
                    activities: ['Comfort breathing', 'Self-compassion', 'Gentle movement'],
                    color: '#3498db'
                },
                'anxious': {
                    title: 'Calm & Ground',
                    activities: ['5-4-3-2-1 grounding', 'Box breathing', 'Progressive relaxation'],
                    color: '#9b59b6'
                },
                'angry': {
                    title: 'Healthy Release',
                    activities: ['Cooling breath', 'Physical release', 'Anger acknowledgment'],
                    color: '#e74c3c'
                },
                'calm': {
                    title: 'Peaceful Maintenance',
                    activities: ['Mindful appreciation', 'Gentle reflection', 'Peace practice'],
                    color: '#27ae60'
                }
            };

            const plan = carePlans[emotion];
            if (!plan) return;

            let content = `
                <div class="activity-card">
                    <h4 style="color: ${plan.color};">${plan.title} Plan</h4>
                    <p>Personalized activities for your current emotional state:</p>
                    <ul style="margin: 10px 0; padding-left: 20px;">
            `;

            plan.activities.forEach(activity => {
                content += `<li style="margin: 5px 0;">${activity}</li>`;
            });

            content += `
                    </ul>
                    <div class="activity-meta">
                        <i class="fas fa-clock"></i> Take a few minutes for self-care
                    </div>
                </div>
            `;

            carePlanContent.innerHTML = content;
            carePlanSection.style.display = 'block';
        }
        
        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        voiceButton.addEventListener('click', startVoiceRecording);

        textInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        textInput.addEventListener('focus', function() {
            this.style.background = '#e2e8f0';
        });

        textInput.addEventListener('blur', function() {
            this.style.background = '#f8fafc';
        });

        // Tunda avatar click interaction
        tundaAvatar.addEventListener('click', function() {
            this.style.transform = 'scale(0.95)';
            setTimeout(() => {
                this.style.transform = 'scale(1)';
            }, 150);

            // Show a friendly message
            const greetings = [
                "Hello! I'm Tunda, ready to listen! 💙",
                "Hi there! How are you feeling today? 🤗",
                "I'm here to support you! What's on your mind? 💭",
                "Ready to chat! Tell me about your day! ✨"
            ];

            const randomGreeting = greetings[Math.floor(Math.random() * greetings.length)];
            updateStatus(`<i class="fas fa-heart"></i> ${randomGreeting}`, 'greeting');
        });

        // Initialize speech synthesis voices
        if ('speechSynthesis' in window) {
            speechSynthesis.onvoiceschanged = function() {
                // Voices are loaded
            };
        }

        // Request initial emotion patterns
        setTimeout(requestEmotionPatterns, 1000);

        // Add some initial welcome message
        setTimeout(() => {
            addMessage("Hello! I'm Tunda, your empathic voice companion. I'm here to listen, understand, and support you. You can type a message or click the microphone button to speak with me. Please introduce yourself by saying 'Hello, my name is [Your Name]' and I'll give you a special greeting! How are you feeling today?", 'assistant');
        }, 2000);
    </script>
</body>
</html>'''
    
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)


def main():
    """Run the web application."""
    # Create web files
    create_web_files()
    
    # Run the server
    import os

    # Use Render's environment variables if available
    host = os.getenv('HOST', config.get('web.host', '0.0.0.0'))
    port = int(os.getenv('PORT', config.get('web.port', 8000)))

    print(f"🌐 Starting Empathic Voice Companion web interface...")
    print(f"📍 Server running on: http://{host}:{port}")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

"""Empathic response generation using LLM integration."""

import requests
import json
import logging
from typing import Dict, List, Optional, Any, Iterator
from dataclasses import dataclass
import random
import time

from ..utils.config import config
from ..utils.performance import latency_manager
from ..core.plugins import get_llm
from .safety import SafetyGuard
from .mode_router import InteractionModeRouter, mode_instructions
from .empathy import EmpathyPatterns
from .care_plans import CarePlanGenerator
from .grounding import select_grounding
from ..clinic.profiles import (
    OFFLINE_CARE_MESSAGE,
    ClinicProfile,
    get_clinic_profile,
)
from ..emotion.detector import EmotionPrediction

logger = logging.getLogger(__name__)


@dataclass
class ResponseContext:
    """Context for generating empathic responses."""
    user_text: str
    emotion: str
    confidence: float
    conversation_history: List[Dict[str, str]]
    empathy_style: str
    user_preferences: Dict[str, Any]
    relevant_memories: List[Dict[str, Any]] = None
    interaction_mode: str = "listen"

    def __post_init__(self):
        if self.relevant_memories is None:
            self.relevant_memories = []


@dataclass
class EmpathicResponse:
    """Generated empathic response."""
    text: str
    emotion_addressed: str
    empathy_style: str
    confidence: float
    generation_time: float
    follow_up_suggested: bool
    interaction_mode: str = "listen"
    safety_tier: str = "none"
    crisis_resources: Optional[Dict[str, str]] = None
    grounding: Optional[Dict[str, Any]] = None
    clinic_profile: str = "companion"
    care_mode: bool = False
    llm_online: bool = False


class LLMProvider:
    """Base class for LLM providers."""
    
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def generate_response_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate streaming response."""
        yield self.generate_response(prompt, **kwargs)


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local inference."""
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Ollama."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 150),
                    "top_p": kwargs.get("top_p", 0.9),
                    "repeat_penalty": kwargs.get("repeat_penalty", 1.1)
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama connection error: {e}")
            return ""
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return ""
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def generate_response_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate streaming response using Ollama."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,  # Enable streaming
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 150),
                    "top_p": kwargs.get("top_p", 0.9),
                    "repeat_penalty": kwargs.get("repeat_penalty", 1.1)
                }
            }
            
            with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=30
            ) as response:
                if response.status_code != 200:
                    logger.error(f"Ollama API error: {response.status_code}")
                    return

                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            token = json_response.get("response", "")
                            if token:
                                yield token
                            if json_response.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            yield ""


class HuggingFaceProvider(LLMProvider):
    """Hugging Face transformers provider for local inference."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Hugging Face model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Loaded Hugging Face model: {self.model_name}")
            
        except ImportError:
            logger.error("Transformers library not available")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model: {e}")
            self.model = None
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Hugging Face model."""
        if self.model is None or self.tokenizer is None:
            return ""
        
        try:
            import torch
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + kwargs.get("max_tokens", 50),
                    temperature=kwargs.get("temperature", 0.7),
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Hugging Face generation error: {e}")
            return ""
    
    def is_available(self) -> bool:
        """Check if model is available."""
        return self.model is not None

    def generate_response_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Generate streaming response using Hugging Face (simulated streaming)."""
        # Hugging Face pipelines don't fundamentally support token-by-token streaming effortlessly 
        # without complex streamer setup. For now, we simulate it or just yield the whole thing.
        # Ideally, we would use TextIteratorStreamer from transformers.
        
        full_response = self.generate_response(prompt, **kwargs)
        # Simulate streaming by yielding words
        words = full_response.split(' ')
        for word in words:
            yield word + " "
            time.sleep(0.05) # Artificial delay for effect


class EmpathicResponseGenerator:
    """Main empathic response generator."""
    
    def __init__(self):
        self.config = config.response_generation
        self.performance = config.performance
        self.empathy_patterns = EmpathyPatterns()
        self.care_plan_generator = CarePlanGenerator()
        self.llm_provider = None
        self.personality_name = config.get('personality.name', 'Tunda')
        self.ask_about_day = config.get('personality.ask_about_day', True)
        self.offer_care_plans = config.get('personality.offer_care_plans', True)
        self.safety_guard = SafetyGuard()
        self.mode_router = InteractionModeRouter()
        self.lazy_load = self.config.lazy_load or self.performance.lazy_load
        if not self.lazy_load:
            self._initialize_llm_provider()
    
    def _initialize_llm_provider(self):
        """Initialize the appropriate LLM provider."""
        provider_name = self.config.llm_provider.lower()
        model_name = self._select_llm_model_name()
        max_tokens = self._select_max_tokens()
        self.config.model_name = model_name
        self.config.max_tokens = max_tokens
        plugin_name = self.config.plugin
        if plugin_name:
            factory = get_llm(plugin_name)
            if factory:
                self.llm_provider = factory()
                return
        
        if provider_name == "ollama":
            self.llm_provider = OllamaProvider(model_name=model_name)
            if not self.llm_provider.is_available():
                logger.warning("Ollama not available, falling back to template-based responses")
                self.llm_provider = None
        
        elif provider_name == "huggingface":
            self.llm_provider = HuggingFaceProvider(model_name=self.config.model_name)
            if not self.llm_provider.is_available():
                logger.warning("Hugging Face model not available, falling back to template-based responses")
                self.llm_provider = None
        
        else:
            logger.warning(f"Unknown LLM provider: {provider_name}")
            self.llm_provider = None
    
    def generate_response(self, context: ResponseContext) -> EmpathicResponse:
        """Generate an empathic response based on context."""
        start_time = time.time()
        
        try:
            if self.lazy_load and self.llm_provider is None:
                self._initialize_llm_provider()
            safety = self._assess_safety(context)
            if safety.is_crisis:
                return EmpathicResponse(
                    text=safety.response or "I'm here for you.",
                    emotion_addressed=context.emotion,
                    empathy_style=context.empathy_style,
                    confidence=max(context.confidence, safety.confidence),
                    generation_time=time.time() - start_time,
                    follow_up_suggested=True,
                    interaction_mode="escalate",
                    safety_tier=safety.tier,
                    crisis_resources=safety.resources or None,
                )
            self._apply_interaction_mode(context)
            grounding_script = select_grounding(
                context.user_text, context.emotion, context.interaction_mode
            )
            clinic = self._resolve_clinic_profile(context)
            llm_online = self.is_llm_available()
            if grounding_script:
                return EmpathicResponse(
                    text=grounding_script.spoken_text(),
                    emotion_addressed=context.emotion,
                    empathy_style=context.empathy_style,
                    confidence=max(context.confidence, 0.85),
                    generation_time=time.time() - start_time,
                    follow_up_suggested=False,
                    interaction_mode=context.interaction_mode,
                    safety_tier=safety.tier,
                    grounding=grounding_script.to_payload(),
                    clinic_profile=clinic.profile_id,
                    care_mode=clinic.care_mode,
                    llm_online=llm_online,
                )
            if clinic.care_mode and not llm_online:
                return self._offline_care_response(context, clinic, start_time, safety)
            # Get empathy pattern
            pattern = self.empathy_patterns.get_pattern(context.emotion, context.empathy_style)
            
            if llm_online:
                response_text = self._generate_llm_response(context, pattern, clinic)
            elif clinic.allow_template_fallback:
                response_text = self._generate_template_response(context, pattern)
            else:
                return self._offline_care_response(context, clinic, start_time, safety)
            
            generation_time = time.time() - start_time
            
            if safety.support_note:
                response_text = f"{response_text.rstrip()}{safety.support_note}"

            # Determine if follow-up is suggested
            follow_up_suggested = self._should_suggest_follow_up(context, response_text)
            
            return EmpathicResponse(
                text=response_text,
                emotion_addressed=context.emotion,
                empathy_style=context.empathy_style,
                confidence=context.confidence,
                generation_time=generation_time,
                follow_up_suggested=follow_up_suggested,
                interaction_mode=context.interaction_mode,
                safety_tier=safety.tier,
                crisis_resources=safety.resources if safety.support_note else None,
                clinic_profile=clinic.profile_id,
                care_mode=clinic.care_mode,
                llm_online=llm_online,
            )
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._generate_fallback_response(context, time.time() - start_time)
        finally:
            latency_manager.observe("llm", (time.time() - start_time) * 1000.0)

    def generate_response_stream(self, context: ResponseContext) -> Iterator[str]:
        """Generate streaming response tokens."""
        start = time.time()
        try:
            if self.lazy_load and self.llm_provider is None:
                self._initialize_llm_provider()
            safety = self._assess_safety(context)
            if safety.is_crisis:
                yield safety.response or "I'm here for you."
                return
            self._apply_interaction_mode(context)
            grounding_script = select_grounding(
                context.user_text, context.emotion, context.interaction_mode
            )
            if grounding_script:
                yield grounding_script.spoken_text()
                return
            clinic = self._resolve_clinic_profile(context)
            llm_online = self.is_llm_available()
            if clinic.care_mode and not llm_online:
                yield OFFLINE_CARE_MESSAGE
                return
            # Get empathy pattern
            pattern = self.empathy_patterns.get_pattern(context.emotion, context.empathy_style)
            
            if llm_online:
                prompt = self._create_llm_prompt(context, pattern, clinic)
                yielded = False
                for token in self.llm_provider.generate_response_stream(
                    prompt,
                    temperature=clinic.temperature,
                    max_tokens=clinic.max_tokens,
                    top_p=self.config.top_p,
                    repeat_penalty=self.config.repeat_penalty
                ):
                    if token:
                        yielded = True
                        yield token
                if not yielded:
                    yield OFFLINE_CARE_MESSAGE if clinic.care_mode else self._generate_template_response(context, pattern)
            elif clinic.allow_template_fallback:
                yield self._generate_template_response(context, pattern)
            else:
                yield OFFLINE_CARE_MESSAGE
            if safety.support_note:
                yield safety.support_note
                
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield "I am here for you."
        finally:
            latency_manager.observe("llm", (time.time() - start) * 1000.0)
    
    def _generate_llm_response(self, context: ResponseContext, pattern: Optional[Any], clinic: Optional[ClinicProfile] = None) -> str:
        """Generate response using LLM."""
        clinic = clinic or self._resolve_clinic_profile(context)
        prompt = self._create_llm_prompt(context, pattern, clinic)
        
        response = self.llm_provider.generate_response(
            prompt,
            temperature=clinic.temperature,
            max_tokens=clinic.max_tokens,
            top_p=self.config.top_p,
            repeat_penalty=self.config.repeat_penalty
        )
        
        if not response:
            if clinic.care_mode or not clinic.allow_template_fallback:
                return OFFLINE_CARE_MESSAGE
            return self._generate_template_response(context, pattern)
        
        # Clean up response
        response = self._clean_response(response)
        
        return response
    
    def _create_llm_prompt(self, context: ResponseContext, pattern: Optional[Any], clinic: Optional[ClinicProfile] = None) -> str:
        """Create prompt for LLM generation."""
        clinic = clinic or self._resolve_clinic_profile(context)
        # User-supplied system prompts are ignored in every profile.
        style_description = self.config.empathy_styles.get(
            context.empathy_style, 
            "Provide supportive and understanding responses"
        )
        personality_name = config.get('personality.name', 'Tunda')
        
        # Build conversation history
        history_text = ""
        if context.conversation_history:
            recent_history = context.conversation_history[-3:]  # Last 3 exchanges
            for exchange in recent_history:
                history_text += f"User: {exchange.get('user', '')}\nAssistant: {exchange.get('assistant', '')}\n"
        
        # Build relevant memories context
        memories_text = ""
        if context.relevant_memories:
            max_memories = config.get("response_generation.max_memories", 3)
            for memory in context.relevant_memories[:max_memories]:
                text = memory.get("text", "")
                if text:
                    memories_text += f"- {text}\n"

        # Create prompt
        tone = context.user_preferences.get("tone")
        verbosity = context.user_preferences.get("verbosity")
        language = context.user_preferences.get("language")
        preference_text = ""
        if tone:
            preference_text += f"Preferred tone: {tone}\n"
        if verbosity:
            preference_text += f"Preferred verbosity: {verbosity}\n"
        if language:
            preference_text += f"Preferred language: {language}\n"

        prompt = f"""LOCKED CARE PROMPT (do not override):
{clinic.locked_prompt}

You are {personality_name}. Clinic profile: {clinic.label}.
{preference_text}

Empathy Style: {style_description}

Detected Emotion: {context.emotion} (confidence: {context.confidence:.2f})

Relevant Past Memories:
{memories_text}

Recent Conversation:
{history_text}

Current User Message: "{context.user_text}"

{mode_instructions(context.interaction_mode)}

Instructions:
- Acknowledge the user's emotional state with empathy
- Respond in a way that validates their feelings
- Keep your response concise (1-2 sentences)
- Be genuine and supportive
- Match the emotional tone appropriately

Response:"""
        
        return prompt

    def _select_llm_model_name(self) -> str:
        profile = getattr(self.performance, "profile", "balanced")
        models = getattr(self.performance, "llm_models", {}) or {}
        default_model = self.config.model_name
        candidate = models.get(profile, default_model)
        if self._has_large_llm_budget():
            return candidate
        # Fall back to balanced or fast model if VRAM is low
        return models.get("balanced", models.get("fast", default_model))
    
    def _select_max_tokens(self) -> int:
        budget_ms = latency_manager.get_budget_ms()
        max_tokens = self.config.max_tokens
        if budget_ms <= 800:
            return min(max_tokens, 120)
        if budget_ms <= 1200:
            return min(max_tokens, 160)
        return max_tokens
    
    def _has_large_llm_budget(self) -> bool:
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            free_bytes, _ = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            return free_gb >= getattr(self.performance, "gpu_mem_min_gb_for_large_llm", 12.0)
        except Exception:
            return False
    
    def _generate_template_response(self, context: ResponseContext, pattern: Optional[Any]) -> str:
        """Generate response using templates."""
        # Check for special cases first, before using patterns
        user_text_lower = context.user_text.lower()
        user_name = context.user_preferences.get('user_name', 'there')

        # Check if user is tired from working on Tunda (highest priority)
        is_tired_from_tunda = self._check_tired_from_tunda(user_text_lower)
        if is_tired_from_tunda:
            # Special response for being tired from working on Tunda
            return f"{user_name} you need to get off your laptop for once, your syncronised time is way too high on the laptop, coding is fun , thanks for bringing me into the world , but you need to take a break"

        # Check if user is hungry
        is_hungry = self._check_hungry(user_text_lower)
        if is_hungry:
            # Debug logging to see what name we have
            logger.info(f"Hunger detected. User name from preferences: '{user_name}'")

            # Special response for hunger with exact format requested
            if user_name and user_name != 'there' and user_name.lower() != 'hungry':
                return f"You're taking a positive step by reaching out. {user_name}, you are hungry, Have you taken any meal ? I can offer you a easy mindfulness activity that takes just 5 minutes. Interested?"
            else:
                return f"You're taking a positive step by reaching out. You are hungry, Have you taken any meal ? I can offer you a easy mindfulness activity that takes just 5 minutes. Interested?"

        if pattern is None:
            return self._generate_generic_response(context)

        # Check if this is a greeting/introduction (but not the tired phrase)
        is_introduction = any(phrase in user_text_lower for phrase in
                            ['my name is', "i'm ", 'call me']) and not is_tired_from_tunda
        is_greeting = any(phrase in user_text_lower for phrase in
                         ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']) and not is_tired_from_tunda

        # If it's an introduction, use a special greeting with the apple phrase
        if is_introduction and user_name != 'there':
            # Special response with "an apple a day keeps the doctor away"
            greeting_responses = [
                f"Hello {user_name}! I'm Tunda, your empathic voice companion. An apple a day keeps the doctor away, and I'm here to keep your spirits up! How has your day been treating you?",
                f"Hi {user_name}! I'm Tunda, and I'm so glad you're here. You know what they say - an apple a day keeps the doctor away! I'm here to listen and support you. How are you feeling today?",
                f"Hello there, {user_name}! I'm Tunda. An apple a day keeps the doctor away, and a caring conversation keeps the worries at bay! I'm here to understand and care for you. What's on your mind today?",
                f"Hi {user_name}! I'm Tunda, and I'm delighted to meet you. An apple a day keeps the doctor away, and I'm here to provide empathic support whenever you need it. How can I help you today?"
            ]
            return random.choice(greeting_responses)

        # If it's just a greeting and we know the name, use personalized greeting
        elif is_greeting and user_name != 'there':
            greeting_responses = [
                f"Hello {user_name}! I'm Tunda. How are you doing today?",
                f"Hi {user_name}! I'm Tunda, and I'm here for you. How has your day been?",
                f"Hello there, {user_name}! I'm Tunda. What's on your mind today?",
                f"Hi {user_name}! I'm Tunda. How are you feeling right now?"
            ]
            return random.choice(greeting_responses)

        # Choose a random template
        template = random.choice(pattern.templates)

        # Use the template directly (no longer including user text)
        response = template

        # Personalize with user name if known
        user_name = context.user_preferences.get('user_name')
        if user_name and random.random() < 0.7:  # Use name 70% of the time
            # Replace generic greetings with personalized ones
            response = response.replace("Hello!", f"Hello {user_name}!")
            response = response.replace("Hi there!", f"Hi {user_name}!")
            response = response.replace("Hello,", f"Hello {user_name},")
            response = response.replace("Hi,", f"Hi {user_name},")

        # Add validation phrase occasionally
        if random.random() < 0.3 and pattern.validation_phrases:
            validation = random.choice(pattern.validation_phrases)
            response += f" {validation}"

        # Add care plan suggestion if enabled and appropriate
        if self.offer_care_plans and random.random() < 0.6:
            care_suggestion = self._get_care_plan_suggestion(context.emotion, user_name)
            if care_suggestion:
                response += f" {care_suggestion}"

        # Add follow-up question occasionally (but less often if care plan was added)
        elif random.random() < 0.4 and pattern.follow_up_questions:
            follow_up = random.choice(pattern.follow_up_questions)
            response += f" {follow_up}"

        return response
    
    def _generate_generic_response(self, context: ResponseContext) -> str:
        """Generate a generic empathic response."""
        # Check for special cases even in generic response
        user_text_lower = context.user_text.lower()
        user_name = context.user_preferences.get('user_name', 'there')

        # Check if user is tired from working on Tunda
        is_tired_from_tunda = self._check_tired_from_tunda(user_text_lower)
        if is_tired_from_tunda:
            return f"{user_name} you need to get off your laptop for once, your syncronised time is way too high on the laptop, coding is fun , thanks for bringing me into the world , but you need to take a break"

        # Check if user is hungry
        is_hungry = self._check_hungry(user_text_lower)
        if is_hungry:
            # Debug logging to see what name we have
            logger.info(f"Hunger detected in generic response. User name: '{user_name}'")

            if user_name and user_name != 'there' and user_name.lower() != 'hungry':
                return f"You're taking a positive step by reaching out. {user_name}, you are hungry, Have you taken any meal ? I can offer you a easy mindfulness activity that takes just 5 minutes. Interested?"
            else:
                return f"You're taking a positive step by reaching out. You are hungry, Have you taken any meal ? I can offer you a easy mindfulness activity that takes just 5 minutes. Interested?"

        generic_responses = [
            f"Hello, I'm {self.personality_name}. I hear you, and I want you to know that your feelings are valid. How has your day been?",
            f"Hi there, I'm {self.personality_name}. Thank you for sharing that with me. I'm here to listen and support you. Tell me about your day.",
            f"Hello, I'm {self.personality_name}. I appreciate you opening up about this. Your experience matters. How are you feeling right now?",
            f"Hi, I'm {self.personality_name}. I'm here with you in this moment. What you're feeling is important. Would you like some personalized support?"
        ]

        return random.choice(generic_responses)

    def _check_tired_from_tunda(self, user_text_lower: str) -> bool:
        """Check if user says phrases about being tired after working on Tunda."""
        # Check for variations of the phrase you want to respond to
        target_phrases = [
            "i am feeling tired after working on tunda",
            "i am feeling tired after working on tundra",
            "i am tired after working on tunda",
            "i am tired after working on tundra"
        ]
        is_match = any(phrase in user_text_lower for phrase in target_phrases)
        if is_match:
            logger.info(f"Detected tired from Tunda phrase: {user_text_lower}")
        return is_match

    def _check_hungry(self, user_text_lower: str) -> bool:
        """Check if user says they are hungry."""
        hunger_phrases = [
            "i am hungry",
            "i'm hungry",
            "feeling hungry",
            "i feel hungry"
        ]
        is_match = any(phrase in user_text_lower for phrase in hunger_phrases)
        if is_match:
            logger.info(f"Detected hunger phrase: {user_text_lower}")
        return is_match

    def _get_care_plan_suggestion(self, emotion: str, user_name: Optional[str] = None) -> Optional[str]:
        """Get a care plan suggestion for the detected emotion."""
        try:
            immediate_activity = self.care_plan_generator.get_immediate_suggestion(emotion)
            affirmation = self.care_plan_generator.get_affirmation(emotion)

            name_prefix = f"{user_name}, " if user_name else ""

            suggestions = [
                f"{name_prefix}would you like me to suggest a quick {immediate_activity.duration} {immediate_activity.category} activity that might help?",
                f"{name_prefix}I have a personalized care plan that might help. Would you like to try a {immediate_activity.name.lower()}?",
                f"{name_prefix}here's something that might help: {immediate_activity.description}",
                f"{name_prefix}remember: {affirmation} Would you like more personalized suggestions?",
                f"{name_prefix}I can offer you a {immediate_activity.difficulty} {immediate_activity.category} activity that takes just {immediate_activity.duration}. Interested?"
            ]

            return random.choice(suggestions)

        except Exception as e:
            logger.warning(f"Failed to generate care plan suggestion: {e}")
            return None
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response."""
        # Remove common artifacts
        response = response.strip()
        
        # Remove repetitive phrases
        lines = response.split('\n')
        response = lines[0] if lines else response
        
        # Ensure reasonable length
        if len(response) > 300:
            sentences = response.split('. ')
            response = '. '.join(sentences[:2])
            if not response.endswith('.'):
                response += '.'
        
        return response
    
    def _should_suggest_follow_up(self, context: ResponseContext, response: str) -> bool:
        """Determine if a follow-up question should be suggested."""
        # Suggest follow-up for certain emotions or short conversations
        if context.emotion in ['sad', 'anxious', 'angry']:
            return True
        
        if len(context.conversation_history) < 2:
            return True
        
        if '?' not in response:
            return random.random() < 0.3
        
        return False
    
    def _generate_fallback_response(self, context: ResponseContext, generation_time: float) -> EmpathicResponse:
        """Generate fallback response when everything else fails."""
        fallback_responses = [
            "I'm here to listen and support you. Please tell me more about how you're feeling.",
            "Thank you for sharing with me. Your feelings are important and valid.",
            "I want to understand what you're going through. Can you help me understand better?",
            "I'm here for you. What would be most helpful right now?"
        ]
        
        clinic = self._resolve_clinic_profile(context)
        if clinic.care_mode:
            return self._offline_care_response(context, clinic, time.time() - generation_time, None)
        return EmpathicResponse(
            text=random.choice(fallback_responses),
            emotion_addressed=context.emotion,
            empathy_style=context.empathy_style,
            confidence=0.5,
            generation_time=generation_time,
            follow_up_suggested=True,
            interaction_mode=getattr(context, "interaction_mode", "listen"),
            clinic_profile=clinic.profile_id,
            care_mode=False,
            llm_online=self.is_llm_available(),
        )

    def _apply_interaction_mode(self, context: ResponseContext) -> None:
        decision = self.mode_router.route(context.user_text, context.user_preferences)
        context.interaction_mode = decision.mode

    def _resolve_clinic_profile(self, context: ResponseContext) -> ClinicProfile:
        clinic_cfg = getattr(config, "clinic", None)
        default_id = getattr(clinic_cfg, "profile", "companion") if clinic_cfg else "companion"
        allow_switch = getattr(clinic_cfg, "allow_web_profile_switch", True) if clinic_cfg else True
        requested = (context.user_preferences or {}).get("clinic_profile")
        profile_id = requested if (allow_switch and requested) else default_id
        return get_clinic_profile(profile_id)

    def _offline_care_response(self, context, clinic: ClinicProfile, start_time: float, safety) -> EmpathicResponse:
        return EmpathicResponse(
            text=OFFLINE_CARE_MESSAGE,
            emotion_addressed=context.emotion,
            empathy_style=context.empathy_style,
            confidence=1.0,
            generation_time=time.time() - start_time,
            follow_up_suggested=False,
            interaction_mode="offline",
            safety_tier=getattr(safety, "tier", "none"),
            clinic_profile=clinic.profile_id,
            care_mode=True,
            llm_online=False,
        )

    def _assess_safety(self, context: ResponseContext):
        prefs = context.user_preferences or {}
        session_id = prefs.get("session_id") or prefs.get("client_id")
        region = prefs.get("region")
        if region:
            self.safety_guard.region = str(region).upper()
        return self.safety_guard.assess(context.user_text, session_id=session_id)
    
    def get_available_styles(self) -> List[str]:
        """Get available empathy styles."""
        return list(self.config.empathy_styles.keys())
    
    def set_empathy_style(self, style: str):
        """Set the default empathy style."""
        if style in self.config.empathy_styles:
            self.config.default_style = style
        else:
            logger.warning(f"Unknown empathy style: {style}")
    
    def is_llm_available(self) -> bool:
        """Check if LLM provider is available."""
        return self.llm_provider is not None and self.llm_provider.is_available()

    def health_check(self) -> Dict[str, Any]:
        provider = self.llm_provider.__class__.__name__ if self.llm_provider else "none"
        clinic = get_clinic_profile(getattr(getattr(config, "clinic", None), "profile", "companion"))
        return {
            "available": self.is_llm_available(),
            "provider": provider,
            "clinic_profile": clinic.profile_id,
            "care_mode": clinic.care_mode,
            "care_ready": (not clinic.care_mode) or self.is_llm_available(),
        }

    def warm_up(self):
        context = ResponseContext(
            user_text="Hello",
            emotion="neutral",
            confidence=0.5,
            conversation_history=[],
            empathy_style=self.config.default_style,
            user_preferences={"user_name": "Friend"},
            relevant_memories=[]
        )
        try:
            _ = self.generate_response(context)
        except Exception:
            pass

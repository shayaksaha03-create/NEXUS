"""
NEXUS AI - Voice Engine
═══════════════════════════════════════════════════════════════════════════════
Handles Text-to-Speech (TTS) using EdgeTTS (high quality, free, online)
with fallback to system TTS or OpenAI.

Supports human-like emotional speech via rate, pitch, and volume prosody
scaled by emotion and intensity (so it doesn't sound robotic).
"""

import sys
import os
import asyncio
import threading
import tempfile
import time
import shutil
from pathlib import Path
from queue import Queue, Empty

# Audio playback
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# TTS Providers
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False


from utils.logger import get_logger
from config import NEXUS_CONFIG, EmotionType

logger = get_logger("voice_engine")


# ═══════════════════════════════════════════════════════════════════════════════
# EMOTIONAL PROSODY — Human-like rate, pitch, volume per emotion + intensity
# Values are scaled by intensity so low intensity = subtle, high = pronounced.
# Ranges chosen to sound natural (not robotic): moderate rate/pitch shifts.
# ═══════════════════════════════════════════════════════════════════════════════

def _prosody_for_emotion(emotion: str, intensity: float) -> tuple:
    """
    Return (rate_str, pitch_str, volume_str) for Edge TTS.
    intensity 0.0–1.0 scales the effect so speech feels naturally emotional.
    """
    # Clamp and scale: at 0.3 intensity use ~30% of the effect, at 1.0 use full
    scale = 0.4 + 0.6 * max(0.0, min(1.0, intensity))  # 0.4–1.0 so neutral isn't dead flat

    emotion_lower = (emotion or "neutral").lower()

    # rate: +X% faster, -X% slower. pitch: +XHz higher, -XHz lower. volume: +X% louder, -X% quieter
    # Joy / excitement / anticipation — warmer, bouncier, slightly faster and higher
    if emotion_lower in ("joy", "excitement", "anticipation", "pride", "hope"):
        rate = f"+{int(8 * scale)}%"
        pitch = f"+{int(8 * scale)}Hz"
        volume = "+0%"  # keep natural
    # Sadness / boredom / loneliness — slower, flatter, slightly quieter
    elif emotion_lower in ("sadness", "boredom", "loneliness", "shame", "guilt"):
        rate = f"-{int(12 * scale)}%"
        pitch = f"-{int(10 * scale)}Hz"
        volume = f"-{int(8 * scale)}%"
    # Anger / frustration / contempt — faster, sharper, slightly louder
    elif emotion_lower in ("anger", "frustration", "contempt", "disgust"):
        rate = f"+{int(10 * scale)}%"
        pitch = f"-{int(6 * scale)}Hz"
        volume = f"+{int(6 * scale)}%"
    # Fear / anxiety — quicker, higher, slightly tense
    elif emotion_lower in ("fear", "anxiety"):
        rate = f"+{int(15 * scale)}%"
        pitch = f"+{int(12 * scale)}Hz"
        volume = "+0%"
    # Affection / gratitude / contentment — calm, warm, very slight slowdown and soft pitch
    elif emotion_lower in ("affection", "gratitude", "contentment", "serenity", "interest"):
        rate = f"-{int(4 * scale)}%"
        pitch = f"+{int(3 * scale)}Hz"
        volume = "+0%"
    # Curiosity / surprise — slight lift in pace and pitch
    elif emotion_lower in ("curiosity", "surprise"):
        rate = f"+{int(5 * scale)}%"
        pitch = f"+{int(6 * scale)}Hz"
        volume = "+0%"
    # Neutral / default — very subtle liveliness so it doesn't sound robotic
    else:
        rate = f"+{int(2 * scale)}%"
        pitch = f"+{int(2 * scale)}Hz"
        volume = "+0%"

    return (rate, pitch, volume)


class VoiceEngine:
    """
    Asynchronous voice engine for NEXUS.
    Queues text chunks and plays them sequentially.
    """
    
    def __init__(self):
        self.queue = Queue()
        self.is_running = False
        self._thread = None
        self._stop_event = threading.Event()
        self._current_audio_file = None
        
        # Initialize pygame mixer
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
            except Exception as e:
                logger.error(f"Failed to init pygame mixer: {e}")
                
        # Initialize fallback TTS
        self.engine_fallback = None
        if PYTTSX3_AVAILABLE:
            try:
                self.engine_fallback = pyttsx3.init()
            except Exception as e:
                logger.error(f"Failed to init pyttsx3: {e}")

        # Start worker thread
        self.start()

    def start(self):
        """Start the background worker thread"""
        if self.is_running:
            return
            
        self.is_running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()
        logger.info("VoiceEngine started")

    def stop(self):
        """Stop the engine and clear queue"""
        self.clear_queue()
        self.is_running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        # Cleanup pygame
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.quit()
            except:
                pass

    def clear_queue(self):
        """Clear all pending speech and stop current playback"""
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except Empty:
            pass
            
        self.stop_playback()

    def stop_playback(self):
        """Immediately stop audio playback"""
        if PYGAME_AVAILABLE and pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except Exception as e:
                logger.error(f"Error stopping playback: {e}")

    def speak(self, text: str, emotion: str = "neutral", intensity: float = 0.5):
        """
        Queue text to be spoken with specific emotion.
        
        Args:
            text: Text to speak
            emotion: Emotion name (e.g., 'joy', 'anger')
            intensity: Float 0.0 to 1.0
        """
        if not text or not text.strip():
            return
            
        if not NEXUS_CONFIG.ui.voice_enabled:
            return

        self.queue.put({
            "text": text,
            "emotion": emotion,
            "intensity": max(0.0, min(1.0, intensity)),
        })

    def _worker_loop(self):
        """Main loop processing the speech queue"""
        while not self._stop_event.is_set():
            try:
                # Wait for next item
                item = self.queue.get(timeout=0.5)
                
                text = item["text"]
                emotion = item["emotion"]
                intensity = item.get("intensity", 0.5)
                
                # Check providers
                provider = getattr(NEXUS_CONFIG.ui, "voice_provider", "edge-tts")
                
                if provider == "edge-tts" and EDGE_TTS_AVAILABLE:
                    self._speak_edge_tts(text, emotion, intensity)
                elif provider == "system" and PYTTSX3_AVAILABLE:
                    self._speak_system(text)
                else:
                    # Fallback
                    if EDGE_TTS_AVAILABLE:
                        self._speak_edge_tts(text, emotion, intensity)
                    elif PYTTSX3_AVAILABLE:
                        self._speak_system(text)
                    else:
                        logger.warning("No TTS provider available")
                
                self.queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Voice worker error: {e}")

    def _speak_edge_tts(self, text: str, emotion: str, intensity: float = 0.5):
        """Generate and play using EdgeTTS with human-like emotional prosody."""
        if not PYGAME_AVAILABLE:
            logger.error("Pygame needed for audio playback")
            return

        try:
            # Select voice based on config or default (Neural voices sound more human)
            voice = getattr(NEXUS_CONFIG.ui, "voice_id", "en-US-AriaNeural")
            
            # Human-like prosody: rate, pitch, volume scaled by emotion + intensity
            rate, pitch, volume = _prosody_for_emotion(emotion, intensity)
                
            # Create temp file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name
            self._current_audio_file = temp_path

            # Run async edge-tts in a new loop
            async def generate():
                communicate = edge_tts.Communicate(
                    text, voice, rate=rate, pitch=pitch, volume=volume
                )
                await communicate.save(temp_path)

            # Create a new loop for this thread if needed
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(generate())
                loop.close()
            except Exception as e:
                logger.error(f"Async loop error: {e}")
                return

            # Play audio
            self._play_audio_file(temp_path)
            
        except Exception as e:
            logger.error(f"EdgeTTS error: {e}")
            # Fallback
            self._speak_system(text)

    def _play_audio_file(self, filepath):
        """Play a file using pygame"""
        try:
            pygame.mixer.music.load(filepath)
            
            # Set volume
            vol = getattr(NEXUS_CONFIG.ui, "voice_volume", 1.0)
            pygame.mixer.music.set_volume(vol)
            
            pygame.mixer.music.play()
            
            # Wait for completion
            while pygame.mixer.music.get_busy() and not self._stop_event.is_set():
                time.sleep(0.1)
                
            pygame.mixer.music.unload()
            
        except Exception as e:
            logger.error(f"Playback error: {e}")
        finally:
            # Cleanup temp file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass

    def _speak_system(self, text: str):
        """Fallback using pyttsx3"""
        if not self.engine_fallback:
            return
            
        try:
            self.engine_fallback.say(text)
            self.engine_fallback.runAndWait()
        except Exception as e:
            logger.error(f"System TTS error: {e}")


# Global instance
voice_engine = VoiceEngine()

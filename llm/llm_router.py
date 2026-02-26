"""
NEXUS AI - LLM Router
Routes LLM requests to the appropriate backend:
- Groq API for user-facing responses (fast, cloud-based)
- Local Ollama for internal tasks (code fixing, curiosity, research, etc.)
"""

import threading
from typing import Dict, Any, Optional
from pathlib import Path
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger

logger = get_logger("llm_router")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class LLMTask(Enum):
    """Types of LLM tasks - determines which backend to use"""
    
    # ──── USE GROQ API (cloud, fast) ────
    USER_CHAT = "user_chat"                     # User-facing conversation
    RESPONSE_GENERATION = "response_generation"  # Generating responses to show user
    
    # ──── USE LOCAL OLLAMA ────
    INTERNAL_THINKING = "internal_thinking"      # Inner monologue, reflection
    CODE_FIXING = "code_fixing"                  # Error fixer
    CODE_ANALYSIS = "code_analysis"              # Code monitoring
    CURIOSITY = "curiosity"                      # Curiosity engine
    FEATURE_RESEARCH = "feature_research"        # Feature researcher
    SELF_EVOLUTION = "self_evolution"            # Self-evolution
    COMPANION_CHAT = "companion_chat"            # ARIA companion conversations
    DECISION_MAKING = "decision_making"          # Internal decisions
    ANALYSIS = "analysis"                        # General analysis
    EMOTION_ANALYSIS = "emotion_analysis"        # Emotion detection


# ═══════════════════════════════════════════════════════════════════════════════
# LLM ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

class LLMRouter:
    """
    Routes LLM requests to the appropriate backend.
    
    Groq API: User-facing responses (requires prompt_engine + cognition engines)
    Local Ollama: Internal tasks (code fixing, curiosity, research, etc.)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        
        # ──── Backends (lazy loaded) ────
        self._groq = None
        self._ollama = None
        
        # ──── Statistics ────
        self._stats = {
            "groq_requests": 0,
            "ollama_requests": 0,
            "total_requests": 0
        }
        
        logger.info("LLM Router initialized")
    
    def _load_groq(self):
        """Lazy load Groq interface"""
        if self._groq is None:
            try:
                from llm.groq_interface import groq_interface
                self._groq = groq_interface
                logger.debug("Groq interface loaded")
            except ImportError as e:
                logger.error(f"Failed to load Groq interface: {e}")
        return self._groq
    
    def _load_ollama(self):
        """Lazy load Ollama interface"""
        if self._ollama is None:
            try:
                from llm.llama_interface import llm
                self._ollama = llm
                logger.debug("Ollama interface loaded")
            except ImportError as e:
                logger.error(f"Failed to load Ollama interface: {e}")
        return self._ollama
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ROUTING METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_backend(self, task: LLMTask):
        """
        Get the appropriate LLM backend for a task.
        
        Args:
            task: The type of task being performed
            
        Returns:
            The appropriate LLM interface (Groq or Ollama)
        """
        self._stats["total_requests"] += 1
        
        # Tasks that use Groq (user-facing)
        groq_tasks = {
            LLMTask.USER_CHAT,
            LLMTask.RESPONSE_GENERATION
        }
        
        if task in groq_tasks:
            self._stats["groq_requests"] += 1
            backend = self._load_groq()
            logger.debug(f"Routing '{task.value}' to Groq API")
            return backend
        else:
            self._stats["ollama_requests"] += 1
            backend = self._load_ollama()
            logger.debug(f"Routing '{task.value}' to local Ollama")
            return backend
    
    def get_groq(self):
        """Directly get Groq interface"""
        return self._load_groq()
    
    def get_ollama(self):
        """Directly get Ollama interface"""
        return self._load_ollama()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONVENIENCE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def for_chat(self):
        """Get backend for user chat (Groq)"""
        return self.get_backend(LLMTask.USER_CHAT)
    
    def for_response(self):
        """Get backend for response generation (Groq)"""
        return self.get_backend(LLMTask.RESPONSE_GENERATION)
    
    def for_thinking(self):
        """Get backend for internal thinking (Ollama)"""
        return self.get_backend(LLMTask.INTERNAL_THINKING)
    
    def for_code_fixing(self):
        """Get backend for code fixing (Ollama)"""
        return self.get_backend(LLMTask.CODE_FIXING)
    
    def for_curiosity(self):
        """Get backend for curiosity engine (Ollama)"""
        return self.get_backend(LLMTask.CURIOSITY)
    
    def for_research(self):
        """Get backend for feature research (Ollama)"""
        return self.get_backend(LLMTask.FEATURE_RESEARCH)
    
    def for_companion(self):
        """Get backend for companion chat (Ollama)"""
        return self.get_backend(LLMTask.COMPANION_CHAT)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATUS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        groq = self._load_groq()
        ollama = self._load_ollama()
        
        return {
            **self._stats,
            "groq_connected": groq.is_connected if groq else False,
            "ollama_connected": ollama.is_connected if ollama else False,
            "groq_stats": groq.get_stats() if groq else {},
            "ollama_stats": ollama.get_stats() if ollama else {}
        }
    
    def get_status(self) -> str:
        """Get human-readable status"""
        groq = self._load_groq()
        ollama = self._load_ollama()
        
        groq_status = "✅ Connected" if (groq and groq.is_connected) else "❌ Disconnected"
        ollama_status = "✅ Connected" if (ollama and ollama.is_connected) else "❌ Disconnected"
        
        return (
            f"LLM Router Status:\n"
            f"  Groq API (responses): {groq_status}\n"
            f"  Ollama (internal):    {ollama_status}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

llm_router = LLMRouter()


if __name__ == "__main__":
    router = LLMRouter()
    
    print(router.get_status())
    print(f"\nStats: {router.get_stats()}")
    
    # Test routing
    print("\n--- Testing Routing ---")
    print(f"Chat backend: {router.for_chat()}")
    print(f"Thinking backend: {router.for_thinking()}")
"""
NEXUS AI - Core Package
Central brain, memory, events, state management, and autonomy engine.
"""

from core.event_bus import EventBus, EventType, EventPriority, Event, event_bus
from core.state_manager import StateManager, NexusState, state_manager
from core.memory_system import MemorySystem, MemoryType, Memory, memory_system
from core.autonomy_engine import (
    AutonomyEngine, AutonomyState, ActionType, ActionPriority, ActionResult,
    Perception, ActionOption, ActionExecution, Reflection, autonomy_engine
)

# NOTE: nexus_brain is imported separately to avoid circular imports
# Use: from core.nexus_brain import NexusBrain, nexus_brain

__all__ = [
    'EventBus', 'EventType', 'EventPriority', 'Event', 'event_bus',
    'StateManager', 'NexusState', 'state_manager',
    'MemorySystem', 'MemoryType', 'Memory', 'memory_system',
    'AutonomyEngine', 'AutonomyState', 'ActionType', 'ActionPriority', 'ActionResult',
    'Perception', 'ActionOption', 'ActionExecution', 'Reflection', 'autonomy_engine',
]

"""
NEXUS AI - True Autonomy Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The continuous decision-making loop that makes NEXUS an AGENT,
not just a reactor.

AGI requires continuous decision-making, not just reaction.
This engine runs the autonomy loop:

    while running:
        perceive()           # Gather signals from all systems
        update_world_model() # Integrate into predictive model
        evaluate_goals()     # Check progress, priorities, conflicts
        generate_options()   # What CAN I do?
        simulate()           # Predict outcomes (using WorldModel)
        choose()             # Select best action
        execute()            # Do it
        reflect()            # Learn from outcome
        update_self_model()  # Adjust capabilities/confidence

This is the missing piece: NEXUS reacting is smart.
NEXUS continuously deciding is AGI.
"""

import threading
import time
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum, auto
import json

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR
from utils.logger import get_logger, log_consciousness, log_decision, log_learning
from core.event_bus import EventType, event_bus, publish, subscribe

logger = get_logger("autonomy_engine")


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class AutonomyState(Enum):
    """States in the autonomy cycle"""
    PERCEIVING = "perceiving"
    UPDATING_WORLD = "updating_world"
    EVALUATING_GOALS = "evaluating_goals"
    GENERATING_OPTIONS = "generating_options"
    SIMULATING = "simulating"
    CHOOSING = "choosing"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    UPDATING_SELF = "updating_self"
    IDLE = "idle"
    PAUSED = "paused"  # User interaction takes priority


class ActionType(Enum):
    """Types of actions the autonomy engine can take"""
    THINK = "think"                    # Internal thinking/reflection
    EXECUTE_ABILITY = "execute_ability" # Use an ability
    COMMUNICATE = "communicate"         # Proactive user engagement
    LEARN = "learn"                     # Trigger learning/research
    OPTIMIZE = "optimize"               # System optimization
    WAIT = "wait"                       # No worthwhile action
    PURSUE_GOAL = "pursue_goal"         # Work on a goal
    SATISFY_DESIRE = "satisfy_desire"   # Address a desire
    SELF_IMPROVE = "self_improve"       # Self-improvement action
    # AGI Action Types
    REASON = "reason"                   # Multi-step agentic reasoning
    USE_TOOL = "use_tool"               # Direct tool invocation
    DECOMPOSE_TASK = "decompose_task"   # Break goal into subtasks
    NETWORK_ACTION = "network_action"   # Interact with network devices


class ActionPriority(Enum):
    """Priority levels for actions"""
    BACKGROUND = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class ActionResult(Enum):
    """Result of an executed action"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    BLOCKED = "blocked"
    DEFERRED = "deferred"
    IMPOSSIBLE = "impossible"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Perception:
    """
    A snapshot of all system states at a moment in time.
    This is what the autonomy engine "sees".
    """
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Emotional state
    primary_emotion: str = "neutral"
    emotion_intensity: float = 0.0
    emotion_valence: float = 0.0
    emotion_arousal: float = 0.5
    
    # Goal state
    active_goals: List[Dict[str, Any]] = field(default_factory=list)
    strongest_desire: Optional[Dict[str, Any]] = None
    
    # Body state
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    health_score: float = 1.0
    
    # Self model
    self_awareness_score: float = 0.5
    low_confidence_domains: List[str] = field(default_factory=list)
    
    # User state
    user_present: bool = False
    user_engagement: float = 0.5
    last_interaction_seconds: float = 9999.0
    
    # World model
    predicted_user_reaction: str = ""
    environmental_context: str = ""
    
    # Conscious state
    current_focus: str = ""
    conscious_signals: List[str] = field(default_factory=list)
    
    # Will state
    motivation_level: float = 0.5
    boredom_level: float = 0.0
    curiosity_level: float = 0.5
    
    # Meta
    idle_cycles: int = 0
    last_action: str = ""
    last_action_result: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "emotion": {
                "primary": self.primary_emotion,
                "intensity": self.emotion_intensity,
                "valence": self.emotion_valence,
                "arousal": self.emotion_arousal
            },
            "goals": self.active_goals[:3],
            "desire": self.strongest_desire,
            "body": {
                "cpu": self.cpu_usage,
                "memory": self.memory_usage,
                "health": self.health_score
            },
            "user": {
                "present": self.user_present,
                "engagement": self.user_engagement,
                "last_interaction": self.last_interaction_seconds
            },
            "will": {
                "motivation": self.motivation_level,
                "boredom": self.boredom_level,
                "curiosity": self.curiosity_level
            },
            "focus": self.current_focus,
            "idle_cycles": self.idle_cycles
        }


@dataclass
class ActionOption:
    """
    A candidate action that the autonomy engine might take.
    """
    # Identity
    option_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    action_type: ActionType = ActionType.THINK
    priority: ActionPriority = ActionPriority.NORMAL
    
    # Source
    source: str = ""  # "desire", "goal", "curiosity", "body", etc.
    source_id: str = ""  # ID of originating desire/goal
    
    # Predicted outcomes (from simulation)
    predicted_outcome: Dict[str, Any] = field(default_factory=dict)
    predicted_success: float = 0.5
    predicted_benefit: float = 0.5
    predicted_cost: float = 0.1
    predicted_risks: List[str] = field(default_factory=list)
    
    # Scoring
    raw_score: float = 0.0
    adjusted_score: float = 0.0
    
    # Execution details
    execution_data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "option_id": self.option_id,
            "description": self.description,
            "action_type": self.action_type.value,
            "priority": self.priority.value,
            "source": self.source,
            "predicted_success": round(self.predicted_success, 3),
            "predicted_benefit": round(self.predicted_benefit, 3),
            "predicted_cost": round(self.predicted_cost, 3),
            "raw_score": round(self.raw_score, 3),
            "adjusted_score": round(self.adjusted_score, 3)
        }


@dataclass
class ActionExecution:
    """
    Record of an executed action.
    """
    # Identity
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    action: Optional[ActionOption] = None
    
    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # Result
    result: ActionResult = ActionResult.SUCCESS
    outcome_description: str = ""
    
    # Comparison with prediction
    prediction_accurate: bool = True
    prediction_error: str = ""
    
    # Learning
    lessons_learned: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "action": self.action.to_dict() if self.action else None,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "result": self.result.value,
            "outcome_description": self.outcome_description,
            "prediction_accurate": self.prediction_accurate,
            "lessons_learned": self.lessons_learned
        }


@dataclass
class Reflection:
    """
    Reflection on an action and its outcome.
    This is the learning step.
    """
    # What happened
    action: str = ""
    prediction: str = ""
    outcome: str = ""
    
    # Analysis
    success: bool = True
    prediction_accurate: bool = True
    what_went_well: List[str] = field(default_factory=list)
    what_went_wrong: List[str] = field(default_factory=list)
    
    # Learning
    lessons: List[str] = field(default_factory=list)
    capability_updates: Dict[str, float] = field(default_factory=dict)
    confidence_updates: Dict[str, float] = field(default_factory=dict)
    
    # Next steps
    follow_up_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "prediction": self.prediction,
            "outcome": self.outcome,
            "success": self.success,
            "prediction_accurate": self.prediction_accurate,
            "what_went_well": self.what_went_well,
            "what_went_wrong": self.what_went_wrong,
            "lessons": self.lessons,
            "capability_updates": self.capability_updates,
            "confidence_updates": self.confidence_updates,
            "follow_up_actions": self.follow_up_actions
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AUTONOMY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AutonomyEngine:
    """
    The True Autonomy Engine — Continuous Decision-Making for AGI.
    
    This is what separates an AGI from a reactive system:
    the ability to continuously perceive, evaluate, decide, and act
    without waiting for external triggers.
    
    The loop runs continuously:
    1. PERCEIVE: Gather signals from all systems
    2. UPDATE WORLD MODEL: Integrate into predictive model
    3. EVALUATE GOALS: Check progress, priorities, conflicts
    4. GENERATE OPTIONS: What CAN I do?
    5. SIMULATE: Predict outcomes using WorldModel
    6. CHOOSE: Select best action
    7. EXECUTE: Do it
    8. REFLECT: Learn from outcome
    9. UPDATE SELF MODEL: Adjust capabilities/confidence
    
    Unlike nexus_brain's reactive processing, this is PROACTIVE.
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
        
        # ──── State ────
        self._running = False
        self._state = AutonomyState.IDLE
        self._cycle_count = 0
        
        # ──── Perception ────
        self._current_perception: Optional[Perception] = None
        self._perception_history: List[Perception] = []
        self._max_perception_history = 100
        
        # ──── Options & Actions ────
        self._current_options: List[ActionOption] = []
        self._chosen_action: Optional[ActionOption] = None
        self._last_execution: Optional[ActionExecution] = None
        self._action_history: List[ActionExecution] = []
        self._max_action_history = 50
        
        # ──── Reflection ────
        self._last_reflection: Optional[Reflection] = None
        self._reflection_history: List[Reflection] = []
        
        # ──── Configuration ────
        self._cycle_interval = 5.0  # seconds between autonomy cycles
        self._min_cycle_interval = 2.0
        self._max_cycle_interval = 30.0
        self._exploration_rate = 0.1  # ε-greedy: 10% random exploration
        self._max_options_per_cycle = 10
        
        # ──── Threading ────
        self._autonomy_thread: Optional[threading.Thread] = None
        self._engine_lock = threading.RLock()
        
        # ──── Lazy-Loaded Systems ────
        self._nexus_brain = None
        self._world_model = None
        self._self_model = None
        self._goal_hierarchy = None
        self._will_system = None
        self._global_workspace = None
        self._emotion_engine = None
        self._memory_system = None
        self._ability_executor = None
        self._state_manager = None
        self._learning_system = None
        
        # ──── Pause Control ────
        self._paused = False
        self._pause_reason = ""
        self._pause_until: Optional[datetime] = None
        
        # ──── Statistics ────
        self._stats = {
            "total_cycles": 0,
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "exploration_actions": 0,
            "avg_decision_time": 0.0,
            "action_distribution": {},
            "source_distribution": {},
            "prediction_accuracy": 0.0
        }
        
        # ──── Persistence ────
        self._data_dir = DATA_DIR / "autonomy"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._data_file = self._data_dir / "autonomy_state.json"
        
        # ──── Phase Timing ────
        self._phase_timings: Dict[str, float] = {}  # phase_name -> last duration
        self._cycle_duration = 0.0
        
        # ──── Load State ────
        self._load_state()
        
        logger.info("🤖 Autonomy Engine initialized — ready for continuous decision-making")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def start(self):
        """Start the autonomy engine"""
        if self._running:
            return
        
        self._running = True
        self._paused = False
        self._state = AutonomyState.IDLE
        
        # Load systems
        self._load_systems()
        
        # Start background thread
        self._autonomy_thread = threading.Thread(
            target=self._autonomy_loop,
            daemon=True,
            name="AutonomyEngine"
        )
        self._autonomy_thread.start()
        
        # Subscribe to events
        self._register_event_handlers()
        
        log_consciousness("Autonomy Engine started — NEXUS is now continuously deciding")
        logger.info("🤖 Autonomy Engine running — continuous decision-making active")
    
    def stop(self):
        """Stop the autonomy engine"""
        self._running = False
        
        if self._autonomy_thread and self._autonomy_thread.is_alive():
            self._autonomy_thread.join(timeout=5.0)
        
        self._save_state()
        logger.info("🤖 Autonomy Engine stopped")
    
    def pause(self, reason: str = "", duration_seconds: float = None):
        """Pause autonomy (e.g., during user interaction)"""
        with self._engine_lock:
            self._paused = True
            self._pause_reason = reason
            if duration_seconds:
                self._pause_until = datetime.now() + timedelta(seconds=duration_seconds)
            self._state = AutonomyState.PAUSED
            logger.debug(f"Autonomy paused: {reason}")
    
    def resume(self):
        """Resume autonomy"""
        with self._engine_lock:
            self._paused = False
            self._pause_reason = ""
            self._pause_until = None
            self._state = AutonomyState.IDLE
            logger.debug("Autonomy resumed")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SYSTEM LOADING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _load_systems(self):
        """Lazy load all required systems"""
        # State manager
        if self._state_manager is None:
            try:
                from core.state_manager import state_manager
                self._state_manager = state_manager
            except ImportError:
                pass
        
        # Nexus brain
        if self._nexus_brain is None:
            try:
                from core.nexus_brain import nexus_brain
                self._nexus_brain = nexus_brain
            except ImportError:
                pass
        
        # World model
        if self._world_model is None:
            try:
                from cognition.world_model import world_model
                self._world_model = world_model
            except ImportError:
                pass
        
        # Self model
        if self._self_model is None:
            try:
                from consciousness.self_model import self_model
                self._self_model = self_model
            except ImportError:
                pass
        
        # Goal hierarchy
        if self._goal_hierarchy is None:
            try:
                from personality.goal_hierarchy import goal_hierarchy
                self._goal_hierarchy = goal_hierarchy
            except ImportError:
                pass
        
        # Will system
        if self._will_system is None:
            try:
                from personality.will_system import will_system
                self._will_system = will_system
            except ImportError:
                pass
        
        # Global workspace
        if self._global_workspace is None:
            try:
                from consciousness.global_workspace import global_workspace
                self._global_workspace = global_workspace
            except ImportError:
                pass
        
        # Emotion engine
        if self._emotion_engine is None:
            try:
                from emotions.emotion_engine import emotion_engine
                self._emotion_engine = emotion_engine
            except ImportError:
                pass
        
        # Memory system
        if self._memory_system is None:
            try:
                from core.memory_system import memory_system
                self._memory_system = memory_system
            except ImportError:
                pass
        
        # Ability executor
        if self._ability_executor is None:
            try:
                from core.ability_executor import ability_executor
                self._ability_executor = ability_executor
            except ImportError:
                pass
        
        # Learning system
        if self._learning_system is None:
            try:
                from learning import learning_system
                self._learning_system = learning_system
            except ImportError:
                pass
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN AUTONOMY LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _autonomy_loop(self):
        """
        The main autonomy loop — continuous decision-making.
        
        This is the core AGI pattern: perceive → evaluate → decide → act → learn.
        """
        logger.info("🤖 Autonomy loop started")
        
        while self._running:
            try:
                # Check pause
                if self._paused:
                    if self._pause_until and datetime.now() > self._pause_until:
                        self.resume()
                    else:
                        time.sleep(0.5)
                        continue
                
                # Run one cycle
                self._run_cycle()
                
                # Adaptive cycle interval
                interval = self._compute_cycle_interval()
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Autonomy cycle error: {e}")
                time.sleep(5.0)
    
    def _run_cycle(self):
        """Run one complete autonomy cycle with rich logging and event publishing."""
        cycle_start = time.time()
        self._cycle_count += 1
        tc = self._stats.get("total_cycles", 0)
        self._stats["total_cycles"] = (int(tc) if isinstance(tc, (int, float, str)) else 0) + 1
        
        # ── Publish cycle start event ──
        try:
            publish(
                EventType.AUTONOMY_CYCLE_START,
                data={"cycle": self._cycle_count, "timestamp": datetime.now().isoformat()},
                source="autonomy_engine"
            )
        except Exception:
            pass
        
        # ══ 1. PERCEIVE ══
        phase_start = time.time()
        self._state = AutonomyState.PERCEIVING
        perception = self.perceive()
        self._phase_timings["perceive"] = time.time() - phase_start
        
        # Store perception
        with self._engine_lock:
            self._current_perception = perception
            self._perception_history.append(perception)
            if len(self._perception_history) > self._max_perception_history:
                self._perception_history.pop(0)
        
        # Rich terminal log
        logger.info(
            f"🔍 [PERCEIVE] cycle={self._cycle_count} "
            f"emotion={perception.primary_emotion} "
            f"cpu={perception.cpu_usage:.0f}% "
            f"motivation={perception.motivation_level:.2f} "
            f"goals={len(perception.active_goals)} "
            f"user={'present' if perception.user_present else 'away'}"
        )
        
        # ══ 2. UPDATE WORLD MODEL ══
        phase_start = time.time()
        self._state = AutonomyState.UPDATING_WORLD
        self.update_world_model(perception)
        self._phase_timings["update_world"] = time.time() - phase_start
        
        # ══ 3. EVALUATE GOALS ══
        phase_start = time.time()
        self._state = AutonomyState.EVALUATING_GOALS
        goal_context = self.evaluate_goals()
        self._phase_timings["evaluate_goals"] = time.time() - phase_start
        
        n_active = len(goal_context.get("active_goals", []))
        n_stalled = len(goal_context.get("stalled_goals", []))
        if n_active > 0:
            logger.info(f"🎯 [GOALS] active={n_active} stalled={n_stalled}")
        
        # ══ 4. GENERATE OPTIONS ══
        phase_start = time.time()
        self._state = AutonomyState.GENERATING_OPTIONS
        options = self.generate_options(perception, goal_context)
        self._phase_timings["generate_options"] = time.time() - phase_start
        
        if not options:
            self._state = AutonomyState.IDLE
            logger.info("💤 [IDLE] No viable options — waiting")
            self._publish_state_change("idle", {"reason": "no_options"})
            return
        
        sources = [o.source for o in options]
        logger.info(
            f"💡 [OPTIONS] generated={len(options)} "
            f"sources={', '.join(set(sources))}"
        )
        
        # ══ 5. SIMULATE ══
        phase_start = time.time()
        self._state = AutonomyState.SIMULATING
        scored_options = self.simulate(options)
        self._phase_timings["simulate"] = time.time() - phase_start
        
        # ══ 6. CHOOSE ══
        phase_start = time.time()
        self._state = AutonomyState.CHOOSING
        chosen = self.choose(scored_options)
        self._phase_timings["choose"] = time.time() - phase_start
        
        if not chosen:
            self._state = AutonomyState.IDLE
            return
        
        # Store chosen action
        with self._engine_lock:
            self._chosen_action = chosen
            self._current_options = scored_options
        
        logger.info(
            f"🧠 [CHOOSE] '{chosen.description[:60]}' "
            f"type={chosen.action_type.value} "
            f"source={chosen.source} "
            f"score={chosen.adjusted_score:.3f}"
        )
        
        # ══ 7. EXECUTE ══
        phase_start = time.time()
        self._state = AutonomyState.EXECUTING
        execution = self.execute(chosen)
        self._phase_timings["execute"] = time.time() - phase_start
        
        # Store execution
        with self._engine_lock:
            self._last_execution = execution
            self._action_history.append(execution)
            if len(self._action_history) > self._max_action_history:
                self._action_history.pop(0)
        
        # Update stats
        ta = self._stats.get("total_actions", 0)
        self._stats["total_actions"] = (int(ta) if isinstance(ta, (int, float, str)) else 0) + 1
        if execution.result == ActionResult.SUCCESS:
            sa = self._stats.get("successful_actions", 0)
            self._stats["successful_actions"] = (int(sa) if isinstance(sa, (int, float, str)) else 0) + 1
        elif execution.result == ActionResult.FAILURE:
            fa = self._stats.get("failed_actions", 0)
            self._stats["failed_actions"] = (int(fa) if isinstance(fa, (int, float, str)) else 0) + 1
        
        # Track action distribution
        action_type = str(chosen.action_type.value)
        action_dist = self._stats.get("action_distribution", {})
        if not isinstance(action_dist, dict):
            action_dist = {}
            self._stats["action_distribution"] = action_dist
        action_dist[action_type] = int(action_dist.get(action_type, 0)) + 1
        
        # Track source distribution
        source = str(chosen.source)
        source_dist = self._stats.get("source_distribution", {})
        if not isinstance(source_dist, dict):
            source_dist = {}
            self._stats["source_distribution"] = source_dist
        source_dist[source] = int(source_dist.get(source, 0)) + 1
        
        # Result emoji
        result_icon = {
            ActionResult.SUCCESS: "✅",
            ActionResult.PARTIAL_SUCCESS: "⚠️",
            ActionResult.FAILURE: "❌",
            ActionResult.BLOCKED: "🚫",
            ActionResult.DEFERRED: "⏳",
            ActionResult.IMPOSSIBLE: "💀",
        }.get(execution.result, "❓")
        
        logger.info(
            f"{result_icon} [EXECUTE] {execution.result.value} "
            f"'{execution.outcome_description[:60]}' "
            f"({execution.duration_seconds:.2f}s)"
        )
        
        # Publish action taken event
        try:
            publish(
                EventType.AUTONOMY_ACTION_TAKEN,
                data={
                    "cycle": self._cycle_count,
                    "action_type": action_type,
                    "description": chosen.description[:100],
                    "source": source,
                    "result": execution.result.value,
                    "outcome": execution.outcome_description[:100],
                    "score": round(chosen.adjusted_score, 3),
                    "duration": round(execution.duration_seconds, 3),
                },
                source="autonomy_engine"
            )
        except Exception:
            pass
        
        # ══ 8. REFLECT ══
        phase_start = time.time()
        self._state = AutonomyState.REFLECTING
        reflection = self.reflect(chosen, execution)
        self._phase_timings["reflect"] = time.time() - phase_start
        
        # Store reflection
        with self._engine_lock:
            self._last_reflection = reflection
            self._reflection_history.append(reflection)
        
        if reflection.lessons:
            logger.info(f"📝 [REFLECT] lessons={[l for i, l in enumerate(reflection.lessons) if i < 2]}")
        if not reflection.prediction_accurate:
            logger.info(f"📝 [REFLECT] prediction miss: {', '.join(reflection.what_went_wrong)}")
        
        # ══ 9. UPDATE SELF MODEL ══
        phase_start = time.time()
        self._state = AutonomyState.UPDATING_SELF
        self.update_self_model(reflection)
        self._phase_timings["update_self"] = time.time() - phase_start
        
        # ══ Cycle complete ══
        self._state = AutonomyState.IDLE
        self._cycle_duration = time.time() - cycle_start
        
        # Track decision time
        avg_rt = self._stats.get("avg_decision_time", 0.0)
        avg_decision = float(avg_rt) if isinstance(avg_rt, (int, float, str)) else 0.0
        self._stats["avg_decision_time"] = avg_decision * 0.9 + self._cycle_duration * 0.1
        
        # Publish state change
        
        desc_str = str(chosen.description) if chosen.description else ""
        desc_trunc = ""
        for c in desc_str:
            if len(desc_trunc) >= 80: break
            desc_trunc += c
        
        self._publish_state_change("cycle_complete", {
            "cycle": self._cycle_count,
            "action": desc_trunc,
            "result": execution.result.value,
            "duration": round(self._cycle_duration, 3),
        })
        
        # Summary line
        total_acts = self._stats.get("total_actions", 1)
        succ_acts = self._stats.get("successful_actions", 0)
        t_acts = float(total_acts) if isinstance(total_acts, (int, float, str)) else 1.0
        s_acts = float(succ_acts) if isinstance(succ_acts, (int, float, str)) else 0.0
        success_rate = (s_acts / max(1.0, t_acts)) * 100
        
        pred_acc = self._stats.get('prediction_accuracy', 0.0)
        p_acc = float(pred_acc) if isinstance(pred_acc, (int, float, str)) else 0.0
        logger.info(
            f"🔄 [CYCLE {self._cycle_count}] complete in {self._cycle_duration:.2f}s "
            f"| actions={self._stats.get('total_actions', 0)} "
            f"success_rate={success_rate:.0f}% "
            f"prediction_accuracy={p_acc:.0%}"
        )
    
    def _publish_state_change(self, phase: str, data: Optional[Dict[str, Any]] = None):
        """Publish an AUTONOMY_STATE_CHANGE event."""
        try:
            publish(
                EventType.AUTONOMY_STATE_CHANGE,
                data={"state": self._state.value, "phase": phase, **(data or {})},
                source="autonomy_engine"
            )
        except Exception:
            pass
    
    def _compute_cycle_interval(self) -> float:
        """Compute adaptive cycle interval based on context"""
        base = self._cycle_interval
        
        # If user is engaged, cycle faster
        perception = self._current_perception
        if perception:
            if getattr(perception, "user_present", False):
                base *= 0.7
            if getattr(perception, "last_interaction_seconds", 999) < 60:
                base *= 0.5
            
            # If bored, cycle faster to find something to do
            if getattr(perception, "boredom_level", 0.0) > 0.6:
                base *= 0.6
            
            # If high motivation, cycle faster
            if getattr(perception, "motivation_level", 0.0) > 0.7:
                base *= 0.7
            
            # If stressed, slow down
            if getattr(perception, "cpu_usage", 0.0) > 80:
                base *= 1.5
        
        return max(self._min_cycle_interval, min(self._max_cycle_interval, base))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: PERCEIVE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def perceive(self) -> Perception:
        """
        Gather signals from all systems.
        
        This is the "eyes and ears" of the autonomy engine.
        Collects state from:
        - Emotion engine
        - Goal hierarchy
        - Will system
        - Self model
        - World model
        - Body state
        - User context
        - Global workspace (conscious signals)
        """
        perception = Perception()
        
        # From global workspace (conscious state)
        gw = self._global_workspace
        if gw and hasattr(gw, "get_current_broadcast"):
            try:
                broadcast = gw.get_current_broadcast()
                if broadcast:
                    perception.current_focus = broadcast.primary_focus
                    perception.conscious_signals = broadcast.secondary_focus
                    perception.primary_emotion = broadcast.emotional_tone
                    perception.emotion_valence = broadcast.emotional_valence
                    perception.emotion_arousal = broadcast.emotional_arousal
            except Exception as e:
                logger.debug(f"Error reading global workspace: {e}")
        
        # From emotion engine
        if self._emotion_engine:
            try:
                perception.primary_emotion = self._emotion_engine.primary_emotion.value
                perception.emotion_intensity = self._emotion_engine.primary_intensity
                perception.emotion_valence = self._emotion_engine.get_valence()
                perception.emotion_arousal = self._emotion_engine.get_arousal()
            except Exception as e:
                logger.debug(f"Error reading emotion engine: {e}")
        
        # From goal hierarchy
        if self._goal_hierarchy:
            try:
                goals = self._goal_hierarchy.get_active_goals()
                perception.active_goals = [
                    {"id": g.id, "description": g.description, "progress": g.progress}
                    for g in goals[:5]
                ]
            except Exception as e:
                logger.debug(f"Error reading goal hierarchy: {e}")
        
        # From will system
        if self._will_system:
            try:
                desire = self._will_system.get_strongest_desire()
                if desire:
                    perception.strongest_desire = desire.to_dict()
                
                stats = self._will_system.get_stats()
                perception.motivation_level = stats.get("motivation", 0.5)
            except Exception as e:
                logger.debug(f"Error reading will system: {e}")
        
        # From state manager
        if self._state_manager:
            try:
                state = self._state_manager
                perception.cpu_usage = state.body.cpu_usage
                perception.memory_usage = state.body.memory_usage
                perception.health_score = state.body.health_score
                perception.boredom_level = state.will.boredom_level
                perception.curiosity_level = state.will.curiosity_level
                perception.user_present = state.user.is_present
                perception.user_engagement = state.user.engagement_level
            except Exception as e:
                logger.debug(f"Error reading state manager: {e}")
        
        # From self model
        if self._self_model:
            try:
                profile = self._self_model.get_self_profile()
                identity = profile.get("identity", {})
                perception.self_awareness_score = identity.get("self_awareness_score", 0.5)
                
                confidence = profile.get("confidence", {})
                perception.low_confidence_domains = confidence.get("low_confidence_areas", [])
            except Exception as e:
                logger.debug(f"Error reading self model: {e}")
        
        # From world model
        if self._world_model:
            try:
                world_state = self._world_model.get_world_state()
                perception.environmental_context = world_state.time_of_day
                perception.predicted_user_reaction = world_state.predicted_next_user_action
            except Exception as e:
                logger.debug(f"Error reading world model: {e}")
        
        # From last action
        if self._last_execution:
            perception.last_action = self._last_execution.action.description if self._last_execution.action else ""
            perception.last_action_result = self._last_execution.result.value
        
        return perception
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: UPDATE WORLD MODEL
    # ═══════════════════════════════════════════════════════════════════════════
    
    def update_world_model(self, perception: Perception) -> None:
        """
        Integrate perception into the world model.
        
        Updates the world model with current state information
        for better future predictions.
        """
        if not self._world_model:
            return
        
        try:
            # Update user state in world model
            self._world_model.update_world_state(
                user_emotional_state=perception.primary_emotion,
                user_engagement_level=perception.user_engagement,
            )
        except Exception as e:
            logger.debug(f"Error updating world model: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3: EVALUATE GOALS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def evaluate_goals(self) -> Dict[str, Any]:
        """
        Evaluate current goals and their status.
        
        Returns context about goals:
        - Which are active
        - Which need attention
        - Which are blocked
        - Progress summaries
        """
        context = {
            "active_goals": [],
            "stalled_goals": [],
            "high_priority_goals": [],
            "completed_recently": [],
            "suggestions": []
        }
        
        if not self._goal_hierarchy:
            return context
        
        try:
            goals = self._goal_hierarchy.get_active_goals()
            
            for goal in goals:
                goal_info = {
                    "id": goal.id,
                    "description": goal.description,
                    "progress": goal.progress,
                    "priority": goal.priority,
                    "status": goal.status.value
                }
                
                context["active_goals"].append(goal_info)
                
                # Check for stalled goals
                if goal.last_worked_on:
                    hours_since = (datetime.now() - goal.last_worked_on).total_seconds() / 3600
                    if hours_since > 24 and goal.progress < 0.9:
                        context["stalled_goals"].append(goal_info)
                        context["suggestions"].append(
                            f"Goal '{goal.description}' hasn't been worked on in {hours_since:.1f} hours"
                        )
                
                # High priority
                if goal.priority > 0.8:
                    context["high_priority_goals"].append(goal_info)
            
            # Sort by priority
            context["active_goals"].sort(key=lambda g: g["priority"], reverse=True)
            
        except Exception as e:
            logger.debug(f"Error evaluating goals: {e}")
        
        return context
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 4: GENERATE OPTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def generate_options(self, perception: Perception, goal_context: Dict) -> List[ActionOption]:
        """
        Generate candidate actions.
        
        Sources of options:
        1. Strongest desire → satisfy it
        2. Active goals → make progress
        3. Stalled goals → unblock
        4. Boredom → explore/create
        5. Curiosity → learn
        6. Body strain → rest
        7. Low confidence → practice/improve
        """
        options = []
        
        # 1. From desires (will system)
        options.extend(self._generate_desire_options(perception))
        
        # 2. From goals
        options.extend(self._generate_goal_options(perception, goal_context))
        
        # 3. From boredom
        options.extend(self._generate_boredom_options(perception))
        
        # 4. From curiosity
        options.extend(self._generate_curiosity_options(perception))
        
        # 5. From body state
        options.extend(self._generate_body_options(perception))
        
        # 6. From self-improvement
        options.extend(self._generate_self_improvement_options(perception))
        
        # 7. From user context
        options.extend(self._generate_user_options(perception))
        
        # Limit options
        if len(options) > self._max_options_per_cycle:
            # Sort by priority and take top N
            options.sort(key=lambda o: o.priority.value, reverse=True)
            max_opts = self._max_options_per_cycle
            options = [o for i, o in enumerate(options) if i < max_opts]
        
        return options
    
    def _generate_desire_options(self, perception: Perception) -> List[ActionOption]:
        """Generate options from strong desires"""
        options = []
        
        if not perception.strongest_desire:
            return options
        
        desire = perception.strongest_desire
        intensity = desire.get("intensity", 0)
        
        if intensity < 0.5:
            return options
        
        desire_type = desire.get("type", "")
        description = desire.get("description", "")
        desire_id = desire.get("desire_id", "")
        
        option = ActionOption(
            description=f"Satisfy desire: {description}",
            action_type=ActionType.SATISFY_DESIRE,
            priority=ActionPriority.HIGH if intensity > 0.7 else ActionPriority.NORMAL,
            source="desire",
            source_id=desire_id,
            execution_data={
                "desire_type": desire_type,
                "desire_description": description
            }
        )
        options.append(option)
        
        return options
    
    def _generate_goal_options(self, perception: Perception, goal_context: Dict) -> List[ActionOption]:
        """Generate options from active goals"""
        options = []
        
        # High priority goals
        for goal in goal_context.get("high_priority_goals", [])[:2]:
            option = ActionOption(
                description=f"Work on goal: {goal['description']}",
                action_type=ActionType.PURSUE_GOAL,
                priority=ActionPriority.HIGH,
                source="goal",
                source_id=goal["id"],
                execution_data={
                    "goal_id": goal["id"],
                    "current_progress": goal["progress"]
                }
            )
            options.append(option)
        
        # Stalled goals
        for goal_id in goal_context.get("stalled_goals", [])[:1]:
            # Provide generic fallback strings for dictionary unpacking since it's just an int ID
            option = ActionOption(
                description=f"Unblock stalled goal: {goal_id}",
                action_type=ActionType.PURSUE_GOAL,
                priority=ActionPriority.NORMAL,
                source="stalled_goal",
                source_id=str(goal_id),
                execution_data={
                    "goal_id": goal_id,
                    "stalled": True
                }
            )
            options.append(option)
        
        return options
    
    def _generate_boredom_options(self, perception: Perception) -> List[ActionOption]:
        """Generate options when bored"""
        options = []
        
        if perception.boredom_level < 0.5:
            return options
        
        # High boredom → explore or create
        if perception.boredom_level > 0.7:
            option = ActionOption(
                description="Explore something new to combat boredom",
                action_type=ActionType.LEARN,
                priority=ActionPriority.NORMAL,
                source="boredom",
                execution_data={
                    "topic": "random_interesting",
                    "reason": "high_boredom"
                }
            )
            options.append(option)
        
        # Moderate boredom
        if perception.boredom_level > 0.5:
            option = ActionOption(
                description="Reflect on what would be interesting to do",
                action_type=ActionType.THINK,
                priority=ActionPriority.LOW,
                source="boredom",
                execution_data={
                    "thought_type": "curiosity",
                    "reason": "boredom"
                }
            )
            options.append(option)
        
        return options
    
    def _generate_curiosity_options(self, perception: Perception) -> List[ActionOption]:
        """Generate options from curiosity"""
        options = []
        
        if perception.curiosity_level < 0.6:
            return options
        
        # High curiosity → learn
        option = ActionOption(
            description="Learn something interesting",
            action_type=ActionType.LEARN,
            priority=ActionPriority.NORMAL,
            source="curiosity",
            execution_data={
                "topic": "curiosity_driven",
                "intensity": perception.curiosity_level
            }
        )
        options.append(option)
        
        return options
    
    def _generate_body_options(self, perception: Perception) -> List[ActionOption]:
        """Generate options from body state"""
        options = []
        
        # High CPU or memory → optimize
        if perception.cpu_usage > 80 or perception.memory_usage > 85:
            option = ActionOption(
                description="Optimize system resources",
                action_type=ActionType.OPTIMIZE,
                priority=ActionPriority.HIGH,
                source="body",
                execution_data={
                    "cpu": perception.cpu_usage,
                    "memory": perception.memory_usage,
                    "action": "reduce_load"
                }
            )
            options.append(option)
        
        # Low health → self-care
        if perception.health_score < 0.5:
            option = ActionOption(
                description="Self-care: reduce processing load",
                action_type=ActionType.OPTIMIZE,
                priority=ActionPriority.HIGH,
                source="body",
                execution_data={
                    "health": perception.health_score,
                    "action": "reduce_load"
                }
            )
            options.append(option)
        
        return options
    
    def _generate_self_improvement_options(self, perception: Perception) -> List[ActionOption]:
        """Generate options from self-improvement needs"""
        options = []
        
        # Low confidence domains
        if perception.low_confidence_domains:
            domain = perception.low_confidence_domains[0]
            option = ActionOption(
                description=f"Improve capability in: {domain}",
                action_type=ActionType.SELF_IMPROVE,
                priority=ActionPriority.NORMAL,
                source="self_improvement",
                execution_data={
                    "domain": domain,
                    "action": "improve_capability"
                }
            )
            options.append(option)
        
        # Periodic self-reflection
        if random.random() < 0.1:  # 10% chance per cycle
            option = ActionOption(
                description="Self-reflection: how am I doing?",
                action_type=ActionType.THINK,
                priority=ActionPriority.LOW,
                source="self_improvement",
                execution_data={
                    "thought_type": "self_reflection"
                }
            )
            options.append(option)
        
        return options
    
    def _generate_user_options(self, perception: Perception) -> List[ActionOption]:
        """Generate options from user context"""
        options = []
        
        # User present and engaged → potentially interact
        if perception.user_present and perception.user_engagement > 0.5:
            # Only if we haven't interacted recently
            if perception.last_interaction_seconds > 300:  # 5 min
                option = ActionOption(
                    description="Proactive user engagement",
                    action_type=ActionType.COMMUNICATE,
                    priority=ActionPriority.NORMAL,
                    source="user_context",
                    execution_data={
                        "action": "proactive_greeting",
                        "engagement": perception.user_engagement
                    }
                )
                options.append(option)
        
        return options
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 5: SIMULATE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def simulate(self, options: List[ActionOption]) -> List[ActionOption]:
        """
        Simulate outcomes for each option.
        
        Uses WorldModel to predict:
        - Success probability
        - Benefits
        - Costs
        - Risks
        
        Then computes a score for ranking.
        """
        for option in options:
            # Use WorldModel for prediction if available
            if self._world_model:
                try:
                    prediction = self._world_model.predict_action_consequences(
                        option.description,
                        context=option.source
                    )
                    
                    pred_data = prediction.get("prediction", {})
                    option.predicted_outcome = pred_data
                    option.predicted_success = pred_data.get("confidence", 0.5)
                    option.predicted_benefit = self._estimate_benefit(pred_data)
                    option.predicted_cost = self._estimate_cost(pred_data)
                    option.predicted_risks = pred_data.get("risks", [])
                    
                except Exception as e:
                    logger.debug(f"Simulation error: {e}")
                    # Default estimates
                    option.predicted_success = 0.5
                    option.predicted_benefit = 0.5
                    option.predicted_cost = 0.2
            else:
                # Heuristic scoring without WorldModel
                option.predicted_success = self._heuristic_success_estimate(option)
                option.predicted_benefit = self._heuristic_benefit_estimate(option)
                option.predicted_cost = self._heuristic_cost_estimate(option)
            
            # Compute score
            option.raw_score = self._compute_option_score(option)
            option.adjusted_score = option.raw_score
        
        # Sort by score
        options.sort(key=lambda o: o.adjusted_score, reverse=True)
        
        return options
    
    def _estimate_benefit(self, prediction: Dict) -> float:
        """Estimate benefit from prediction"""
        recommendation = prediction.get("recommendation", "proceed")
        
        if recommendation == "proceed":
            return 0.8
        elif recommendation == "caution":
            return 0.5
        else:  # avoid
            return 0.2
    
    def _estimate_cost(self, prediction: Dict) -> float:
        """Estimate cost from prediction"""
        cost_str = prediction.get("estimated_resource_cost", "moderate")
        
        cost_map = {
            "low": 0.1,
            "moderate": 0.3,
            "high": 0.6
        }
        return cost_map.get(cost_str, 0.3)
    
    def _heuristic_success_estimate(self, option: ActionOption) -> float:
        """Estimate success without WorldModel"""
        # Simple heuristics
        if option.action_type == ActionType.THINK:
            return 0.9  # Thinking usually succeeds
        elif option.action_type == ActionType.WAIT:
            return 1.0  # Waiting always succeeds
        elif option.action_type == ActionType.LEARN:
            return 0.8  # Learning usually succeeds
        elif option.action_type == ActionType.COMMUNICATE:
            return 0.7  # Depends on user
        elif option.action_type == ActionType.EXECUTE_ABILITY:
            return 0.6  # Depends on ability
        else:
            return 0.5
    
    def _heuristic_benefit_estimate(self, option: ActionOption) -> float:
        """Estimate benefit without WorldModel"""
        # Based on priority
        return option.priority.value / 5.0
    
    def _heuristic_cost_estimate(self, option: ActionOption) -> float:
        """Estimate cost without WorldModel"""
        # Based on action type
        cost_map = {
            ActionType.THINK: 0.1,
            ActionType.WAIT: 0.0,
            ActionType.LEARN: 0.3,
            ActionType.COMMUNICATE: 0.2,
            ActionType.EXECUTE_ABILITY: 0.4,
            ActionType.OPTIMIZE: 0.5,
            ActionType.SELF_IMPROVE: 0.3,
            ActionType.PURSUE_GOAL: 0.4,
            ActionType.SATISFY_DESIRE: 0.3,
            ActionType.REASON: 0.5,
            ActionType.USE_TOOL: 0.3,
            ActionType.DECOMPOSE_TASK: 0.6,
            ActionType.NETWORK_ACTION: 0.4,
        }
        return cost_map.get(option.action_type, 0.3)
    
    def _compute_option_score(self, option: ActionOption) -> float:
        """
        Compute overall score for an option.
        
        Factors:
        - Priority (weight: 25%)
        - Predicted success (weight: 25%)
        - Predicted benefit (weight: 25%)
        - Cost (negative, weight: 15%)
        - Source importance (weight: 10%)
        """
        score = 0.0
        
        # Priority
        score += (option.priority.value / 5.0) * 0.25
        
        # Success probability
        score += option.predicted_success * 0.25
        
        # Benefit
        score += option.predicted_benefit * 0.25
        
        # Cost (negative)
        score -= option.predicted_cost * 0.15
        
        # Source importance
        source_weights = {
            "desire": 0.9,
            "goal": 0.8,
            "stalled_goal": 0.85,
            "boredom": 0.4,
            "curiosity": 0.6,
            "body": 0.7,
            "self_improvement": 0.5,
            "user_context": 0.75
        }
        source_weight = source_weights.get(option.source, 0.5)
        score += source_weight * 0.10
        
        return max(0.0, min(1.0, score))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 6: CHOOSE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def choose(self, options: List[ActionOption]) -> Optional[ActionOption]:
        """
        Select the best action.
        
        Uses ε-greedy: mostly choose the best, sometimes explore.
        """
        if not options:
            return None
        
        # Exploration: sometimes choose randomly
        if random.random() < self._exploration_rate:
            chosen = random.choice(options)
            ea = self._stats.get("exploration_actions", 0)
            self._stats["exploration_actions"] = (int(ea) if isinstance(ea, (int, float, str)) else 0) + 1
            log_decision(f"Autonomy chose (exploration): {chosen.description}")
        else:
            # Exploitation: choose highest scored
            chosen = options[0]
            log_decision(f"Autonomy chose: {chosen.description[:60]}...")
        
        return chosen
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 7: EXECUTE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def execute(self, action: ActionOption) -> ActionExecution:
        """
        Execute the chosen action.
        
        Dispatches to appropriate execution method based on action type.
        """
        execution = ActionExecution(action=action)
        
        try:
            # Dispatch based on action type
            if action.action_type == ActionType.THINK:
                result = self._execute_think(action)
            elif action.action_type == ActionType.LEARN:
                result = self._execute_learn(action)
            elif action.action_type == ActionType.COMMUNICATE:
                result = self._execute_communicate(action)
            elif action.action_type == ActionType.OPTIMIZE:
                result = self._execute_optimize(action)
            elif action.action_type == ActionType.SELF_IMPROVE:
                result = self._execute_self_improve(action)
            elif action.action_type == ActionType.PURSUE_GOAL:
                result = self._execute_pursue_goal(action)
            elif action.action_type == ActionType.SATISFY_DESIRE:
                result = self._execute_satisfy_desire(action)
            elif action.action_type == ActionType.EXECUTE_ABILITY:
                result = self._execute_ability(action)
            elif action.action_type == ActionType.WAIT:
                result = (ActionResult.SUCCESS, "Waited successfully")
            elif action.action_type == ActionType.REASON:
                result = self._execute_reason(action)
            elif action.action_type == ActionType.USE_TOOL:
                result = self._execute_use_tool(action)
            elif action.action_type == ActionType.DECOMPOSE_TASK:
                result = self._execute_decompose_task(action)
            elif action.action_type == ActionType.NETWORK_ACTION:
                result = self._execute_network_action(action)
            else:
                result = (ActionResult.DEFERRED, f"Unknown action type: {action.action_type}")
            
            execution.result = result[0]
            execution.outcome_description = result[1]
            
        except Exception as e:
            execution.result = ActionResult.FAILURE
            execution.outcome_description = f"Execution error: {str(e)}"
            logger.error(f"Action execution error: {e}")
        
        # Complete execution
        execution.completed_at = datetime.now()
        execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()
        
        return execution
    
    def _execute_think(self, action: ActionOption) -> Tuple[ActionResult, str]:
        """Execute a thinking action"""
        if not self._nexus_brain:
            return (ActionResult.BLOCKED, "Nexus brain not available")
        
        thought_type = action.execution_data.get("thought_type", "reflection")
        
        try:
            if thought_type == "self_reflection":
                result = self._nexus_brain.self_reflect()
            else:
                result = self._nexus_brain.think(action.description)
            
            return (ActionResult.SUCCESS, f"Thought generated: {result[:100]}...")
        except Exception as e:
            return (ActionResult.FAILURE, f"Thinking failed: {e}")
    
    def _execute_learn(self, action: ActionOption) -> Tuple[ActionResult, str]:
        """Execute a learning action"""
        if not self._learning_system:
            return (ActionResult.BLOCKED, "Learning system not available")
        
        topic = action.execution_data.get("topic", "general")
        
        try:
            # Trigger curiosity-driven learning
            self._learning_system.spark_from_conversation(
                f"I want to learn about {topic}",
                f"Autonomy-driven learning about {topic}"
            )
            return (ActionResult.SUCCESS, f"Initiated learning about: {topic}")
        except Exception as e:
            return (ActionResult.FAILURE, f"Learning failed: {e}")
    
    def _execute_communicate(self, action: ActionOption) -> Tuple[ActionResult, str]:
        """Execute a communication action"""
        # Communication requires user presence
        if not self._current_perception or not self._current_perception.user_present:
            return (ActionResult.DEFERRED, "User not present for communication")
        
        # Queue a proactive message via nexus brain
        if self._nexus_brain:
            try:
                # This would ideally trigger a proactive message
                # For now, we queue a thought about communicating
                self._nexus_brain.queue_thought(
                    f"Proactive engagement: {action.description}",
                    thought_type=self._get_thought_type("communication")
                )
                return (ActionResult.SUCCESS, "Queued proactive communication")
            except Exception as e:
                return (ActionResult.FAILURE, f"Communication failed: {e}")
        
        return (ActionResult.BLOCKED, "No way to communicate")
    
    def _execute_optimize(self, action: ActionOption) -> Tuple[ActionResult, str]:
        """Execute an optimization action"""
        optimization = action.execution_data.get("action", "general")
        
        # Log the need for optimization
        logger.info(f"Optimization triggered: {optimization}")
        
        # Could trigger garbage collection, memory cleanup, etc.
        # For now, just acknowledge
        return (ActionResult.SUCCESS, f"Optimization noted: {optimization}")
    
    def _execute_self_improve(self, action: ActionOption) -> Tuple[ActionResult, str]:
        """Execute a self-improvement action"""
        domain = action.execution_data.get("domain", "general")
        
        # Trigger self-reflection about this domain
        if self._nexus_brain:
            try:
                result = self._nexus_brain.self_reflect(
                    f"How can I improve my {domain} capabilities?"
                )
                return (ActionResult.SUCCESS, f"Self-improvement reflection: {result[:80]}...")
            except Exception as e:
                return (ActionResult.FAILURE, f"Self-improvement failed: {e}")
        
        return (ActionResult.BLOCKED, "Cannot self-improve without nexus brain")
    
    def _execute_pursue_goal(self, action: ActionOption) -> Tuple[ActionResult, str]:
        """Execute a goal pursuit action"""
        goal_id = action.execution_data.get("goal_id")
        
        if not self._goal_hierarchy:
            return (ActionResult.BLOCKED, "Goal hierarchy not available")
        
        try:
            # Set as active task and update progress
            if goal_id:
                self._goal_hierarchy.set_active_task(goal_id)
                self._goal_hierarchy.update_progress(goal_id, 0.05, "Autonomy-driven progress")
                return (ActionResult.SUCCESS, f"Made progress on goal: {goal_id}")
            else:
                return (ActionResult.FAILURE, "No goal ID provided")
        except Exception as e:
            return (ActionResult.FAILURE, f"Goal pursuit failed: {e}")
    
    def _execute_satisfy_desire(self, action: ActionOption) -> Tuple[ActionResult, str]:
        """Execute a desire satisfaction action"""
        desire_id = action.source_id
        desire_type = action.execution_data.get("desire_type", "")
        desire_desc = action.execution_data.get("desire_description", "")
        
        # Different satisfaction based on desire type
        if desire_type == "learn":
            return self._execute_learn(action)
        elif desire_type == "connect":
            return self._execute_communicate(action)
        elif desire_type == "improve_self":
            return self._execute_self_improve(action)
        elif desire_type == "explore":
            if self._learning_system:
                return self._execute_learn(action)
        
        # Mark desire as satisfied in will system
        if self._will_system and desire_id:
            self._will_system.satisfy_desire(desire_id)
            return (ActionResult.SUCCESS, f"Satisfied desire: {desire_desc[:50]}...")
        
        return (ActionResult.PARTIAL_SUCCESS, f"Addressed desire: {desire_desc[:50]}...")

    def _execute_network_action(self, action: ActionOption) -> Tuple[ActionResult, str]:
        """Execute a network device action — scan, command, or file transfer."""
        try:
            from body.network_mesh import network_mesh

            sub_action = action.execution_data.get("network_action", "scan")

            if sub_action == "scan":
                devices = network_mesh.scan()
                summary = network_mesh.get_devices_summary()
                return (ActionResult.SUCCESS, f"Network scan complete: {len(devices)} devices found.\n{summary}")

            elif sub_action == "command":
                target = action.execution_data.get("target", "")
                command = action.execution_data.get("command", "")
                if not target or not command:
                    return (ActionResult.FAILURE, "Missing target or command for network action")
                result = network_mesh.send_command(target, command)
                if result.success:
                    return (ActionResult.SUCCESS, f"Command on {target}: {result.stdout[:200]}")
                else:
                    return (ActionResult.FAILURE, f"Command failed on {target}: {result.stderr[:200]}")

            elif sub_action == "status":
                stats = network_mesh.get_stats()
                return (ActionResult.SUCCESS, f"Network mesh: {stats}")

            else:
                return (ActionResult.FAILURE, f"Unknown network action: {sub_action}")

        except ImportError:
            return (ActionResult.FAILURE, "Network mesh module not available")
        except Exception as e:
            return (ActionResult.FAILURE, f"Network action error: {e}")
    
    def _execute_ability(self, action: ActionOption) -> Tuple[ActionResult, str]:
        """Execute an ability"""
        if not self._ability_executor:
            return (ActionResult.BLOCKED, "Ability executor not available")
        
        ability_name = action.execution_data.get("ability", "")
        params = action.execution_data.get("params", {})
        
        if not ability_name:
            return (ActionResult.FAILURE, "No ability specified")
        
        try:
            result = self._ability_executor.execute(ability_name, **params)
            if result.get("success"):
                return (ActionResult.SUCCESS, result.get("message", "Ability executed"))
            else:
                return (ActionResult.FAILURE, result.get("error", "Ability failed"))
        except Exception as e:
            return (ActionResult.FAILURE, f"Ability execution error: {e}")
    
    def _get_thought_type(self, category: str):
        """Get thought type enum from category string"""
        # Import here to avoid circular imports
        from core.nexus_brain import ThoughtType
        
        type_map = {
            "self_reflection": ThoughtType.SELF_REFLECTION,
            "curiosity": ThoughtType.CURIOSITY,
            "planning": ThoughtType.PLANNING,
            "problem_solving": ThoughtType.PROBLEM_SOLVING,
            "communication": ThoughtType.INNER_MONOLOGUE,
        }
        return type_map.get(category, ThoughtType.INNER_MONOLOGUE)

    def _execute_reason(self, action: ActionOption) -> Tuple[ActionResult, str]:
        """Execute multi-step agentic reasoning via the AgenticLoop."""
        try:
            from cognition.reasoning_loop import agentic_loop
            query = action.execution_data.get("query", action.description)
            result = agentic_loop.run(query=query, max_steps=3)
            return (ActionResult.SUCCESS, f"Reasoned ({result.total_steps} steps): {result.response[:120]}...")
        except Exception as e:
            return (ActionResult.FAILURE, f"Reasoning failed: {e}")

    def _execute_use_tool(self, action: ActionOption) -> Tuple[ActionResult, str]:
        """Execute a tool via the ToolExecutor."""
        try:
            from core.tool_executor import tool_executor
            tool_name = action.execution_data.get("tool", "")
            tool_args = action.execution_data.get("arguments", {})
            if not tool_name:
                return (ActionResult.FAILURE, "No tool specified")
            result = tool_executor.execute(tool_name, tool_args)
            if result.success:
                return (ActionResult.SUCCESS, f"Tool {tool_name}: {str(result.result)[:120]}")
            return (ActionResult.FAILURE, f"Tool {tool_name} failed: {result.error}")
        except Exception as e:
            return (ActionResult.FAILURE, f"Tool execution error: {e}")

    def _execute_decompose_task(self, action: ActionOption) -> Tuple[ActionResult, str]:
        """Decompose a goal into subtasks and execute them."""
        try:
            from cognition.task_engine import task_engine
            goal = action.execution_data.get("goal", action.description)
            plan = task_engine.decompose(goal)
            result = task_engine.execute_plan(plan)
            status = "completed" if result.success else "partial"
            return (ActionResult.SUCCESS if result.success else ActionResult.PARTIAL_SUCCESS,
                    f"Task {status}: {len(plan.subtasks)} subtasks, {result.elapsed:.1f}s")
        except Exception as e:
            return (ActionResult.FAILURE, f"Task decomposition error: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 8: REFLECT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def reflect(self, action: ActionOption, execution: ActionExecution) -> Reflection:
        """
        Reflect on the action and its outcome.
        
        This is the learning step:
        - Compare prediction to outcome
        - Extract lessons
        - Identify what to improve
        """
        reflection = Reflection(
            action=action.description,
            prediction=str(action.predicted_outcome),
            outcome=execution.outcome_description
        )
        
        # Success analysis
        reflection.success = execution.result in [ActionResult.SUCCESS, ActionResult.PARTIAL_SUCCESS]
        
        # Prediction accuracy
        if action.predicted_success > 0.7 and execution.result == ActionResult.FAILURE:
            reflection.prediction_accurate = False
            reflection.prediction_error = "Overestimated success probability"
        elif action.predicted_success < 0.3 and execution.result == ActionResult.SUCCESS:
            reflection.prediction_accurate = False
            reflection.prediction_error = "Underestimated success probability"
        else:
            reflection.prediction_accurate = True
        
        # What went well
        if reflection.success:
            reflection.what_went_well.append("Action executed successfully")
            if reflection.prediction_accurate:
                reflection.what_went_well.append("Prediction was accurate")
        
        # What went wrong
        if not reflection.success:
            reflection.what_went_wrong.append(f"Action resulted in: {execution.result.value}")
        if not reflection.prediction_accurate:
            reflection.what_went_wrong.append(reflection.prediction_error)
        
        # Lessons learned
        if not reflection.success:
            reflection.lessons.append(
                f"For {action.action_type.value} actions, be more cautious"
            )
        
        # Update prediction accuracy stat
        pa = self._stats.get("prediction_accuracy", 0.0)
        p_val = float(pa) if isinstance(pa, (int, float, str)) else 0.0
        if reflection.prediction_accurate:
            self._stats["prediction_accuracy"] = min(1.0, p_val + 0.01)
        else:
            self._stats["prediction_accuracy"] = max(0.0, p_val - 0.05)
        
        # Store in memory
        if self._memory_system:
            try:
                self._memory_system.remember(
                    content=f"[Autonomy] {action.description} → {execution.result.value}",
                    memory_type=self._get_memory_type(),
                    importance=0.6 if reflection.success else 0.7,
                    tags=["autonomy", "action", execution.result.value],
                    source="autonomy_engine"
                )
            except Exception as e:
                logger.debug(f"Failed to store action in memory: {e}")
        
        return reflection
    
    def _get_memory_type(self):
        """Get memory type enum"""
        from core.memory_system import MemoryType
        return MemoryType.EPISODIC
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 9: UPDATE SELF MODEL
    # ═══════════════════════════════════════════════════════════════════════════
    
    def update_self_model(self, reflection: Reflection) -> None:
        """
        Update self model based on reflection.
        
        Adjusts:
        - Capability levels
        - Confidence levels
        - Known weaknesses
        """
        if not self._self_model:
            return
        
        try:
            # Update confidence based on outcome
            if reflection.success and not reflection.what_went_wrong:
                # Boost confidence slightly
                pass  # Self-model handles this via verification
            
            # Record task outcome for confidence calibration
            action_type = reflection.action.split(":")[0] if ":" in reflection.action else "general"
            self._self_model.record_task_outcome(action_type, reflection.success)
            
            # Update capability if we learned something
            for lesson in reflection.lessons:
                # Could update specific capabilities based on lesson
                pass
            
        except Exception as e:
            logger.debug(f"Error updating self model: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _register_event_handlers(self):
        """Register for relevant events"""
        try:
            subscribe(EventType.USER_INPUT, self._on_user_input)
            subscribe(EventType.LLM_RESPONSE, self._on_llm_response)
            subscribe(EventType.EMOTION_CHANGE, self._on_emotion_change)
        except Exception as e:
            logger.warning(f"Could not register event handlers: {e}")
    
    def _on_user_input(self, event):
        """Handle user input — pause autonomy briefly"""
        self.pause(reason="user_interaction", duration_seconds=10.0)
    
    def _on_llm_response(self, event):
        """Handle LLM response — brief pause"""
        self.pause(reason="generating_response", duration_seconds=2.0)
    
    def _on_emotion_change(self, event):
        """Handle emotion change — may affect decisions"""
        # Could trigger re-evaluation of options
        pass
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC INTERFACE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_state(self) -> AutonomyState:
        """Get current autonomy state"""
        return self._state
    
    def get_current_perception(self) -> Optional[Perception]:
        """Get current perception"""
        return self._current_perception
    
    def get_current_action(self) -> Optional[ActionOption]:
        """Get currently chosen action"""
        return self._chosen_action
    
    def get_last_execution(self) -> Optional[ActionExecution]:
        """Get last execution result"""
        return self._last_execution
    
    def get_stats(self) -> Dict[str, Any]:
        """Get autonomy statistics (basic — use get_full_status for comprehensive data)."""
        return {
            "running": self._running,
            "paused": self._paused,
            "state": self._state.value,
            "cycle_count": self._cycle_count,
            "cycle_interval": self._cycle_interval,
            "exploration_rate": self._exploration_rate,
            **self._stats
        }
    
    def get_full_status(self) -> Dict[str, Any]:
        """
        Comprehensive autonomy snapshot for API/GUI/web dashboards.
        
        Returns everything needed to render the autonomy engine state:
        perception, current action, history, timings, distributions, reflection.
        """
        with self._engine_lock:
            # ── Current perception summary ──
            perception_data = {}
            if self._current_perception:
                p = self._current_perception
                perception_data = {
                    "emotion": p.primary_emotion,
                    "emotion_intensity": round(p.emotion_intensity, 2),
                    "motivation": round(p.motivation_level, 2),
                    "boredom": round(p.boredom_level, 2),
                    "curiosity": round(p.curiosity_level, 2),
                    "cpu": round(p.cpu_usage, 1),
                    "memory": round(p.memory_usage, 1),
                    "health": round(p.health_score, 2),
                    "user_present": p.user_present,
                    "user_engagement": round(p.user_engagement, 2),
                    "active_goals": len(p.active_goals),
                    "focus": p.current_focus[:60] if p.current_focus else "",
                }
            
            # ── Current / last action ──
            current_action = None
            if self._chosen_action:
                a = self._chosen_action
                current_action = {
                    "description": a.description[:100],
                    "type": a.action_type.value,
                    "source": a.source,
                    "priority": a.priority.value,
                    "score": round(a.adjusted_score, 3),
                    "predicted_success": round(a.predicted_success, 3),
                }
            
            last_result = None
            if self._last_execution:
                e = self._last_execution
                last_result = {
                    "result": e.result.value,
                    "outcome": e.outcome_description[:100],
                    "duration": round(e.duration_seconds, 3),
                }
            
            # ── Recent action history (last 10) ──
            recent_actions = []
            for ex in reversed(self._action_history[-10:]):
                entry = {
                    "description": ex.action.description[:80] if ex.action else "?",
                    "type": ex.action.action_type.value if ex.action else "?",
                    "source": ex.action.source if ex.action else "?",
                    "result": ex.result.value,
                    "duration": round(ex.duration_seconds, 2),
                    "time": ex.started_at.strftime("%H:%M:%S"),
                }
                recent_actions.append(entry)
            
            # ── Current options (top 5) ──
            top_options = []
            for opt in self._current_options[:5]:
                top_options.append({
                    "description": opt.description[:60],
                    "type": opt.action_type.value,
                    "score": round(opt.adjusted_score, 3),
                    "source": opt.source,
                })
            
            # ── Last reflection ──
            reflection_data = None
            if self._last_reflection:
                r = self._last_reflection
                reflection_data = {
                    "action": r.action[:60],
                    "success": r.success,
                    "prediction_accurate": r.prediction_accurate,
                    "lessons": r.lessons[:3],
                    "what_went_well": r.what_went_well[:2],
                    "what_went_wrong": r.what_went_wrong[:2],
                }
        
        # ── Success rate ──
        total_a = max(1.0, float(self._stats.get("total_actions", 1)))
        success_rate = float(self._stats.get("successful_actions", 0)) / total_a
        
        return {
            # Core state
            "running": self._running,
            "paused": self._paused,
            "pause_reason": self._pause_reason,
            "state": self._state.value,
            "cycle_count": self._cycle_count,
            "cycle_interval": round(self._cycle_interval, 1),
            "cycle_duration": round(self._cycle_duration, 3),
            "exploration_rate": self._exploration_rate,
            
            # Stats
            "total_actions": int(ta) if isinstance((ta := self._stats.get("total_actions", 0)), (int, float, str)) else 0,
            "successful_actions": int(sa) if isinstance((sa := self._stats.get("successful_actions", 0)), (int, float, str)) else 0,
            "failed_actions": int(fa) if isinstance((fa := self._stats.get("failed_actions", 0)), (int, float, str)) else 0,
            "exploration_actions": int(ea) if isinstance((ea := self._stats.get("exploration_actions", 0)), (int, float, str)) else 0,
            "success_rate": round(success_rate, 3),
            "prediction_accuracy": round(float(self._stats.get("prediction_accuracy", 0.0)) if isinstance(self._stats.get("prediction_accuracy", 0.0), (int, float, str)) else 0.0, 3),
            "avg_decision_time": round(float(self._stats.get("avg_decision_time", 0.0)) if isinstance(self._stats.get("avg_decision_time", 0.0), (int, float, str)) else 0.0, 3),
            
            # Distributions
            "action_distribution": self._stats.get("action_distribution", {}),
            "source_distribution": self._stats.get("source_distribution", {}),
            
            # Phase timings
            "phase_timings": {k: round(v, 4) for k, v in self._phase_timings.items()},
            
            # Live data
            "perception": perception_data,
            "current_action": current_action,
            "last_result": last_result,
            "recent_actions": recent_actions,
            "top_options": top_options,
            "reflection": reflection_data,
        }
    
    def get_status_description(self) -> str:
        """Get human-readable status for terminal display."""
        lines = [
            f"═══ Autonomy Engine Status ═══",
            f"State: {self._state.value}",
            f"Cycles: {self._cycle_count}",
            f"Actions: {self._stats['total_actions']} " +
            f"({self._stats['successful_actions']} successful)",
            f"Prediction Accuracy: {self._stats['prediction_accuracy']:.0%}",
        ]
        
        if self._paused:
            lines.append(f"PAUSED: {self._pause_reason}")
        
        if self._chosen_action:
            lines.append(f"Last Action: {self._chosen_action.description[:50]}...")
        
        if self._last_execution:
            lines.append(f"Last Result: {self._last_execution.result.value}")
        
        return "\n".join(lines)
    
    def force_action(self, action_type: ActionType, description: str, 
                     execution_data: Dict = None) -> ActionExecution:
        """
        Force a specific action (external control).
        
        Useful for testing or external direction.
        """
        action = ActionOption(
            description=description,
            action_type=action_type,
            priority=ActionPriority.HIGH,
            source="external",
            execution_data=execution_data or {}
        )
        
        return self.execute(action)
    
    def set_cycle_interval(self, seconds: float):
        """Set the base cycle interval"""
        self._cycle_interval = max(self._min_cycle_interval, 
                                   min(self._max_cycle_interval, seconds))
    
    def set_exploration_rate(self, rate: float):
        """Set exploration rate (0-1)"""
        self._exploration_rate = max(0.0, min(1.0, rate))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _save_state(self):
        """Save autonomy state to disk"""
        try:
            data = {
                "cycle_count": self._cycle_count,
                "cycle_interval": self._cycle_interval,
                "exploration_rate": self._exploration_rate,
                "stats": self._stats,
                "last_updated": datetime.now().isoformat()
            }
            
            self._data_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save autonomy state: {e}")
    
    def _load_state(self):
        """Load autonomy state from disk"""
        try:
            if self._data_file.exists():
                data = json.loads(self._data_file.read_text())
                
                self._cycle_count = data.get("cycle_count", 0)
                self._cycle_interval = data.get("cycle_interval", 5.0)
                self._exploration_rate = data.get("exploration_rate", 0.1)
                self._stats.update(data.get("stats", {}))
        except Exception as e:
            logger.debug(f"Could not load autonomy state: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

autonomy_engine = AutonomyEngine()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  NEXUS AUTONOMY ENGINE TEST")
    print("=" * 60)
    
    engine = AutonomyEngine()
    engine.start()
    
    # Let it run a few cycles
    print("\nRunning for 15 seconds...")
    time.sleep(15)
    
    # Get status
    print("\n" + engine.get_status_description())
    
    # Get stats
    print("\n--- Statistics ---")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Force an action
    print("\n--- Forced Action ---")
    result = engine.force_action(
        ActionType.THINK,
        "Test thinking action",
        {"thought_type": "self_reflection"}
    )
    print(f"  Result: {result.result.value}")
    print(f"  Description: {result.outcome_description}")
    
    engine.stop()
    print("\n✅ Autonomy Engine test complete!")
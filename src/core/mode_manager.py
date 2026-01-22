"""
Mode management system for ASEA-X
Handles system mode transitions and validation
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

from pydantic import BaseModel, validator


class SystemMode(str, Enum):
    """System operational modes"""
    CHAT = "chat"
    PLANNER = "planner"
    DEV = "dev"
    DEBUG = "debug"
    LINT = "lint"
    REVIEW = "review"
    SAFETY_OVERRIDE = "safety_override"


@dataclass
class ModeTransition:
    """Represents a mode transition"""
    from_mode: SystemMode
    to_mode: SystemMode
    reason: str
    timestamp: float
    user_initiated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModeRequirements(BaseModel):
    """Requirements for each mode"""
    requires_plan: bool = False
    requires_code: bool = False
    requires_lint: bool = False
    allows_execution: bool = False
    allows_install: bool = False
    safety_checks: bool = True
    max_runtime: Optional[int] = None


class ModeManager:
    """Manages system modes and transitions"""
    
    # Mode requirements mapping
    MODE_REQUIREMENTS = {
        SystemMode.CHAT: ModeRequirements(
            requires_plan=False,
            requires_code=False,
            requires_lint=False,
            allows_execution=False,
            allows_install=False,
            safety_checks=True
        ),
        SystemMode.PLANNER: ModeRequirements(
            requires_plan=False,
            requires_code=False,
            requires_lint=False,
            allows_execution=False,
            allows_install=False,
            safety_checks=True
        ),
        SystemMode.DEV: ModeRequirements(
            requires_plan=True,
            requires_code=True,
            requires_lint=False,
            allows_execution=True,
            allows_install=True,
            safety_checks=True,
            max_runtime=30
        ),
        SystemMode.DEBUG: ModeRequirements(
            requires_plan=True,
            requires_code=True,
            requires_lint=False,
            allows_execution=True,
            allows_install=False,
            safety_checks=True,
            max_runtime=60
        ),
        SystemMode.LINT: ModeRequirements(
            requires_plan=False,
            requires_code=True,
            requires_lint=True,
            allows_execution=False,
            allows_install=False,
            safety_checks=True
        ),
        SystemMode.SAFETY_OVERRIDE: ModeRequirements(
            requires_plan=False,
            requires_code=False,
            requires_lint=False,
            allows_execution=True,
            allows_install=True,
            safety_checks=False,
            max_runtime=None
        )
    }
    
    # Valid mode transitions
    VALID_TRANSITIONS = {
        SystemMode.CHAT: {
            SystemMode.PLANNER,
            SystemMode.DEV,
            SystemMode.LINT,
            SystemMode.SAFETY_OVERRIDE
        },
        SystemMode.PLANNER: {
            SystemMode.CHAT,
            SystemMode.DEV,
            SystemMode.LINT
        },
        SystemMode.DEV: {
            SystemMode.CHAT,
            SystemMode.DEBUG,
            SystemMode.LINT
        },
        SystemMode.DEBUG: {
            SystemMode.CHAT,
            SystemMode.DEV,
            SystemMode.LINT
        },
        SystemMode.LINT: {
            SystemMode.CHAT,
            SystemMode.DEV
        },
        SystemMode.SAFETY_OVERRIDE: {
            SystemMode.CHAT
        }
    }
    
    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.current_mode = SystemMode.CHAT
        self.transition_history: list[ModeTransition] = []
        self.logger = logging.getLogger(__name__)
    
    def can_transition(self, to_mode: SystemMode, 
                      context: Optional[Dict[str, Any]] = None) -> tuple[bool, str]:
        """
        Check if transition to target mode is valid
        
        Returns:
            tuple[bool, str]: (can_transition, reason)
        """
        # Check basic transition validity
        if to_mode not in self.VALID_TRANSITIONS.get(self.current_mode, set()):
            return False, f"Invalid transition from {self.current_mode} to {to_mode}"
        
        # Get requirements for target mode
        requirements = self.MODE_REQUIREMENTS.get(to_mode)
        if not requirements:
            return False, f"No requirements defined for mode {to_mode}"
        
        # Check state requirements
        state = self.state_manager.get_state()
        
        if requirements.requires_plan and not state.task_history:
            return False, "Mode requires existing plan/task history"
        
        if requirements.requires_code and not state.file_context:
            return False, "Mode requires loaded code files"
        
        # Safety checks
        if requirements.safety_checks and to_mode == SystemMode.SAFETY_OVERRIDE:
            return False, "Safety override requires explicit user command"
        
        # All checks passed
        return True, ""
    
    def transition(self, to_mode: SystemMode, reason: str, 
                  user_initiated: bool = False,
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Transition to a new mode
        
        Returns:
            bool: True if transition was successful
        """
        can_transition, error = self.can_transition(to_mode, metadata)
        
        if not can_transition:
            self.logger.error(f"Cannot transition to {to_mode}: {error}")
            return False
        
        # Create transition record
        transition = ModeTransition(
            from_mode=self.current_mode,
            to_mode=to_mode,
            reason=reason,
            timestamp=time.time(),
            user_initiated=user_initiated,
            metadata=metadata or {}
        )
        
        # Update state
        self.transition_history.append(transition)
        self.current_mode = to_mode
        
        # Update global state
        self.state_manager.update_state(current_mode=to_mode.value)
        
        # Log the transition
        self.state_manager.log_action(
            agent="mode_manager",
            action="mode_transition",
            details={
                "from": transition.from_mode.value,
                "to": transition.to_mode.value,
                "reason": reason,
                "user_initiated": user_initiated
            }
        )
        
        self.logger.info(f"Transitioned from {transition.from_mode} to {to_mode}: {reason}")
        return True
    
    def auto_transition(self, context: Dict[str, Any]) -> bool:
        """
        Automatically transition based on context analysis
        
        Returns:
            bool: True if transition occurred
        """
        current_state = self.state_manager.get_state()
        
        # Check for error conditions first
        if current_state.last_error:
            if self.current_mode != SystemMode.DEBUG:
                return self.transition(
                    SystemMode.DEBUG,
                    "Automatic transition to debug due to error",
                    user_initiated=False,
                    metadata={"error": current_state.last_error}
                )
        
        # Check for completion of planning
        if self.current_mode == SystemMode.PLANNER:
            pending_tasks = [
                task for task in self.state_manager._tasks.values()
                if task.status in ["pending", "executing"]
            ]
            if not pending_tasks and self.state_manager._tasks:
                return self.transition(
                    SystemMode.DEV,
                    "Planning complete, transitioning to development",
                    user_initiated=False
                )
        
        # Check for code that needs linting
        if (self.current_mode == SystemMode.DEV and 
            "modified_files" in context and 
            context["modified_files"]):
            return self.transition(
                SystemMode.LINT,
                "Code modifications detected, transitioning to lint",
                user_initiated=False,
                metadata={"files": context["modified_files"]}
            )
        
        return False
    
    def get_mode_requirements(self, mode: SystemMode) -> ModeRequirements:
        """Get requirements for a specific mode"""
        return self.MODE_REQUIREMENTS.get(mode, ModeRequirements())
    
    def get_transition_history(self) -> list[ModeTransition]:
        """Get transition history"""
        return self.transition_history.copy()
    
    def suggest_next_mode(self, context: Dict[str, Any]) -> Optional[SystemMode]:
        """
        Suggest next mode based on current context
        
        Returns:
            Optional[SystemMode]: Suggested mode or None
        """
        current_state = self.state_manager.get_state()
        
        # If there's an error, suggest debug
        if current_state.last_error:
            return SystemMode.DEBUG
        
        # If in chat mode with task description, suggest planner
        if (self.current_mode == SystemMode.CHAT and 
            "task_description" in context and 
            context["task_description"]):
            return SystemMode.PLANNER
        
        # If planning is done and tasks exist, suggest dev
        if self.current_mode == SystemMode.PLANNER:
            has_pending_tasks = any(
                task.status == "pending" 
                for task in self.state_manager._tasks.values()
            )
            if has_pending_tasks:
                return SystemMode.DEV
        
        # If development completed, suggest lint
        if (self.current_mode == SystemMode.DEV and 
            "code_written" in context and 
            context["code_written"]):
            return SystemMode.LINT
        
        return None


# Import time at the end to avoid circular dependency
import time

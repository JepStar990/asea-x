"""
Global state management for ASEA-X
Maintains system-wide state and provides thread-safe access
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict
from dataclasses import dataclass, field, asdict
from enum import Enum
import time

from pydantic import BaseModel, Field
import redis


class TaskStatus(str, Enum):
    """Status of a task in the system"""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    LINTING = "linting"
    DEBUGGING = "debugging"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class SystemState(BaseModel):
    """Global system state"""
    current_mode: str = "chat"
    current_task: Optional[str] = None
    task_history: list[Dict[str, Any]] = Field(default_factory=list)
    active_agents: set[str] = Field(default_factory=set)
    safety_enabled: bool = True
    git_branch: Optional[str] = None
    workdir: Path = Field(default=Path("./workdir"))
    dependencies_installed: set[str] = Field(default_factory=set)
    file_context: Dict[str, str] = Field(default_factory=dict)
    last_error: Optional[str] = None
    execution_count: int = 0
    start_time: float = Field(default_factory=time.time)
    user_override: bool = False


class Task(BaseModel):
    """Represents a single task in the system"""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    assigned_to: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StateManager:
    """Thread-safe state manager with persistence"""
    
    def __init__(self, workdir: Path = Path("./workdir")):
        self.workdir = workdir
        self.workdir.mkdir(exist_ok=True)
        
        self.state_file = workdir / "system_state.json"
        self.lock = threading.RLock()
        
        # Initialize state
        self._state = SystemState(workdir=workdir)
        self._tasks: Dict[str, Task] = {}
        self._task_queue: list[str] = []
        
        # Try to load Redis for distributed mode
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(
                host='localhost', 
                port=6379, 
                decode_responses=True
            )
            self.redis_client.ping()
        except:
            self.redis_client = None
        
        self.load_state()
    
    def save_state(self) -> None:
        """Save state to disk"""
        with self.lock:
            state_data = self._state.dict()
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            # Also save to Redis if available
            if self.redis_client:
                self.redis_client.set(
                    "aseax:system_state",
                    json.dumps(state_data, default=str)
                )
    
    def load_state(self) -> None:
        """Load state from disk"""
        with self.lock:
            if self.state_file.exists():
                try:
                    with open(self.state_file, 'r') as f:
                        state_data = json.load(f)
                    
                    # Convert path strings back to Path objects
                    if 'workdir' in state_data:
                        state_data['workdir'] = Path(state_data['workdir'])
                    
                    self._state = SystemState(**state_data)
                except Exception as e:
                    print(f"Failed to load state: {e}")
    
    def update_state(self, **kwargs) -> None:
        """Update state with thread safety"""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
                else:
                    raise AttributeError(f"State has no attribute {key}")
            self.save_state()
    
    def get_state(self) -> SystemState:
        """Get current state"""
        with self.lock:
            return self._state.copy()
    
    def create_task(self, description: str, metadata: Optional[Dict] = None) -> str:
        """Create a new task"""
        with self.lock:
            task_id = f"task_{len(self._tasks) + 1:06d}"
            task = Task(
                id=task_id,
                description=description,
                metadata=metadata or {}
            )
            self._tasks[task_id] = task
            self._task_queue.append(task_id)
            self.save_state()
            return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        with self.lock:
            return self._tasks.get(task_id)
    
    def update_task(self, task_id: str, **kwargs) -> bool:
        """Update task status"""
        with self.lock:
            if task_id not in self._tasks:
                return False
            
            task = self._tasks[task_id]
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)
                else:
                    task.metadata[key] = value
            
            self.save_state()
            return True
    
    def get_next_task(self) -> Optional[Task]:
        """Get next pending task"""
        with self.lock:
            for task_id in self._task_queue:
                task = self._tasks[task_id]
                if task.status == TaskStatus.PENDING:
                    return task
            return None
    
    def get_running_tasks(self) -> list[Task]:
        """Get all running tasks"""
        with self.lock:
            return [
                task for task in self._tasks.values()
                if task.status in [TaskStatus.EXECUTING, TaskStatus.PLANNING]
            ]
    
    def add_file_context(self, file_path: str, content: str) -> None:
        """Add file to context"""
        with self.lock:
            self._state.file_context[file_path] = content
            self.save_state()
    
    def get_file_context(self, file_path: str) -> Optional[str]:
        """Get file from context"""
        with self.lock:
            return self._state.file_context.get(file_path)
    
    def clear_context(self) -> None:
        """Clear all file context"""
        with self.lock:
            self._state.file_context.clear()
            self.save_state()
    
    def log_action(self, agent: str, action: str, details: Dict[str, Any]) -> None:
        """Log agent action"""
        with self.lock:
            log_entry = {
                "timestamp": time.time(),
                "agent": agent,
                "action": action,
                "details": details,
                "mode": self._state.current_mode
            }
            self._state.task_history.append(log_entry)
            
            # Append to actions log file
            log_file = self.workdir / ".agent.actions.log"
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
            
            self.save_state()
    
    def reset(self) -> None:
        """Reset system state"""
        with self.lock:
            self._state = SystemState(workdir=self.workdir)
            self._tasks.clear()
            self._task_queue.clear()
            self.save_state()

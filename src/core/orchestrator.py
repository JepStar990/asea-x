"""
Orchestrator Agent - The brain of ASEA-X
Coordinates all agents and manages system flow
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

from pydantic import BaseModel, Field, validator

from src.agents.base_agent import BaseAgent, AgentCapability, AgentContext, AgentResponse
from src.agents.planner import PlannerAgent
from src.agents.developer import DeveloperAgent
from src.agents.linter import LinterAgent
from src.agents.debugger import DebuggerAgent
from src.core.mode_manager import SystemMode
from src.core.safety_system import SafetySystem
from src.llm.deepseek_client import DeepSeekClient


class OrchestratorDecision(BaseModel):
    """Decision made by orchestrator"""
    action: str
    target_agent: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_confirmation: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskProgress(BaseModel):
    """Task progress tracking"""
    task_id: str
    phase: str
    status: str
    progress: float = Field(ge=0.0, le=1.0)
    estimated_time_remaining: Optional[float] = None
    last_update: datetime = Field(default_factory=datetime.now)


class Orchestrator:
    """
    Orchestrator Agent - Coordinates all ASEA-X agents
    
    Responsibilities:
    1. Interpret user intent
    2. Route to appropriate agents
    3. Maintain system flow
    4. Make autonomous decisions
    5. Handle errors and escalations
    """
    
    def __init__(
        self,
        state_manager,
        mode_manager,
        safety_system: SafetySystem,
        llm_client: DeepSeekClient
    ):
        self.state_manager = state_manager
        self.mode_manager = mode_manager
        self.safety_system = safety_system
        self.llm_client = llm_client
        
        self.logger = logging.getLogger(__name__)
        self.agents: Dict[str, BaseAgent] = {}
        self.task_progress: Dict[str, TaskProgress] = {}
        
        # Agent registry
        self._register_agents()
        
        # Decision history
        self.decision_history: List[OrchestratorDecision] = []
        
        # User intent cache
        self.user_intent_cache: Dict[str, Any] = {}
        
        # Active tasks
        self.active_tasks: Set[str] = set()
    
    def _register_agents(self):
        """Register all available agents"""
        # Initialize agents
        planner = PlannerAgent()
        developer = DeveloperAgent()
        linter = LinterAgent()
        debugger = DebuggerAgent()
        
        # Set dependencies
        for agent in [planner, developer, linter, debugger]:
            agent.set_llm_client(self.llm_client)
            agent.set_state_manager(self.state_manager)
        
        # Register agents
        self.agents = {
            "planner": planner,
            "developer": developer,
            "linter": linter,
            "debugger": debugger
        }
        
        self.logger.info(f"Registered {len(self.agents)} agents")
    
    async def initialize(self):
        """Initialize orchestrator"""
        self.logger.info("Initializing orchestrator...")
        
        # Load previous state if any
        state = self.state_manager.get_state()
        
        # Initialize task progress tracking
        for task_id in self.state_manager._tasks:
            self.task_progress[task_id] = TaskProgress(
                task_id=task_id,
                phase="pending",
                status="waiting",
                progress=0.0
            )
        
        self.logger.info("Orchestrator initialized")
    
    async def shutdown(self):
        """Shutdown orchestrator"""
        self.logger.info("Shutting down orchestrator...")
        
        # Save current state
        self.state_manager.save_state()
        
        # Cancel any running tasks
        for task_id in list(self.active_tasks):
            await self._cancel_task(task_id)
    
    async def process_input(self, user_input: str) -> str:
        """
        Process user input and route to appropriate agents
        
        Args:
            user_input: User command or query
            
        Returns:
            System response
        """
        self.logger.info(f"Processing input: {user_input[:100]}...")
        
        # Parse command
        command, args = self._parse_command(user_input)
        
        # Handle system commands
        if command in self._get_system_commands():
            return await self._handle_system_command(command, args, user_input)
        
        # Analyze user intent
        intent = await self._analyze_intent(user_input)
        
        # Update state
        self.state_manager.update_state(current_task=intent.get("task", "unknown"))
        
        # Make routing decision
        decision = await self._make_routing_decision(intent)
        
        # Execute decision
        result = await self._execute_decision(decision, intent)
        
        # Update mode if needed
        await self._update_system_mode(result)
        
        # Suggest next steps
        next_steps = await self._suggest_next_steps(result)
        
        # Format response
        response = self._format_response(result, next_steps)
        
        return response
    
    def _parse_command(self, input_str: str) -> tuple[str, List[str]]:
        """Parse command string"""
        input_str = input_str.strip()
        
        # Check for slash commands
        if input_str.startswith('/'):
            parts = input_str[1:].split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1].split() if len(parts) > 1 else []
            return command, args
        
        # Default to chat mode
        return "chat", [input_str]
    
    def _get_system_commands(self) -> Set[str]:
        """Get supported system commands"""
        return {
            "help", "chat", "planner", "dev", "debug", "lint",
            "load", "save", "status", "reset", "unsafe",
            "mode", "tasks", "files", "history", "clear"
        }
    
    async def _handle_system_command(
        self, 
        command: str, 
        args: List[str],
        original_input: str
    ) -> str:
        """Handle system-level commands"""
        if command == "help":
            return self._get_help_text()
        
        elif command == "chat":
            self.mode_manager.transition(
                SystemMode.CHAT,
                "User requested chat mode",
                user_initiated=True
            )
            return "Switched to chat mode. How can I help you?"
        
        elif command == "planner":
            success = self.mode_manager.transition(
                SystemMode.PLANNER,
                "User requested planner mode",
                user_initiated=True
            )
            if success:
                return "Switched to planner mode. Describe what you want to build."
            else:
                return "Cannot switch to planner mode. Check system state."
        
        elif command == "dev":
            success = self.mode_manager.transition(
                SystemMode.DEV,
                "User requested development mode",
                user_initiated=True
            )
            if success:
                return "Switched to development mode. Ready for code execution."
            else:
                return "Cannot switch to development mode. Planning may be required first."
        
        elif command == "debug":
            success = self.mode_manager.transition(
                SystemMode.DEBUG,
                "User requested debug mode",
                user_initiated=True
            )
            if success:
                return "Switched to debug mode. Ready for error analysis."
            else:
                return "Cannot switch to debug mode. No errors detected."
        
        elif command == "lint":
            success = self.mode_manager.transition(
                SystemMode.LINT,
                "User requested lint mode",
                user_initiated=True
            )
            if success:
                return "Switched to lint mode. Ready for code quality checks."
            else:
                return "Cannot switch to lint mode. No code files loaded."
        
        elif command == "load":
            if not args:
                return "Usage: /load <file_path>"
            
            file_path = args[0]
            return await self._load_file(file_path)
        
        elif command == "unsafe":
            if not args or args[0] not in ["on", "off"]:
                return "Usage: /unsafe <on|off>"
            
            enable = args[0] == "on"
            reason = " ".join(args[1:]) if len(args) > 1 else "user request"
            
            if enable:
                success = self.safety_system.disable(reason)
                if success:
                    return f"⚠️ Safety override enabled: {reason}"
                else:
                    return "Failed to enable safety override"
            else:
                self.safety_system.enable()
                return "✅ Safety system re-enabled"
        
        elif command == "status":
            return await self._get_system_status()
        
        elif command == "tasks":
            return await self._list_tasks()
        
        elif command == "files":
            return await self._list_files()
        
        elif command == "history":
            return await self._show_history()
        
        elif command == "clear":
            self.user_intent_cache.clear()
            return "Cleared intent cache."
        
        else:
            return f"Unknown command: {command}. Use /help for available commands."
    
    async def _analyze_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Analyze user intent using LLM
        
        Returns:
            Dict with intent classification and metadata
        """
        # Check cache
        cache_key = f"intent:{hash(user_input)}"
        if cache_key in self.user_intent_cache:
            return self.user_intent_cache[cache_key]
        
        prompt = f"""
        Analyze the user's intent and classify what they want to do.
        
        User input: {user_input}
        
        Classify into one of these categories:
        1. chat_general - General conversation or questions
        2. planning_request - Request to plan a project or feature
        3. coding_request - Request to write, modify, or debug code
        4. system_command - System-level operation
        5. file_operation - Working with files
        6. dependency_request - Managing dependencies
        7. testing_request - Testing related
        8. documentation_request - Documentation related
        
        Return a JSON object with:
        - category: The intent category
        - confidence: 0.0 to 1.0
        - urgency: "low", "medium", "high"
        - estimated_complexity: "simple", "moderate", "complex"
        - requires_agents: List of agent types needed
        - suggested_mode: Which system mode to use
        - task_description: Brief description of the task
        """
        
        try:
            response = self.llm_client.chat_completion([
                {"role": "system", "content": "You are an intent analyzer for ASEA-X."},
                {"role": "user", "content": prompt}
            ])
            
            # Parse response
            intent_data = json.loads(response.content)
            
            # Cache result
            self.user_intent_cache[cache_key] = intent_data
            
            # Log intent
            self.state_manager.log_action(
                agent="orchestrator",
                action="intent_analysis",
                details={
                    "input": user_input[:200],
                    "intent": intent_data
                }
            )
            
            return intent_data
            
        except Exception as e:
            self.logger.error(f"Intent analysis failed: {e}")
            
            # Fallback intent
            return {
                "category": "chat_general",
                "confidence": 0.5,
                "urgency": "medium",
                "estimated_complexity": "simple",
                "requires_agents": [],
                "suggested_mode": "chat",
                "task_description": "General conversation"
            }
    
    async def _make_routing_decision(self, intent: Dict[str, Any]) -> OrchestratorDecision:
        """
        Make routing decision based on intent
        
        Returns:
            OrchestratorDecision with target agent and action
        """
        category = intent.get("category", "chat_general")
        current_mode = self.mode_manager.current_mode
        
        # Decision matrix based on intent and current mode
        decision_matrix = {
            "planning_request": {
                "target": "planner",
                "action": "create_plan",
                "requires_confirmation": intent.get("estimated_complexity") == "complex"
            },
            "coding_request": {
                "target": "developer",
                "action": "write_code",
                "requires_confirmation": False
            },
            "debugging_request": {
                "target": "debugger",
                "action": "debug_code",
                "requires_confirmation": False
            },
            "linting_request": {
                "target": "linter",
                "action": "lint_code",
                "requires_confirmation": False
            },
            "documentation_request": {
                "target": "developer",  # Developer handles docs too
                "action": "write_documentation",
                "requires_confirmation": False
            }
        }
        
        # Get decision from matrix or default to chat
        decision_info = decision_matrix.get(category, {
            "target": None,
            "action": "chat",
            "requires_confirmation": False
        })
        
        # Check if we should override based on current mode
        if current_mode == SystemMode.DEBUG and category != "debugging_request":
            # If in debug mode but intent isn't debugging, force debug agent
            decision_info = {
                "target": "debugger",
                "action": "analyze_error",
                "requires_confirmation": False
            }
        
        decision = OrchestratorDecision(
            action=decision_info["action"],
            target_agent=decision_info["target"],
            reason=f"Intent: {category}, Mode: {current_mode}",
            confidence=intent.get("confidence", 0.7),
            requires_confirmation=decision_info["requires_confirmation"],
            metadata={"intent": intent}
        )
        
        # Save decision to history
        self.decision_history.append(decision)
        
        return decision
    
    async def _execute_decision(
        self, 
        decision: OrchestratorDecision, 
        intent: Dict[str, Any]
    ) -> AgentResponse:
        """
        Execute routing decision by calling appropriate agent
        
        Returns:
            AgentResponse from the executed agent
        """
        # If no target agent, handle as chat
        if not decision.target_agent:
            return await self._handle_chat(intent)
        
        # Check if agent exists
        if decision.target_agent not in self.agents:
            return AgentResponse(
                success=False,
                message=f"Agent '{decision.target_agent}' not found",
                requires_human_input=True
            )
        
        # Create task
        task_id = self.state_manager.create_task(
            description=intent.get("task_description", "Unknown task"),
            metadata={"intent": intent, "decision": decision.dict()}
        )
        
        # Create context
        context = AgentContext(
            task_id=task_id,
            task_description=intent.get("task_description", ""),
            system_state=self.state_manager.get_state().dict(),
            file_context=self.state_manager.get_state().file_context,
            previous_results=[],
            user_input=intent.get("raw_input", ""),
            mode=self.mode_manager.current_mode.value
        )
        
        # Get agent
        agent = self.agents[decision.target_agent]
        
        # Track task
        self.active_tasks.add(task_id)
        self.task_progress[task_id] = TaskProgress(
            task_id=task_id,
            phase="executing",
            status="running",
            progress=0.1,
            estimated_time_remaining=None
        )
        
        try:
            # Execute agent
            response = await agent.run(context)
            
            # Update task progress
            self.task_progress[task_id].status = "completed" if response.success else "failed"
            self.task_progress[task_id].progress = 1.0
            self.task_progress[task_id].last_update = datetime.now()
            
            # Update task in state manager
            self.state_manager.update_task(
                task_id,
                status="completed" if response.success else "failed",
                result=response.dict(),
                error=response.error
            )
            
            # Log execution
            self.state_manager.log_action(
                agent="orchestrator",
                action="agent_execution",
                details={
                    "agent": decision.target_agent,
                    "task_id": task_id,
                    "success": response.success,
                    "decision": decision.dict()
                }
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            
            # Update task as failed
            self.task_progress[task_id].status = "failed"
            self.task_progress[task_id].progress = 0.0
            
            self.state_manager.update_task(
                task_id,
                status="failed",
                error=str(e)
            )
            
            return AgentResponse(
                success=False,
                message=f"Agent execution failed: {str(e)}",
                error=str(e),
                requires_human_input=True
            )
            
        finally:
            self.active_tasks.discard(task_id)
    
    async def _handle_chat(self, intent: Dict[str, Any]) -> AgentResponse:
        """Handle general chat/conversation"""
        # Use LLM for chat response
        prompt = f"""
        You are ASEA-X, an Autonomous Software Engineering Agent.
        Current system mode: {self.mode_manager.current_mode}
        
        User: {intent.get('raw_input', '')}
        
        Respond helpfully as a software engineering assistant.
        If the user seems to want to build something, suggest using /planner mode.
        If they have code issues, suggest /dev or /debug mode.
        
        Keep responses concise but helpful.
        """
        
        try:
            response = self.llm_client.chat_completion([
                {"role": "system", "content": prompt}
            ])
            
            return AgentResponse(
                success=True,
                message=response.content,
                data={"type": "chat_response"},
                next_step="continue_chat"
            )
            
        except Exception as e:
            self.logger.error(f"Chat failed: {e}")
            return AgentResponse(
                success=False,
                message="I encountered an error processing your message.",
                error=str(e)
            )
    
    async def _update_system_mode(self, agent_response: AgentResponse):
        """Update system mode based on agent response"""
        current_state = self.state_manager.get_state()
        
        # Check for error conditions
        if agent_response.error and self.mode_manager.current_mode != SystemMode.DEBUG:
            self.mode_manager.transition(
                SystemMode.DEBUG,
                f"Error detected: {agent_response.error[:100]}",
                user_initiated=False
            )
        
        # Check if planning is complete and we should switch to dev
        elif (self.mode_manager.current_mode == SystemMode.PLANNER and 
              agent_response.success and 
              agent_response.data.get("plan_complete", False)):
            
            self.mode_manager.transition(
                SystemMode.DEV,
                "Planning complete, switching to development",
                user_initiated=False
            )
        
        # Check if linting passed and we should return to dev
        elif (self.mode_manager.current_mode == SystemMode.LINT and
              agent_response.success and
              agent_response.data.get("lint_passed", False)):
            
            self.mode_manager.transition(
                SystemMode.DEV,
                "Linting passed, returning to development",
                user_initiated=False
            )
    
    async def _suggest_next_steps(self, agent_response: AgentResponse) -> List[str]:
        """Suggest next steps based on agent response"""
        suggestions = []
        
        if not agent_response.success:
            suggestions.append("Switch to /debug mode to analyze the error")
            return suggestions
        
        # Get current state
        state = self.state_manager.get_state()
        
        # Planning suggestions
        if self.mode_manager.current_mode == SystemMode.PLANNER:
            if agent_response.data.get("has_plan", False):
                suggestions.append("Switch to /dev mode to start implementation")
                suggestions.append("Review and modify the plan with /planner")
        
        # Development suggestions
        elif self.mode_manager.current_mode == SystemMode.DEV:
            if agent_response.data.get("code_written", False):
                suggestions.append("Run /lint to check code quality")
                suggestions.append("Write tests for the new code")
            
            if agent_response.data.get("tests_failed", False):
                suggestions.append("Switch to /debug mode to fix failing tests")
        
        # Debug suggestions
        elif self.mode_manager.current_mode == SystemMode.DEBUG:
            if agent_response.data.get("error_fixed", False):
                suggestions.append("Return to /dev mode to continue development")
                suggestions.append("Add tests to prevent regression")
        
        # Lint suggestions
        elif self.mode_manager.current_mode == SystemMode.LINT:
            if agent_response.data.get("lint_passed", False):
                suggestions.append("Continue in /dev mode")
            else:
                suggestions.append("Fix the linting issues mentioned above")
        
        # Add general suggestions
        if len(state.file_context) > 0:
            suggestions.append("Use /files to see loaded files")
        
        if len(self.state_manager._tasks) > 0:
            suggestions.append("Use /tasks to see task status")
        
        return suggestions
    
    def _format_response(self, agent_response: AgentResponse, next_steps: List[str]) -> str:
        """Format system response"""
        response_parts = []
        
        # Add main message
        response_parts.append(agent_response.message)
        
        # Add data if present
        if agent_response.data:
            # Format code if present
            if "code" in agent_response.data:
                response_parts.append(f"\n```python\n{agent_response.data['code']}\n```")
            
            # Format diff if present
            if "diff" in agent_response.data:
                response_parts.append(f"\n```diff\n{agent_response.data['diff']}\n```")
            
            # Format plan if present
            if "plan" in agent_response.data:
                plan = agent_response.data["plan"]
                response_parts.append(f"\n**Plan:**\n{plan}")
        
        # Add next steps if any
        if next_steps:
            response_parts.append("\n**Suggested next steps:**")
            for i, step in enumerate(next_steps, 1):
                response_parts.append(f"{i}. {step}")
        
        # Add error if present
        if agent_response.error:
            response_parts.append(f"\n⚠️ **Error:** {agent_response.error}")
        
        return "\n".join(response_parts)
    
    async def _load_file(self, file_path: str) -> str:
        """Load file into system context"""
        try:
            from pathlib import Path
            
            path = Path(file_path)
            if not path.exists():
                return f"File not found: {file_path}"
            
            # Read file
            content = path.read_text(encoding='utf-8', errors='ignore')
            
            # Add to context
            self.state_manager.add_file_context(file_path, content)
            
            # Log action
            self.state_manager.log_action(
                agent="orchestrator",
                action="file_load",
                details={
                    "file_path": file_path,
                    "size_bytes": len(content),
                    "lines": len(content.splitlines())
                }
            )
            
            return f"Loaded {file_path} ({len(content)} bytes, {len(content.splitlines())} lines)"
            
        except Exception as e:
            return f"Failed to load file: {str(e)}"
    
    async def _get_system_status(self) -> str:
        """Get system status"""
        state = self.state_manager.get_state()
        
        status_lines = [
            "**System Status:**",
            f"Mode: {state.current_mode}",
            f"Safety: {'✅ Enabled' if state.safety_enabled else '❌ Disabled'}",
            f"Tasks: {len(self.state_manager._tasks)} total, {len(self.active_tasks)} active",
            f"Files in context: {len(state.file_context)}",
            f"Agents: {len(self.agents)} available",
            f"LLM: {self.llm_client.config.model}",
            f"Uptime: {time.time() - state.start_time:.0f}s"
        ]
        
        # Add active tasks
        if self.active_tasks:
            status_lines.append("\n**Active Tasks:**")
            for task_id in self.active_tasks:
                progress = self.task_progress.get(task_id)
                if progress:
                    status_lines.append(f"  - {task_id}: {progress.phase} ({progress.progress:.0%})")
        
        return "\n".join(status_lines)
    
    async def _list_tasks(self) -> str:
        """List all tasks"""
        tasks = self.state_manager._tasks
        
        if not tasks:
            return "No tasks found."
        
        lines = ["**Tasks:**"]
        for task_id, task in sorted(tasks.items(), key=lambda x: x[1].created_at):
            lines.append(
                f"  - {task_id}: {task.description[:50]}... "
                f"[{task.status}] ({task.created_at:.0f}s ago)"
            )
        
        return "\n".join(lines)
    
    async def _list_files(self) -> str:
        """List files in context"""
        files = self.state_manager.get_state().file_context
        
        if not files:
            return "No files loaded. Use /load <file> to load files."
        
        lines = ["**Files in context:**"]
        for file_path, content in files.items():
            lines.append(f"  - {file_path}: {len(content)} bytes, {len(content.splitlines())} lines")
        
        return "\n".join(lines)
    
    async def _show_history(self) -> str:
        """Show recent actions"""
        history = self.state_manager.get_state().task_history[-10:]  # Last 10
        
        if not history:
            return "No history yet."
        
        lines = ["**Recent Actions:**"]
        for entry in reversed(history):
            timestamp = datetime.fromtimestamp(entry["timestamp"]).strftime("%H:%M:%S")
            lines.append(f"  [{timestamp}] {entry['agent']}: {entry['action']}")
        
        return "\n".join(lines)
    
    async def _cancel_task(self, task_id: str):
        """Cancel a running task"""
        if task_id in self.active_tasks:
            self.active_tasks.discard(task_id)
            self.state_manager.update_task(task_id, status="cancelled")
            self.logger.info(f"Cancelled task {task_id}")
    
    def _get_help_text(self) -> str:
        """Get help text"""
        return """
**ASEA-X Commands:**

**Mode Commands:**
  /chat        - Switch to chat mode (default)
  /planner     - Switch to planner mode
  /dev         - Switch to development mode
  /debug       - Switch to debug mode
  /lint        - Switch to lint mode

**System Commands:**
  /load <file> - Load file into context
  /unsafe on|off [reason] - Enable/disable safety override
  /status      - Show system status
  /tasks       - List tasks
  /files       - List files in context
  /history     - Show recent actions
  /clear       - Clear intent cache
  /help        - Show this help

**Usage:**
  1. Start with /planner to plan your project
  2. Switch to /dev to implement the plan
  3. Use /lint to check code quality
  4. Use /debug if you encounter errors
  5. Use /chat for general questions

**Safety:**
  - System is safe by default
  - Use /unsafe on to enable dangerous operations
  - All unsafe operations are logged
        """

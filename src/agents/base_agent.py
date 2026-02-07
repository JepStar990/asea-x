"""
Base agent class for ASEA-X
All agents inherit from this base class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

from pydantic import BaseModel, Field


class AgentCapability(str, Enum):
    """Agent capabilities"""
    CODE_WRITING = "code_writing"
    CODE_READING = "code_reading"
    COMMAND_EXECUTION = "command_execution"
    FILE_OPERATIONS = "file_operations"
    PLANNING = "planning"
    DEBUGGING = "debugging"
    LINTING = "linting"
    DOCUMENTATION = "documentation"


class AgentResponse(BaseModel):
    """Standardized agent response"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    next_step: Optional[str] = None
    requires_human_input: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class AgentContext:
    """Context passed to agents"""
    task_id: str
    task_description: str
    system_state: Dict[str, Any]
    file_context: Dict[str, str]
    previous_results: List[Dict[str, Any]]
    user_input: Optional[str] = None
    mode: str = "chat"


class BaseAgent(ABC):
    """
    Abstract base class for all ASEA-X agents
    
    Agents must:
    1. Inherit from BaseAgent
    2. Implement execute method
    3. Declare capabilities
    4. Handle their own errors
    """
    
    def __init__(self, name: str, capabilities: List[AgentCapability]):
        self.name = name
        self.capabilities = capabilities
        self.logger = logging.getLogger(f"agent.{name}")
        self.llm_client = None
        self.state_manager = None
        
    def set_llm_client(self, llm_client):
        """Set LLM client for agent"""
        self.llm_client = llm_client
    
    def set_state_manager(self, state_manager):
        """Set state manager for agent"""
        self.state_manager = state_manager
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has specific capability"""
        return capability in self.capabilities
    
    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResponse:
        """
        Execute agent's primary function
        
        Args:
            context: Agent context with task info
            
        Returns:
            AgentResponse with results
        """
        pass
    
    def validate_context(self, context: AgentContext) -> Optional[str]:
        """
        Validate context before execution
        
        Returns:
            Optional error message if validation fails
        """
        if not context.task_id:
            return "Missing task_id"
        
        if not context.task_description:
            return "Missing task_description"
        
        return None
    
    def log_execution_start(self, context: AgentContext) -> None:
        """Log execution start"""
        self.logger.info(
            f"Starting execution for task {context.task_id}: {context.task_description}"
        )
        
        if self.state_manager:
            self.state_manager.log_action(
                agent=self.name,
                action="execution_start",
                details={
                    "task_id": context.task_id,
                    "description": context.task_description,
                    "mode": context.mode
                }
            )
    
    def log_execution_end(
        self, 
        context: AgentContext, 
        response: AgentResponse,
        execution_time: float
    ) -> None:
        """Log execution end"""
        log_level = logging.INFO if response.success else logging.ERROR
        
        self.logger.log(
            log_level,
            f"Completed execution for task {context.task_id} "
            f"in {execution_time:.2f}s: {response.message}"
        )
        
        if self.state_manager:
            self.state_manager.log_action(
                agent=self.name,
                action="execution_end",
                details={
                    "task_id": context.task_id,
                    "success": response.success,
                    "execution_time": execution_time,
                    "message": response.message[:100] if response.message else "",
                    "error": response.error
                }
            )
    
    async def run(self, context: AgentContext) -> AgentResponse:
        """
        Run agent with standardized logging and error handling
        
        Args:
            context: Agent context
            
        Returns:
            AgentResponse (guaranteed, even on error)
        """
        start_time = time.time()
        
        # Validate context
        validation_error = self.validate_context(context)
        if validation_error:
            return AgentResponse(
                success=False,
                message=f"Context validation failed: {validation_error}",
                requires_human_input=True
            )
        
        # Log start
        self.log_execution_start(context)
        
        try:
            # Execute agent logic
            response = await self.execute(context)
            response.metadata["execution_time"] = time.time() - start_time
            
        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}", exc_info=True)
            response = AgentResponse(
                success=False,
                message=f"Agent execution failed: {str(e)}",
                error=str(e),
                requires_human_input=True,
                metadata={"execution_time": time.time() - start_time}
            )
        
        # Log end
        self.log_execution_end(context, response, time.time() - start_time)
        
        return response
    
    def format_code_block(self, code: str, language: str = "python") -> str:
        """Format code as markdown block"""
        return f"```{language}\n{code}\n```"
    
    def format_diff(self, old_content: str, new_content: str) -> str:
        """Format diff between old and new content"""
        import difflib
        
        diff = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile='old',
            tofile='new',
            lineterm=''
        )
        
        return "".join(diff)
    
    def suggest_next_agent(self, context: AgentContext, result: AgentResponse) -> Optional[str]:
        """
        Suggest which agent should handle next step
        
        Returns:
            Agent name or None
        """
        # Base implementation - override in specific agents
        return None

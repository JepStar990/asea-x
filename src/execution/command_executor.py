"""
Command Executor - Safely executes system commands
with timeout, monitoring, and safety checks
"""

import subprocess
import shlex
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import time

from src.core.safety_system import SafetySystem, SafetyCheckResult


@dataclass
class CommandResult:
    """Result of command execution"""
    success: bool
    command: str
    stdout: str
    stderr: str
    returncode: int
    execution_time: float
    timeout: bool = False
    killed: bool = False
    safety_check: Optional[SafetyCheckResult] = None


class CommandExecutor:
    """
    Executes system commands with safety and monitoring
    
    Features:
    1. Safety checks before execution
    2. Timeout handling
    3. Resource monitoring
    4. Result parsing
    5. Retry logic
    """
    
    def __init__(self, safety_system: Optional[SafetySystem] = None):
        self.safety_system = safety_system
        self.logger = logging.getLogger(__name__)
        
        # Execution limits
        self.default_timeout = 30  # seconds
        self.max_output_size = 10 * 1024 * 1024  # 10MB
        
        # Command history
        self.history: List[CommandResult] = []
    
    async def execute(
        self,
        command: str,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        allow_unsafe: bool = False
    ) -> CommandResult:
        """
        Execute a command with safety checks
        
        Args:
            command: Command string to execute
            timeout: Timeout in seconds (default: 30)
            cwd: Working directory
            env: Environment variables
            allow_unsafe: Skip safety checks
            
        Returns:
            CommandResult with execution details
        """
        start_time = time.time()
        
        # Safety check
        safety_result = None
        if self.safety_system and not allow_unsafe:
            safety_result = self.safety_system.check_command(command)
            if not safety_result.allowed:
                return CommandResult(
                    success=False,
                    command=command,
                    stdout="",
                    stderr=f"Command blocked by safety system: {safety_result.reason}",
                    returncode=-1,
                    execution_time=time.time() - start_time,
                    safety_check=safety_result
                )
        
        # Parse command
        try:
            args = shlex.split(command)
            if not args:
                return CommandResult(
                    success=False,
                    command=command,
                    stdout="",
                    stderr="Empty command",
                    returncode=-1,
                    execution_time=time.time() - start_time
                )
        except ValueError as e:
            return CommandResult(
                success=False,
                command=command,
                stdout="",
                stderr=f"Invalid command syntax: {e}",
                returncode=-1,
                execution_time=time.time() - start_time
            )
        
        # Set defaults
        timeout = timeout or self.default_timeout
        
        self.logger.info(f"Executing command: {command}")
        
        try:
            # Execute command with timeout
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                # Decode output
                stdout_text = stdout.decode('utf-8', errors='ignore')
                stderr_text = stderr.decode('utf-8', errors='ignore')
                
                # Limit output size
                if len(stdout_text) > self.max_output_size:
                    stdout_text = stdout_text[:self.max_output_size] + "\n...[truncated]"
                if len(stderr_text) > self.max_output_size:
                    stderr_text = stderr_text[:self.max_output_size] + "\n...[truncated]"
                
                success = process.returncode == 0
                
                result = CommandResult(
                    success=success,
                    command=command,
                    stdout=stdout_text,
                    stderr=stderr_text,
                    returncode=process.returncode,
                    execution_time=time.time() - start_time,
                    safety_check=safety_result
                )
                
            except asyncio.TimeoutError:
                # Kill the process
                process.kill()
                await process.wait()
                
                result = CommandResult(
                    success=False,
                    command=command,
                    stdout="",
                    stderr=f"Command timed out after {timeout} seconds",
                    returncode=-1,
                    execution_time=time.time() - start_time,
                    timeout=True,
                    killed=True,
                    safety_check=safety_result
                )
                
        except Exception as e:
            result = CommandResult(
                success=False,
                command=command,
                stdout="",
                stderr=f"Command execution failed: {str(e)}",
                returncode=-1,
                execution_time=time.time() - start_time,
                safety_check=safety_result
            )
        
        # Add to history
        self.history.append(result)
        
        # Log result
        log_level = logging.INFO if result.success else logging.WARNING
        self.logger.log(
            log_level,
            f"Command {'succeeded' if result.success else 'failed'} "
            f"in {result.execution_time:.2f}s: {command}"
        )
        
        return result
    
    def execute_sync(
        self,
        command: str,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        allow_unsafe: bool = False
    ) -> CommandResult:
        """
        Synchronous version of execute
        
        Returns:
            CommandResult
        """
        return asyncio.run(self.execute(command, timeout, cwd, env, allow_unsafe))
    
    async def execute_with_retry(
        self,
        command: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ) -> CommandResult:
        """
        Execute command with retry logic
        
        Args:
            command: Command to execute
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            **kwargs: Additional args for execute()
            
        Returns:
            CommandResult from successful execution or last attempt
        """
        last_result = None
        
        for attempt in range(max_retries):
            if attempt > 0:
                self.logger.info(f"Retry attempt {attempt}/{max_retries} for: {command}")
                await asyncio.sleep(retry_delay)
            
            result = await self.execute(command, **kwargs)
            last_result = result
            
            if result.success:
                return result
            
            # Only retry on certain errors
            if result.timeout or result.returncode in [137, 143]:  # Killed signals
                continue
            else:
                break
        
        return last_result
    
    def get_command_history(self, limit: int = 10) -> List[CommandResult]:
        """Get command execution history"""
        return self.history[-limit:] if self.history else []
    
    def clear_history(self):
        """Clear command history"""
        self.history.clear()
    
    async def check_command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH"""
        # Extract base command (first word)
        parts = shlex.split(command)
        if not parts:
            return False
        
        base_cmd = parts[0]
        
        # Use 'which' or 'where' depending on platform
        check_cmd = f"which {base_cmd}" if os.name != 'nt' else f"where {base_cmd}"
        
        try:
            result = await self.execute(check_cmd, timeout=5)
            return result.success and result.stdout.strip() != ""
        except:
            return False
    
    async def get_command_help(self, command: str) -> Optional[str]:
        """Get help text for a command"""
        help_commands = {
            "python": "python --help",
            "pip": "pip --help",
            "npm": "npm help",
            "node": "node --help",
            "git": "git --help",
            "docker": "docker --help",
            "kubectl": "kubectl --help",
            "terraform": "terraform --help"
        }
        
        # Extract base command
        parts = shlex.split(command)
        if not parts:
            return None
        
        base_cmd = parts[0]
        
        if base_cmd in help_commands:
            result = await self.execute(help_commands[base_cmd], timeout=10)
            if result.success:
                return result.stdout[:1000]  # Limit size
            else:
                return result.stderr[:1000]
        
        return None

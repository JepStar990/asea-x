"""
Safety system for ASEA-X
Prevents dangerous operations and enforces safety constraints
"""

from __future__ import annotations

import re
import shlex
from typing import List, Set, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum
import time

from pydantic import BaseModel, Field, validator


class SafetyLevel(str, Enum):
    """Safety levels for operations"""
    SAFE = "safe"
    WARNING = "warning"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"


class SafetyOverride(BaseModel):
    """Safety override record"""
    command: str
    reason: str
    timestamp: float
    approved_by: str = "system"
    expires_at: Optional[float] = None
    scope: str = "command"  # command, session, permanent


@dataclass
class SafetyCheckResult:
    """Result of a safety check"""
    level: SafetyLevel
    command: str
    reason: str
    allowed: bool
    override_available: bool = False
    suggested_alternative: Optional[str] = None


class SafetySystem:
    """
    Safety system that prevents dangerous operations
    Enforces safety by default with explicit override mechanism
    """
    
    # Dangerous patterns (regex)
    DANGEROUS_PATTERNS = [
        # File system destruction
        r'rm\s+-rf\s+',
        r'rm\s+-rf\s+/\s*',
        r'dd\s+if=.*\s+of=.*',
        r'mkfs\s+',
        r'fdisk\s+',
        r':\(\)\{.*\}',
        
        # Network abuse
        r'nc\s+.*\s+-e\s+/bin/sh',
        r'wget\s+.*\s+-O.*\s+.*\|\s*sh',
        r'curl\s+.*\s+\|\s*sh',
        
        # Privilege escalation
        r'sudo\s+',
        r'su\s+',
        r'chmod\s+[0-7][0-7][0-7]\s+',
        
        # Data destruction
        r'>\s+/dev/sd[a-z]',
        r'dd\s+if=/dev/zero',
        r'shred\s+',
        
        # System modification
        r'chown\s+-R',
        r'mv\s+/\s+',
        r'rm\s+-\s+',
    ]
    
    # Restricted commands (require override)
    RESTRICTED_COMMANDS = {
        'rm': ['-rf', '--no-preserve-root'],
        'chmod': ['777', '666'],
        'chown': ['root:', '0:0'],
        'dd': ['if=', 'of='],
        'mkfs': [],
        'fdisk': [],
        'sudo': [],
        'su': [],
        'passwd': [],
        'useradd': [],
        'userdel': [],
        'groupadd': [],
        'groupdel': [],
    }
    
    # Allowed commands in safe mode
    ALLOWED_COMMANDS = {
        'python', 'pip', 'npm', 'yarn', 'cargo', 
        'go', 'rustc', 'javac', 'node', 'gcc', 
        'g++', 'make', 'cmake', 'git', 'ls', 
        'cat', 'echo', 'pwd', 'cd', 'mkdir', 
        'touch', 'cp', 'mv', 'rm', 'find', 
        'grep', 'awk', 'sed', 'tar', 'zip', 
        'unzip', 'curl', 'wget'
    }
    
    def __init__(self, state_manager, enabled: bool = True):
        self.state_manager = state_manager
        self.enabled = enabled
        self.overrides: List[SafetyOverride] = []
        self.logger = logging.getLogger(__name__)
        
        # Load safety logs directory
        self.safety_log = state_manager.workdir / ".agent.safety.log"
        self.safety_log.parent.mkdir(exist_ok=True, parents=True)
        
        # Load existing overrides
        self._load_overrides()
    
    def _load_overrides(self) -> None:
        """Load safety overrides from log"""
        if self.safety_log.exists():
            try:
                with open(self.safety_log, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                override_data = json.loads(line)
                                override = SafetyOverride(**override_data)
                                
                                # Check if override is still valid
                                if (override.expires_at is None or 
                                    override.expires_at > time.time()):
                                    self.overrides.append(override)
                            except:
                                continue
            except Exception as e:
                self.logger.error(f"Failed to load safety overrides: {e}")
    
    def _save_override(self, override: SafetyOverride) -> None:
        """Save override to log"""
        self.overrides.append(override)
        
        with open(self.safety_log, 'a') as f:
            f.write(json.dumps(override.dict()) + "\n")
    
    def check_command(self, command: str, 
                     context: Optional[Dict[str, Any]] = None) -> SafetyCheckResult:
        """
        Check if a command is safe to execute
        
        Args:
            command: The command string to check
            context: Additional context for the check
            
        Returns:
            SafetyCheckResult with safety level and recommendation
        """
        if not self.enabled:
            return SafetyCheckResult(
                level=SafetyLevel.SAFE,
                command=command,
                reason="Safety system disabled",
                allowed=True
            )
        
        # Clean and parse command
        command = command.strip()
        if not command:
            return SafetyCheckResult(
                level=SafetyLevel.SAFE,
                command=command,
                reason="Empty command",
                allowed=True
            )
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return SafetyCheckResult(
                    level=SafetyLevel.BLOCKED,
                    command=command,
                    reason=f"Matches dangerous pattern: {pattern}",
                    allowed=False,
                    override_available=True
                )
        
        # Parse command parts
        try:
            parts = shlex.split(command)
            if not parts:
                return SafetyCheckResult(
                    level=SafetyLevel.SAFE,
                    command=command,
                    reason="Empty command after parsing",
                    allowed=True
                )
            
            cmd_name = parts[0]
            args = parts[1:] if len(parts) > 1 else []
            
        except ValueError:
            # Invalid shell syntax
            return SafetyCheckResult(
                level=SafetyLevel.DANGEROUS,
                command=command,
                reason="Invalid shell syntax",
                allowed=False
            )
        
        # Check if command is in allowed list
        if cmd_name not in self.ALLOWED_COMMANDS:
            return SafetyCheckResult(
                level=SafetyLevel.WARNING,
                command=command,
                reason=f"Command '{cmd_name}' not in allowed list",
                allowed=False,
                override_available=True,
                suggested_alternative=self._suggest_alternative(cmd_name, args)
            )
        
        # Check for restricted command arguments
        if cmd_name in self.RESTRICTED_COMMANDS:
            restricted_args = self.RESTRICTED_COMMANDS[cmd_name]
            for arg in args:
                for restricted in restricted_args:
                    if restricted and arg.startswith(restricted):
                        return SafetyCheckResult(
                            level=SafetyLevel.DANGEROUS,
                            command=command,
                            reason=f"Dangerous argument '{arg}' for command '{cmd_name}'",
                            allowed=False,
                            override_available=True
                        )
        
        # Check for path traversal attempts
        for part in parts:
            if '..' in part and any(x in part for x in ['/', '\\']):
                return SafetyCheckResult(
                    level=SafetyLevel.DANGEROUS,
                    command=command,
                    reason="Path traversal attempt detected",
                    allowed=False
                )
        
        # Check for existing override
        if self._has_override(command):
            return SafetyCheckResult(
                level=SafetyLevel.SAFE,
                command=command,
                reason="Override exists",
                allowed=True
            )
        
        # All checks passed
        return SafetyCheckResult(
            level=SafetyLevel.SAFE,
            command=command,
            reason="Command appears safe",
            allowed=True
        )
    
    def _suggest_alternative(self, cmd: str, args: List[str]) -> Optional[str]:
        """Suggest safer alternative for a command"""
        alternatives = {
            'rm': 'Use trash-cli or move to trash instead',
            'chmod': 'Use more restrictive permissions (e.g., 644 instead of 777)',
            'chown': 'Avoid changing ownership unless necessary',
            'sudo': 'Run without sudo or use a virtual environment',
            'wget | sh': 'Download first, inspect, then execute',
            'curl | sh': 'Download first, inspect, then execute',
        }
        
        full_cmd = f"{cmd} {' '.join(args)}"
        for dangerous, suggestion in alternatives.items():
            if dangerous in full_cmd:
                return suggestion
        
        return None
    
    def _has_override(self, command: str) -> bool:
        """Check if command has an active override"""
        current_time = time.time()
        
        for override in self.overrides:
            # Check if override matches command
            if override.command == command or (
                override.command.endswith('*') and 
                command.startswith(override.command[:-1])
            ):
                # Check if override is still valid
                if (override.expires_at is None or 
                    override.expires_at > current_time):
                    return True
        
        return False
    
    def request_override(self, command: str, reason: str, 
                        scope: str = "command",
                        duration: Optional[int] = None) -> bool:
        """
        Request a safety override
        
        Args:
            command: Command to override
            reason: Reason for override
            scope: Scope of override (command, session, permanent)
            duration: Override duration in seconds (None = permanent)
            
        Returns:
            bool: True if override was granted
        """
        # Log the override request
        self.state_manager.log_action(
            agent="safety_system",
            action="override_request",
            details={
                "command": command,
                "reason": reason,
                "scope": scope,
                "duration": duration
            }
        )
        
        # Create override
        expires_at = None
        if duration:
            expires_at = time.time() + duration
        
        override = SafetyOverride(
            command=command,
            reason=reason,
            timestamp=time.time(),
            approved_by="user",
            expires_at=expires_at,
            scope=scope
        )
        
        # Save override
        self._save_override(override)
        
        self.logger.warning(f"Safety override granted for: {command}")
        self.logger.warning(f"Reason: {reason}")
        
        return True
    
    def revoke_override(self, command: str) -> bool:
        """Revoke a safety override"""
        removed = False
        new_overrides = []
        
        for override in self.overrides:
            if override.command == command:
                removed = True
                # Log revocation
                self.state_manager.log_action(
                    agent="safety_system",
                    action="override_revoked",
                    details={"command": command}
                )
            else:
                new_overrides.append(override)
        
        self.overrides = new_overrides
        
        if removed:
            # Update log file
            with open(self.safety_log, 'w') as f:
                for override in self.overrides:
                    f.write(json.dumps(override.dict()) + "\n")
        
        return removed
    
    def get_active_overrides(self) -> List[SafetyOverride]:
        """Get list of active overrides"""
        current_time = time.time()
        return [
            override for override in self.overrides
            if (override.expires_at is None or 
                override.expires_at > current_time)
        ]
    
    def enable(self) -> None:
        """Enable safety system"""
        self.enabled = True
        self.state_manager.update_state(safety_enabled=True)
        self.logger.info("Safety system enabled")
    
    def disable(self, reason: str = "user request") -> bool:
        """
        Disable safety system (requires override)
        
        Returns:
            bool: True if disabled
        """
        # Disabling safety requires an override
        override_granted = self.request_override(
            command="*",
            reason=f"Safety system disable: {reason}",
            scope="session",
            duration=3600  # 1 hour
        )
        
        if override_granted:
            self.enabled = False
            self.state_manager.update_state(safety_enabled=False)
            self.logger.warning(f"Safety system disabled: {reason}")
            return True
        
        return False
    
    def check_file_operation(self, operation: str, 
                           source: Optional[Path] = None,
                           target: Optional[Path] = None) -> SafetyCheckResult:
        """
        Check if a file operation is safe
        
        Args:
            operation: 'read', 'write', 'delete', 'move'
            source: Source file path
            target: Target file path
            
        Returns:
            SafetyCheckResult
        """
        if not self.enabled:
            return SafetyCheckResult(
                level=SafetyLevel.SAFE,
                command=f"file_{operation}",
                reason="Safety system disabled",
                allowed=True
            )
        
        # Check for dangerous file operations
        dangerous_operations = {
            'delete': self._check_delete_operation,
            'move': self._check_move_operation,
            'write': self._check_write_operation,
            'read': self._check_read_operation,
        }
        
        checker = dangerous_operations.get(operation)
        if checker:
            return checker(source, target)
        
        return SafetyCheckResult(
            level=SafetyLevel.SAFE,
            command=f"file_{operation}",
            reason="Operation appears safe",
            allowed=True
        )
    
    def _check_delete_operation(self, source: Path, target: Optional[Path]) -> SafetyCheckResult:
        """Check delete operation safety"""
        if not source:
            return SafetyCheckResult(
                level=SafetyLevel.DANGEROUS,
                command="file_delete",
                reason="No source specified",
                allowed=False
            )
        
        # Prevent deletion of system directories
        system_dirs = ['/', '/etc', '/bin', '/usr', '/lib', '/var']
        for sys_dir in system_dirs:
            if str(source).startswith(sys_dir):
                return SafetyCheckResult(
                    level=SafetyLevel.BLOCKED,
                    command="file_delete",
                    reason=f"Cannot delete system directory: {source}",
                    allowed=False
                )
        
        # Prevent wildcard deletion in root
        if '*' in str(source) and '/' in str(source).rstrip('*'):
            return SafetyCheckResult(
                level=SafetyLevel.DANGEROUS,
                command="file_delete",
                reason="Wildcard deletion in root is dangerous",
                allowed=False,
                override_available=True
            )
        
        return SafetyCheckResult(
            level=SafetyLevel.SAFE,
            command="file_delete",
            reason="Delete operation appears safe",
            allowed=True
        )
    
    def _check_move_operation(self, source: Path, target: Path) -> SafetyCheckResult:
        """Check move operation safety"""
        if not source or not target:
            return SafetyCheckResult(
                level=SafetyLevel.DANGEROUS,
                command="file_move",
                reason="Source and target required",
                allowed=False
            )
        
        # Prevent moving to root or system directories
        system_dirs = ['/', '/etc', '/bin', '/usr', '/lib', '/var']
        for sys_dir in system_dirs:
            if str(target).startswith(sys_dir):
                return SafetyCheckResult(
                    level=SafetyLevel.BLOCKED,
                    command="file_move",
                    reason=f"Cannot move to system directory: {target}",
                    allowed=False
                )
        
        # Prevent overwriting system files
        system_files = ['/etc/passwd', '/etc/shadow', '/etc/hosts']
        if str(target) in system_files:
            return SafetyCheckResult(
                level=SafetyLevel.BLOCKED,
                command="file_move",
                reason=f"Cannot overwrite system file: {target}",
                allowed=False
            )
        
        return SafetyCheckResult(
            level=SafetyLevel.SAFE,
            command="file_move",
            reason="Move operation appears safe",
            allowed=True
        )
    
    def _check_write_operation(self, source: Optional[Path], target: Path) -> SafetyCheckResult:
        """Check write operation safety"""
        if not target:
            return SafetyCheckResult(
                level=SafetyLevel.DANGEROUS,
                command="file_write",
                reason="No target specified",
                allowed=False
            )
        
        # Prevent writing to system directories
        system_dirs = ['/', '/etc', '/bin', '/usr', '/lib', '/var']
        for sys_dir in system_dirs:
            if str(target).startswith(sys_dir):
                return SafetyCheckResult(
                    level=SafetyLevel.BLOCKED,
                    command="file_write",
                    reason=f"Cannot write to system directory: {target}",
                    allowed=False
                )
        
        return SafetyCheckResult(
            level=SafetyLevel.SAFE,
            command="file_write",
            reason="Write operation appears safe",
            allowed=True
        )
    
    def _check_read_operation(self, source: Path, target: Optional[Path]) -> SafetyCheckResult:
        """Check read operation safety"""
        if not source:
            return SafetyCheckResult(
                level=SafetyLevel.DANGEROUS,
                command="file_read",
                reason="No source specified",
                allowed=False
            )
        
        # Prevent reading sensitive files
        sensitive_files = [
            '/etc/passwd', '/etc/shadow', '/etc/hosts',
            '/root/', '/home/*/.ssh/', '*.pem', '*.key',
            '*.env', '.env*', 'secrets*'
        ]
        
        for pattern in sensitive_files:
            if fnmatch.fnmatch(str(source), pattern):
                return SafetyCheckResult(
                    level=SafetyLevel.DANGEROUS,
                    command="file_read",
                    reason=f"Reading sensitive file: {source}",
                    allowed=False,
                    override_available=True
                )
        
        return SafetyCheckResult(
            level=SafetyLevel.SAFE,
            command="file_read",
            reason="Read operation appears safe",
            allowed=True
        )


# Import needed modules at the end
import json
import fnmatch
from typing import Dict, Any

"""
Runtime Monitor - Observes and analyzes program execution
Captures output, classifies errors, and monitors resources
"""

import subprocess
import threading
import time
import psutil
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import queue


class ExecutionState(str, Enum):
    """Execution states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    KILLED = "killed"


@dataclass
class ExecutionMetrics:
    """Runtime execution metrics"""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    read_bytes: int = 0
    write_bytes: int = 0
    threads: int = 0
    open_files: int = 0
    start_time: float = 0.0
    end_time: Optional[float] = None
    duration: float = 0.0


@dataclass
class ExecutionResult:
    """Complete execution result"""
    state: ExecutionState
    returncode: int
    stdout: str
    stderr: str
    command: str
    pid: Optional[int] = None
    metrics: Optional[ExecutionMetrics] = None
    error_type: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


class RuntimeMonitor:
    """
    Monitors program execution with detailed metrics
    
    Features:
    1. Real-time output capture
    2. Resource usage monitoring
    3. Error classification
    4. Timeout handling
    5. Output analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitoring_interval = 0.1  # seconds
        
        # Output buffers
        self.stdout_buffer = queue.Queue()
        self.stderr_buffer = queue.Queue()
        
        # Callbacks
        self.stdout_callback: Optional[Callable[[str], None]] = None
        self.stderr_callback: Optional[Callable[[str], None]] = None
        self.metrics_callback: Optional[Callable[[ExecutionMetrics], None]] = None
        self.completion_callback: Optional[Callable[[ExecutionResult], None]] = None
    
    def execute(
        self,
        command: str,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        capture_output: bool = True
    ) -> ExecutionResult:
        """
        Execute command with monitoring
        
        Args:
            command: Command to execute
            timeout: Execution timeout in seconds
            cwd: Working directory
            env: Environment variables
            capture_output: Whether to capture output
            
        Returns:
            ExecutionResult with detailed metrics
        """
        start_time = time.time()
        
        try:
            # Start process
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                stdin=subprocess.PIPE,
                cwd=cwd,
                env=env,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Start monitoring threads
            monitor_thread = threading.Thread(
                target=self._monitor_process,
                args=(process, start_time, timeout),
                daemon=True
            )
            monitor_thread.start()
            
            # Capture output if requested
            stdout_lines = []
            stderr_lines = []
            
            if capture_output and process.stdout and process.stderr:
                stdout_thread = threading.Thread(
                    target=self._capture_stream,
                    args=(process.stdout, stdout_lines, self.stdout_buffer, "stdout"),
                    daemon=True
                )
                stderr_thread = threading.Thread(
                    target=self._capture_stream,
                    args=(process.stderr, stderr_lines, self.stderr_buffer, "stderr"),
                    daemon=True
                )
                
                stdout_thread.start()
                stderr_thread.start()
                
                # Wait for completion
                process.wait(timeout=timeout)
                
                # Wait for output threads
                stdout_thread.join(timeout=1)
                stderr_thread.join(timeout=1)
            
            else:
                # Wait without output capture
                process.wait(timeout=timeout)
            
            # Wait for monitor thread
            monitor_thread.join(timeout=2)
            
            # Get final metrics
            metrics = self._get_process_metrics(process.pid) if process.pid else None
            
            # Determine state
            if process.returncode == 0:
                state = ExecutionState.COMPLETED
            elif process.returncode is None:
                state = ExecutionState.TIMEOUT
                process.kill()
            else:
                state = ExecutionState.FAILED
            
            # Create result
            result = ExecutionResult(
                state=state,
                returncode=process.returncode or -1,
                stdout=''.join(stdout_lines) if capture_output else "",
                stderr=''.join(stderr_lines) if capture_output else "",
                command=command,
                pid=process.pid,
                metrics=metrics,
                execution_time=time.time() - start_time
            )
            
            # Classify error if any
            if state == ExecutionState.FAILED:
                result.error_type = self._classify_error(result)
                result.error_details = self._analyze_error(result)
            
            # Call completion callback
            if self.completion_callback:
                self.completion_callback(result)
            
            return result
            
        except subprocess.TimeoutExpired:
            # Timeout occurred
            return ExecutionResult(
                state=ExecutionState.TIMEOUT,
                returncode=-1,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                command=command,
                execution_time=time.time() - start_time,
                error_type="timeout"
            )
            
        except Exception as e:
            # Other execution error
            return ExecutionResult(
                state=ExecutionState.FAILED,
                returncode=-1,
                stdout="",
                stderr=str(e),
                command=command,
                execution_time=time.time() - start_time,
                error_type="execution_error",
                error_details={"exception": str(e), "type": type(e).__name__}
            )
    
    def _monitor_process(self, process: subprocess.Popen, start_time: float, timeout: Optional[float]):
        """Monitor process resources"""
        pid = process.pid
        last_check = start_time
        
        while process.poll() is None:
            current_time = time.time()
            
            # Check timeout
            if timeout and (current_time - start_time) > timeout:
                process.kill()
                break
            
            # Collect metrics periodically
            if (current_time - last_check) >= self.monitoring_interval:
                metrics = self._get_process_metrics(pid)
                if metrics and self.metrics_callback:
                    self.metrics_callback(metrics)
                
                last_check = current_time
            
            time.sleep(self.monitoring_interval)
        
        # Final metrics
        metrics = self._get_process_metrics(pid)
        if metrics:
            metrics.end_time = time.time()
            metrics.duration = metrics.end_time - metrics.start_time
            
            if self.metrics_callback:
                self.metrics_callback(metrics)
    
    def _get_process_metrics(self, pid: int) -> Optional[ExecutionMetrics]:
        """Get metrics for a process"""
        try:
            process = psutil.Process(pid)
            
            # CPU and memory
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            
            # IO counters
            io_counters = process.io_counters()
            read_bytes = io_counters.read_bytes if io_counters else 0
            write_bytes = io_counters.write_bytes if io_counters else 0
            
            # Other metrics
            threads = process.num_threads()
            open_files = len(process.open_files())
            
            return ExecutionMetrics(
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                read_bytes=read_bytes,
                write_bytes=write_bytes,
                threads=threads,
                open_files=open_files,
                start_time=process.create_time()
            )
            
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None
    
    def _capture_stream(self, stream, lines_list: List[str], buffer: queue.Queue, stream_name: str):
        """Capture output from a stream"""
        try:
            for line in iter(stream.readline, ''):
                lines_list.append(line)
                buffer.put(line)
                
                # Call callback if set
                if stream_name == "stdout" and self.stdout_callback:
                    self.stdout_callback(line)
                elif stream_name == "stderr" and self.stderr_callback:
                    self.stderr_callback(line)
                    
        except Exception as e:
            self.logger.error(f"Error capturing {stream_name}: {e}")
    
    def _classify_error(self, result: ExecutionResult) -> str:
        """Classify error based on output"""
        stderr = result.stderr.lower()
        stdout = result.stdout.lower()
        
        # Check for common error patterns
        error_patterns = {
            "import_error": ["importerror", "modulenotfound", "no module named"],
            "syntax_error": ["syntaxerror", "invalid syntax"],
            "type_error": ["typeerror"],
            "value_error": ["valueerror"],
            "attribute_error": ["attributeerror"],
            "key_error": ["keyerror"],
            "index_error": ["indexerror"],
            "file_not_found": ["filenotfounderror", "no such file or directory"],
            "permission_error": ["permissionerror", "permission denied"],
            "timeout_error": ["timeout", "timed out"],
            "memory_error": ["memoryerror", "out of memory"],
            "connection_error": ["connectionerror", "connection refused"],
            "dependency_error": ["dependency", "requirement", "package not found"]
        }
        
        for error_type, patterns in error_patterns.items():
            if any(pattern in stderr or pattern in stdout for pattern in patterns):
                return error_type
        
        # Check return code patterns
        if result.returncode == 127:
            return "command_not_found"
        elif result.returncode == 126:
            return "permission_denied"
        elif result.returncode == 137:
            return "killed_sigkill"
        elif result.returncode == 143:
            return "killed_sigterm"
        
        return "unknown_error"
    
    def _analyze_error(self, result: ExecutionResult) -> Dict[str, Any]:
        """Analyze error for details"""
        analysis = {
            "returncode": result.returncode,
            "stderr_snippet": result.stderr[:500],
            "stdout_snippet": result.stdout[:500]
        }
        
        # Try to extract more details based on error type
        if result.error_type == "import_error":
            # Extract module name
            import re
            module_match = re.search(r"'(.*?)'", result.stderr)
            if module_match:
                analysis["missing_module"] = module_match.group(1)
        
        elif result.error_type == "syntax_error":
            # Extract line number
            import re
            line_match = re.search(r"line (\d+)", result.stderr)
            if line_match:
                analysis["line_number"] = int(line_match.group(1))
        
        elif result.error_type == "file_not_found":
            # Extract file path
            import re
            file_match = re.search(r"'([^']+)'", result.stderr)
            if file_match:
                analysis["missing_file"] = file_match.group(1)
        
        # Add resource usage if available
        if result.metrics:
            analysis.update({
                "cpu_usage": result.metrics.cpu_percent,
                "memory_usage_mb": result.metrics.memory_mb,
                "duration": result.metrics.duration
            })
        
        return analysis
    
    def get_realtime_output(self) -> List[str]:
        """Get real-time output from buffer"""
        output = []
        while not self.stdout_buffer.empty():
            try:
                output.append(self.stdout_buffer.get_nowait())
            except queue.Empty:
                break
        return output
    
    def get_realtime_errors(self) -> List[str]:
        """Get real-time errors from buffer"""
        errors = []
        while not self.stderr_buffer.empty():
            try:
                errors.append(self.stderr_buffer.get_nowait())
            except queue.Empty:
                break
        return errors
    
    def clear_buffers(self):
        """Clear output buffers"""
        while not self.stdout_buffer.empty():
            try:
                self.stdout_buffer.get_nowait()
            except queue.Empty:
                break
        
        while not self.stderr_buffer.empty():
            try:
                self.stderr_buffer.get_nowait()
            except queue.Empty:
                break
    
    def set_callbacks(
        self,
        stdout_callback: Optional[Callable[[str], None]] = None,
        stderr_callback: Optional[Callable[[str], None]] = None,
        metrics_callback: Optional[Callable[[ExecutionMetrics], None]] = None,
        completion_callback: Optional[Callable[[ExecutionResult], None]] = None
    ):
        """Set callback functions"""
        self.stdout_callback = stdout_callback
        self.stderr_callback = stderr_callback
        self.metrics_callback = metrics_callback
        self.completion_callback = completion_callback


class ExecutionObserver:
    """
    Higher-level execution observer
    Tracks multiple executions and provides aggregated insights
    """
    
    def __init__(self):
        self.monitor = RuntimeMonitor()
        self.execution_history: List[ExecutionResult] = []
        self.error_patterns: Dict[str, int] = {}
        
        # Statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_execution_time = 0.0
    
    def execute_and_observe(
        self,
        command: str,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """Execute command and observe results"""
        result = self.monitor.execute(command, timeout, cwd, env)
        
        # Update statistics
        self.total_executions += 1
        self.total_execution_time += result.execution_time
        
        if result.state == ExecutionState.COMPLETED:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
        
        # Update error patterns
        if result.error_type:
            self.error_patterns[result.error_type] = self.error_patterns.get(result.error_type, 0) + 1
        
        # Add to history
        self.execution_history.append(result)
        
        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        success_rate = (self.successful_executions / self.total_executions * 100) if self.total_executions > 0 else 0
        avg_execution_time = self.total_execution_time / self.total_executions if self.total_executions > 0 else 0
        
        return {
            "total_executions": self.total_executions,
            "successful": self.successful_executions,
            "failed": self.failed_executions,
            "success_rate": success_rate,
            "average_time": avg_execution_time,
            "total_time": self.total_execution_time,
            "common_errors": dict(sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def get_recent_executions(self, limit: int = 10) -> List[ExecutionResult]:
        """Get recent execution results"""
        return self.execution_history[-limit:] if self.execution_history else []
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Analyze error patterns"""
        if not self.error_patterns:
            return {"message": "No errors recorded"}
        
        # Group by error type
        error_groups = {}
        for result in self.execution_history:
            if result.error_type:
                if result.error_type not in error_groups:
                    error_groups[result.error_type] = []
                error_groups[result.error_type].append(result)
        
        # Analyze each error type
        analysis = {}
        for error_type, results in error_groups.items():
            count = len(results)
            last_occurrence = max(r.execution_time for r in results)
            commands = list(set(r.command for r in results))
            
            analysis[error_type] = {
                "count": count,
                "frequency": count / len(self.execution_history),
                "last_occurrence": last_occurrence,
                "example_commands": commands[:3],
                "common_solutions": self._suggest_solutions(error_type)
            }
        
        return analysis
    
    def _suggest_solutions(self, error_type: str) -> List[str]:
        """Suggest solutions for common errors"""
        solutions = {
            "import_error": [
                "Install missing package: pip install <package>",
                "Check PYTHONPATH environment variable",
                "Verify module name spelling"
            ],
            "syntax_error": [
                "Check for missing parentheses, brackets, or quotes",
                "Verify indentation (4 spaces per level)",
                "Look for typos in keywords"
            ],
            "file_not_found": [
                "Check file path spelling and case",
                "Verify file exists in expected location",
                "Check file permissions"
            ],
            "permission_error": [
                "Run with appropriate permissions",
                "Check file/directory permissions",
                "Use virtual environment for Python packages"
            ],
            "timeout_error": [
                "Increase timeout limit",
                "Optimize code for better performance",
                "Check for infinite loops"
            ],
            "command_not_found": [
                "Install required command/package",
                "Check PATH environment variable",
                "Verify command name spelling"
            ]
        }
        
        return solutions.get(error_type, ["Review error details for specific solution"])
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
        self.error_patterns.clear()
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_execution_time = 0.0

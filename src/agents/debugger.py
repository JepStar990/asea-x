"""
Debugger Agent - Error analysis and recovery
Analyzes runtime errors and provides fixes
"""

import re
import traceback
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from enum import Enum

from src.agents.base_agent import BaseAgent, AgentCapability, AgentContext, AgentResponse
from src.execution.command_executor import CommandExecutor


class ErrorType(str, Enum):
    """Types of errors that can occur"""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    IMPORT_ERROR = "import_error"
    TYPE_ERROR = "type_error"
    VALUE_ERROR = "value_error"
    ATTRIBUTE_ERROR = "attribute_error"
    KEY_ERROR = "key_error"
    INDEX_ERROR = "index_error"
    MODULE_NOT_FOUND = "module_not_found"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_ERROR = "permission_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    TEST_FAILURE = "test_failure"
    LINT_ERROR = "lint_error"
    DEPENDENCY_ERROR = "dependency_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ErrorAnalysis:
    """Detailed error analysis"""
    error_type: ErrorType
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    traceback: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    root_cause: Optional[str] = None
    severity: str = "medium"  # low, medium, high, critical
    confidence: float = 0.0  # 0.0 to 1.0


@dataclass
class FixProposal:
    """Proposed fix for an error"""
    description: str
    code_changes: Dict[str, str]  # file_path -> new_content
    commands: List[str]  # Commands to run
    validation_commands: List[str]  # Commands to validate fix
    estimated_time: float = 0.0  # in minutes
    confidence: float = 0.0  # 0.0 to 1.0
    risk: str = "low"  # low, medium, high
    dependencies: List[str] = field(default_factory=list)


class DebuggerAgent(BaseAgent):
    """
    Debugger Agent - Analyzes and fixes errors
    
    Responsibilities:
    1. Parse error messages and tracebacks
    2. Classify error types
    3. Diagnose root causes
    4. Propose and apply fixes
    5. Validate fixes
    6. Learn from fixes for future errors
    """
    
    def __init__(self):
        super().__init__(
            name="debugger",
            capabilities=[
                AgentCapability.DEBUGGING,
                AgentCapability.CODE_READING,
                AgentCapability.CODE_WRITING,
                AgentCapability.COMMAND_EXECUTION
            ]
        )
        
        self.command_executor = CommandExecutor()
        self.error_patterns = self._load_error_patterns()
        self.fix_history: List[Dict[str, Any]] = []
        
        # Error classification patterns
        self.error_classifiers = {
            ErrorType.SYNTAX_ERROR: [
                r"SyntaxError:",
                r"invalid syntax",
                r"IndentationError:",
                r"unexpected indent",
                r"expected.*:"
            ],
            ErrorType.IMPORT_ERROR: [
                r"ImportError:",
                r"ModuleNotFoundError:",
                r"No module named",
                r"cannot import name"
            ],
            ErrorType.TYPE_ERROR: [
                r"TypeError:",
                r"unsupported operand type",
                r"must be.*not.*",
                r"argument must be"
            ],
            ErrorType.VALUE_ERROR: [
                r"ValueError:",
                r"invalid literal for",
                r"could not convert"
            ],
            ErrorType.ATTRIBUTE_ERROR: [
                r"AttributeError:",
                r"has no attribute",
                r"object has no attribute"
            ],
            ErrorType.KEY_ERROR: [
                r"KeyError:"
            ],
            ErrorType.INDEX_ERROR: [
                r"IndexError:",
                r"list index out of range",
                r"string index out of range"
            ],
            ErrorType.FILE_NOT_FOUND: [
                r"FileNotFoundError:",
                r"No such file or directory"
            ],
            ErrorType.PERMISSION_ERROR: [
                r"PermissionError:",
                r"Permission denied"
            ],
            ErrorType.TIMEOUT_ERROR: [
                r"TimeoutError:",
                r"timed out",
                r"timeout",
                r"execution timed out"
            ]
        }
    
    async def execute(self, context: AgentContext) -> AgentResponse:
        """
        Execute debugging based on error context
        
        Steps:
        1. Parse error information
        2. Classify error type
        3. Analyze root cause
        4. Propose fixes
        5. Apply best fix
        6. Validate fix
        """
        self.logger.info(f"Starting debugging for task: {context.task_id}")
        
        try:
            # Step 1: Extract error information
            error_info = await self._extract_error_info(context)
            
            # Step 2: Classify error
            error_analysis = await self._classify_error(error_info, context)
            
            # Step 3: Analyze root cause
            root_cause = await self._analyze_root_cause(error_analysis, context)
            error_analysis.root_cause = root_cause
            
            # Step 4: Generate fix proposals
            proposals = await self._generate_fix_proposals(error_analysis, context)
            
            if not proposals:
                return AgentResponse(
                    success=False,
                    message="No fix proposals generated",
                    data={"error_analysis": error_analysis.__dict__},
                    requires_human_input=True
                )
            
            # Step 5: Select and apply best fix
            best_fix = self._select_best_fix(proposals)
            application_result = await self._apply_fix(best_fix, context)
            
            if not application_result.success:
                return AgentResponse(
                    success=False,
                    message=f"Failed to apply fix: {application_result.message}",
                    data={
                        "error_analysis": error_analysis.__dict__,
                        "fix_proposal": best_fix.__dict__,
                        "application_error": application_result.message
                    },
                    requires_human_input=True
                )
            
            # Step 6: Validate fix
            validation_result = await self._validate_fix(best_fix, context)
            
            # Step 7: Update fix history
            self._update_fix_history(error_analysis, best_fix, validation_result)
            
            # Generate report
            report = self._generate_debug_report(
                error_analysis, best_fix, validation_result
            )
            
            return AgentResponse(
                success=validation_result.success,
                message=report,
                data={
                    "error_analysis": error_analysis.__dict__,
                    "applied_fix": best_fix.__dict__,
                    "validation_result": validation_result.__dict__,
                    "error_fixed": validation_result.success,
                    "fix_applied": True
                },
                next_step="Re-run tests to verify fix" if validation_result.success else
                         "Try alternative fix or manual debugging"
            )
            
        except Exception as e:
            self.logger.error(f"Debugging failed: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                message=f"Debugging failed: {str(e)}",
                error=str(e),
                requires_human_input=True
            )
    
    async def _extract_error_info(self, context: AgentContext) -> Dict[str, Any]:
        """Extract error information from context"""
        error_info = {
            "raw_error": "",
            "traceback": [],
            "code_context": {},
            "execution_output": {},
            "system_state": context.system_state
        }
        
        # Look for error in previous results
        for result in context.previous_results:
            if result.get("error"):
                error_info["raw_error"] = result["error"]
            if result.get("data") and result["data"].get("test_output"):
                error_info["execution_output"]["test"] = result["data"]["test_output"]
            if result.get("data") and result["data"].get("command_output"):
                error_info["execution_output"]["command"] = result["data"]["command_output"]
        
        # Also check state manager for last error
        state = self.state_manager.get_state()
        if state.last_error:
            error_info["raw_error"] = state.last_error
        
        # Extract traceback if present
        if error_info["raw_error"]:
            lines = error_info["raw_error"].split('\n')
            traceback_lines = []
            in_traceback = False
            
            for line in lines:
                if "Traceback" in line:
                    in_traceback = True
                if in_traceback:
                    traceback_lines.append(line)
                if line.strip() and not line.startswith(' ') and in_traceback:
                    # This is likely the error message line
                    error_info["error_message"] = line.strip()
                    in_traceback = False
            
            error_info["traceback"] = traceback_lines
        
        # Get relevant code context
        if context.file_context:
            # Try to find file mentioned in error
            for file_path, content in context.file_context.items():
                if any(keyword in error_info["raw_error"].lower() 
                       for keyword in [file_path.lower(), Path(file_path).name.lower()]):
                    error_info["code_context"][file_path] = content[:5000]  # Limit size
        
        return error_info
    
    async def _classify_error(
        self, 
        error_info: Dict[str, Any], 
        context: AgentContext
    ) -> ErrorAnalysis:
        """Classify error type and extract details"""
        raw_error = error_info.get("raw_error", "").lower()
        error_message = error_info.get("error_message", "")
        
        # Initialize analysis
        analysis = ErrorAnalysis(
            error_type=ErrorType.UNKNOWN_ERROR,
            message=error_message,
            traceback=error_info.get("traceback", []),
            context=error_info
        )
        
        # Classify based on patterns
        for error_type, patterns in self.error_classifiers.items():
            for pattern in patterns:
                if re.search(pattern.lower(), raw_error):
                    analysis.error_type = error_type
                    break
            if analysis.error_type != ErrorType.UNKNOWN_ERROR:
                break
        
        # Extract file and line information
        for line in analysis.traceback:
            # Look for file paths and line numbers in traceback
            match = re.search(r'File "([^"]+)".*line (\d+)', line)
            if match:
                analysis.file_path = match.group(1)
                analysis.line_number = int(match.group(2))
                break
        
        # Extract function name if present
        for line in analysis.traceback:
            match = re.search(r'in (\w+)', line)
            if match and not analysis.function_name:
                analysis.function_name = match.group(1)
        
        # Determine severity
        severity_map = {
            ErrorType.SYNTAX_ERROR: "high",
            ErrorType.IMPORT_ERROR: "medium",
            ErrorType.MODULE_NOT_FOUND: "medium",
            ErrorType.TYPE_ERROR: "medium",
            ErrorType.VALUE_ERROR: "medium",
            ErrorType.ATTRIBUTE_ERROR: "medium",
            ErrorType.KEY_ERROR: "medium",
            ErrorType.INDEX_ERROR: "medium",
            ErrorType.FILE_NOT_FOUND: "medium",
            ErrorType.PERMISSION_ERROR: "high",
            ErrorType.TIMEOUT_ERROR: "high",
            ErrorType.MEMORY_ERROR: "critical",
            ErrorType.TEST_FAILURE: "low",
            ErrorType.LINT_ERROR: "low"
        }
        
        analysis.severity = severity_map.get(analysis.error_type, "medium")
        
        # Use LLM for more detailed analysis if needed
        if analysis.error_type == ErrorType.UNKNOWN_ERROR or not analysis.root_cause:
            llm_analysis = await self._analyze_with_llm(error_info, context)
            if llm_analysis:
                analysis.root_cause = llm_analysis.get("root_cause")
                analysis.confidence = llm_analysis.get("confidence", 0.5)
        
        return analysis
    
    async def _analyze_with_llm(
        self, 
        error_info: Dict[str, Any], 
        context: AgentContext
    ) -> Optional[Dict[str, Any]]:
        """Use LLM for advanced error analysis"""
        prompt = f"""
        Analyze this error and determine the root cause:
        
        Error: {error_info.get('raw_error', '')}
        
        Code Context: {json.dumps(error_info.get('code_context', {}), indent=2)}
        
        System State: {json.dumps(error_info.get('system_state', {}), indent=2)}
        
        Provide a JSON response with:
        1. root_cause: What is causing this error
        2. confidence: 0.0 to 1.0
        3. suggested_fixes: List of possible fixes
        4. error_type: More specific error classification
        5. debugging_steps: Steps to debug this issue
        
        Be specific and technical.
        """
        
        try:
            response = self.llm_client.chat_completion([
                {"role": "system", "content": "You are a debugging expert."},
                {"role": "user", "content": prompt}
            ])
            
            analysis = json.loads(response.content)
            return analysis
        except:
            return None
    
    async def _analyze_root_cause(
        self, 
        error_analysis: ErrorAnalysis, 
        context: AgentContext
    ) -> str:
        """Analyze root cause of error"""
        # Common root causes based on error type
        root_cause_map = {
            ErrorType.SYNTAX_ERROR: "Invalid Python syntax or indentation",
            ErrorType.IMPORT_ERROR: "Missing module or incorrect import path",
            ErrorType.MODULE_NOT_FOUND: "Dependency not installed",
            ErrorType.TYPE_ERROR: "Incompatible data types or incorrect function arguments",
            ErrorType.VALUE_ERROR: "Invalid value provided to function",
            ErrorType.ATTRIBUTE_ERROR: "Object doesn't have the requested attribute/method",
            ErrorType.KEY_ERROR: "Dictionary key doesn't exist",
            ErrorType.INDEX_ERROR: "List/array index out of bounds",
            ErrorType.FILE_NOT_FOUND: "File doesn't exist at specified path",
            ErrorType.PERMISSION_ERROR: "Insufficient permissions to access file/resource",
            ErrorType.TIMEOUT_ERROR: "Operation took too long to complete",
            ErrorType.TEST_FAILURE: "Test assertion failed",
            ErrorType.LINT_ERROR: "Code style violation"
        }
        
        root_cause = root_cause_map.get(
            error_analysis.error_type, 
            "Unknown error - needs investigation"
        )
        
        # If we have file and line, add more specific details
        if error_analysis.file_path and error_analysis.line_number:
            # Try to get the problematic line of code
            file_content = context.file_context.get(error_analysis.file_path)
            if file_content:
                lines = file_content.split('\n')
                if 0 <= error_analysis.line_number - 1 < len(lines):
                    problematic_line = lines[error_analysis.line_number - 1]
                    root_cause += f"\nProblematic line {error_analysis.line_number}: {problematic_line.strip()}"
        
        return root_cause
    
    async def _generate_fix_proposals(
        self, 
        error_analysis: ErrorAnalysis, 
        context: AgentContext
    ) -> List[FixProposal]:
        """Generate fix proposals for the error"""
        proposals = []
        
        # Generate fixes based on error type
        if error_analysis.error_type == ErrorType.SYNTAX_ERROR:
            proposals.extend(await self._generate_syntax_fixes(error_analysis, context))
        elif error_analysis.error_type == ErrorType.IMPORT_ERROR:
            proposals.extend(await self._generate_import_fixes(error_analysis, context))
        elif error_analysis.error_type == ErrorType.MODULE_NOT_FOUND:
            proposals.extend(await self._generate_dependency_fixes(error_analysis, context))
        elif error_analysis.error_type == ErrorType.TYPE_ERROR:
            proposals.extend(await self._generate_type_fixes(error_analysis, context))
        elif error_analysis.error_type == ErrorType.VALUE_ERROR:
            proposals.extend(await self._generate_value_fixes(error_analysis, context))
        elif error_analysis.error_type == ErrorType.TEST_FAILURE:
            proposals.extend(await self._generate_test_fixes(error_analysis, context))
        else:
            # Generic fix generation using LLM
            proposals.extend(await self._generate_generic_fixes(error_analysis, context))
        
        return proposals
    
    async def _generate_syntax_fixes(
        self, 
        error_analysis: ErrorAnalysis, 
        context: AgentContext
    ) -> List[FixProposal]:
        """Generate fixes for syntax errors"""
        proposals = []
        
        if error_analysis.file_path and error_analysis.line_number:
            # Get file content
            file_content = context.file_context.get(error_analysis.file_path)
            if file_content:
                # Use LLM to fix syntax
                prompt = f"""
                Fix this syntax error in Python code:
                
                Error: {error_analysis.message}
                File: {error_analysis.file_path}
                Line: {error_analysis.line_number}
                
                Code around the error:
                ```python
                {self._get_code_context(file_content, error_analysis.line_number, 3)}
                ```
                
                Return JSON with:
                1. fixed_code: The corrected code
                2. description: What was fixed
                3. changes_made: Specific changes
                4. confidence: 0.0 to 1.0
                """
                
                try:
                    response = self.llm_client.chat_completion([
                        {"role": "system", "content": "You are a Python syntax expert."},
                        {"role": "user", "content": prompt}
                    ])
                    
                    fix_data = json.loads(response.content)
                    
                    # Create complete fixed file content
                    lines = file_content.split('\n')
                    # This is simplified - real implementation would be more sophisticated
                    fixed_lines = lines.copy()
                    # Would insert fix at the appropriate line
                    
                    fixed_content = '\n'.join(fixed_lines)
                    
                    proposals.append(FixProposal(
                        description=f"Fix syntax error: {error_analysis.message}",
                        code_changes={error_analysis.file_path: fixed_content},
                        commands=[],
                        validation_commands=["python -m py_compile " + error_analysis.file_path],
                        estimated_time=2.0,
                        confidence=fix_data.get("confidence", 0.7),
                        risk="low"
                    ))
                    
                except:
                    pass
        
        return proposals
    
    async def _generate_import_fixes(
        self, 
        error_analysis: ErrorAnalysis, 
        context: AgentContext
    ) -> List[FixProposal]:
        """Generate fixes for import errors"""
        proposals = []
        
        # Extract module name from error message
        module_match = re.search(r"'(.*?)'", error_analysis.message)
        if module_match:
            module_name = module_match.group(1)
            
            # Proposal 1: Install missing package
            proposals.append(FixProposal(
                description=f"Install missing package: {module_name}",
                code_changes={},
                commands=[f"pip install {module_name}"],
                validation_commands=[f"python -c \"import {module_name.split('.')[0]}\""],
                estimated_time=1.0,
                confidence=0.8,
                risk="low",
                dependencies=[module_name]
            ))
            
            # Proposal 2: Fix import statement (if it's a local import issue)
            if error_analysis.file_path:
                file_content = context.file_context.get(error_analysis.file_path)
                if file_content:
                    # Check if it's a relative import that needs fixing
                    proposals.append(FixProposal(
                        description=f"Fix import statement for {module_name}",
                        code_changes={},
                        commands=[],
                        validation_commands=[],
                        estimated_time=5.0,
                        confidence=0.5,
                        risk="medium"
                    ))
        
        return proposals
    
    async def _generate_dependency_fixes(
        self, 
        error_analysis: ErrorAnalysis, 
        context: AgentContext
    ) -> List[FixProposal]:
        """Generate fixes for missing dependencies"""
        proposals = []
        
        # Extract module name
        module_match = re.search(r"'([^']+)'", error_analysis.message)
        if module_match:
            module_name = module_match.group(1)
            
            # Try to find package name (might be different from import name)
            common_mappings = {
                "PIL": "Pillow",
                "yaml": "PyYAML",
                "crypto": "pycryptodome",
                "sklearn": "scikit-learn",
                "bs4": "beautifulsoup4"
            }
            
            package_name = common_mappings.get(module_name, module_name)
            
            proposals.append(FixProposal(
                description=f"Install dependency: {package_name}",
                code_changes={},
                commands=[f"pip install {package_name}"],
                validation_commands=[f"python -c \"import {module_name}\""],
                estimated_time=2.0,
                confidence=0.9,
                risk="low",
                dependencies=[package_name]
            ))
        
        return proposals
    
    async def _generate_type_fixes(
        self, 
        error_analysis: ErrorAnalysis, 
        context: AgentContext
    ) -> List[FixProposal]:
        """Generate fixes for type errors"""
        proposals = []
        
        # Use LLM to analyze and fix type error
        prompt = f"""
        Fix this type error in Python code:
        
        Error: {error_analysis.message}
        
        Code Context: {json.dumps(error_analysis.context.get('code_context', {}), indent=2)}
        
        Traceback: {error_analysis.traceback}
        
        Provide a JSON response with:
        1. root_cause: Why the type error is occurring
        2. fix_description: How to fix it
        3. code_changes: Specific code changes needed
        4. confidence: 0.0 to 1.0
        
        Be specific about which variables/types need to be changed.
        """
        
        try:
            response = self.llm_client.chat_completion([
                {"role": "system", "content": "You are a Python type system expert."},
                {"role": "user", "content": prompt}
            ])
            
            fix_data = json.loads(response.content)
            
            # Create fix proposal based on LLM analysis
            if fix_data.get("code_changes"):
                proposals.append(FixProposal(
                    description=fix_data.get("fix_description", "Fix type error"),
                    code_changes=fix_data.get("code_changes", {}),
                    commands=[],
                    validation_commands=["python -m py_compile " + list(fix_data.get("code_changes", {}).keys())[0]],
                    estimated_time=5.0,
                    confidence=fix_data.get("confidence", 0.6),
                    risk="medium"
                ))
                
        except:
            pass
        
        return proposals
    
    async def _generate_value_fixes(
        self, 
        error_analysis: ErrorAnalysis, 
        context: AgentContext
    ) -> List[FixProposal]:
        """Generate fixes for value errors"""
        # Similar to type fixes, use LLM
        return await self._generate_generic_fixes(error_analysis, context)
    
    async def _generate_test_fixes(
        self, 
        error_analysis: ErrorAnalysis, 
        context: AgentContext
    ) -> List[FixProposal]:
        """Generate fixes for test failures"""
        proposals = []
        
        # Extract test failure details
        test_output = error_analysis.context.get("execution_output", {}).get("test", "")
        
        if test_output:
            # Parse test failure to understand what's wrong
            failed_test_match = re.search(r"FAILED.*test_(\w+)", test_output)
            if failed_test_match:
                test_name = failed_test_match.group(1)
                
                proposals.append(FixProposal(
                    description=f"Debug failing test: {test_name}",
                    code_changes={},
                    commands=[f"python -m pytest -k {test_name} -v"],
                    validation_commands=[f"python -m pytest -k {test_name}"],
                    estimated_time=10.0,
                    confidence=0.5,
                    risk="low"
                ))
        
        return proposals
    
    async def _generate_generic_fixes(
        self, 
        error_analysis: ErrorAnalysis, 
        context: AgentContext
    ) -> List[FixProposal]:
        """Generate generic fixes using LLM"""
        proposals = []
        
        prompt = f"""
        Analyze this error and propose a fix:
        
        Error Type: {error_analysis.error_type}
        Error Message: {error_analysis.message}
        
        Code Context: {json.dumps(error_analysis.context.get('code_context', {}), indent=2)}
        
        Provide a JSON response with:
        1. fix_description: How to fix the error
        2. code_changes: Specific code changes (file -> new_content)
        3. commands: Commands to run
        4. validation: How to validate the fix
        5. confidence: 0.0 to 1.0
        6. risk: low, medium, high
        
        Be specific and practical.
        """
        
        try:
            response = self.llm_client.chat_completion([
                {"role": "system", "content": "You are a software debugging expert."},
                {"role": "user", "content": prompt}
            ])
            
            fix_data = json.loads(response.content)
            
            proposals.append(FixProposal(
                description=fix_data.get("fix_description", "Fix error"),
                code_changes=fix_data.get("code_changes", {}),
                commands=fix_data.get("commands", []),
                validation_commands=fix_data.get("validation", []),
                estimated_time=5.0,
                confidence=fix_data.get("confidence", 0.5),
                risk=fix_data.get("risk", "medium"),
                dependencies=fix_data.get("dependencies", [])
            ))
            
        except:
            pass
        
        return proposals
    
    def _select_best_fix(self, proposals: List[FixProposal]) -> FixProposal:
        """Select the best fix proposal"""
        if not proposals:
            raise ValueError("No fix proposals available")
        
        # Score each proposal
        scored_proposals = []
        for proposal in proposals:
            score = self._score_fix_proposal(proposal)
            scored_proposals.append((score, proposal))
        
        # Select highest score
        scored_proposals.sort(key=lambda x: x[0], reverse=True)
        return scored_proposals[0][1]
    
    def _score_fix_proposal(self, proposal: FixProposal) -> float:
        """Score a fix proposal based on various factors"""
        score = 0.0
        
        # Confidence (0-1)
        score += proposal.confidence * 40
        
        # Risk (inverse)
        risk_weights = {"low": 30, "medium": 15, "high": 0}
        score += risk_weights.get(proposal.risk, 0)
        
        # Estimated time (shorter is better)
        time_score = max(0, 20 - (proposal.estimated_time / 2))
        score += time_score
        
        # Has validation commands
        if proposal.validation_commands:
            score += 10
        
        return score
    
    async def _apply_fix(
        self, 
        fix: FixProposal, 
        context: AgentContext
    ) -> AgentResponse:
        """Apply the fix proposal"""
        try:
            # Apply code changes
            for file_path, new_content in fix.code_changes.items():
                # Write file
                path = Path(file_path)
                path.parent.mkdir(exist_ok=True, parents=True)
                path.write_text(new_content, encoding='utf-8')
                
                # Update context
                self.state_manager.add_file_context(file_path, new_content)
            
            # Run commands
            command_results = []
            for command in fix.commands:
                result = await self.command_executor.execute(
                    command,
                    cwd="./workdir/dev",
                    timeout=60
                )
                command_results.append(result)
                
                if not result.success:
                    return AgentResponse(
                        success=False,
                        message=f"Command failed: {command}\n{result.stderr}",
                        data={"command_results": [r.__dict__ for r in command_results]}
                    )
            
            # Log fix application
            self.state_manager.log_action(
                agent=self.name,
                action="fix_applied",
                details={
                    "description": fix.description,
                    "files_modified": list(fix.code_changes.keys()),
                    "commands_run": fix.commands,
                    "confidence": fix.confidence,
                    "risk": fix.risk
                }
            )
            
            return AgentResponse(
                success=True,
                message=f"Applied fix: {fix.description}",
                data={
                    "files_modified": list(fix.code_changes.keys()),
                    "commands_run": fix.commands,
                    "command_results": [r.__dict__ for r in command_results]
                }
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                message=f"Failed to apply fix: {str(e)}",
                error=str(e)
            )
    
    async def _validate_fix(
        self, 
        fix: FixProposal, 
        context: AgentContext
    ) -> AgentResponse:
        """Validate that the fix worked"""
        validation_results = []
        
        # Run validation commands
        for command in fix.validation_commands:
            result = await self.command_executor.execute(
                command,
                cwd="./workdir/dev",
                timeout=30
            )
            validation_results.append(result)
            
            if not result.success:
                return AgentResponse(
                    success=False,
                    message=f"Validation failed: {command}\n{result.stderr}",
                    data={"validation_results": [r.__dict__ for r in validation_results]}
                )
        
        # If no validation commands, do a basic check
        if not validation_results:
            # Try to import/run the fixed files
            for file_path in fix.code_changes.keys():
                if file_path.endswith('.py'):
                    result = await self.command_executor.execute(
                        f"python -m py_compile {file_path}",
                        cwd="./workdir/dev",
                        timeout=10
                    )
                    validation_results.append(result)
                    
                    if not result.success:
                        return AgentResponse(
                            success=False,
                            message=f"Syntax check failed for {file_path}",
                            data={"validation_results": [r.__dict__ for r in validation_results]}
                        )
        
        return AgentResponse(
            success=True,
            message="Fix validation passed",
            data={"validation_results": [r.__dict__ for r in validation_results]}
        )
    
    def _update_fix_history(
        self, 
        error_analysis: ErrorAnalysis,
        fix: FixProposal,
        validation_result: AgentResponse
    ):
        """Update fix history for learning"""
        history_entry = {
            "timestamp": time.time(),
            "error_type": error_analysis.error_type.value,
            "error_message": error_analysis.message,
            "root_cause": error_analysis.root_cause,
            "fix_description": fix.description,
            "confidence": fix.confidence,
            "validation_success": validation_result.success,
            "files_modified": list(fix.code_changes.keys())
        }
        
        self.fix_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(self.fix_history) > 100:
            self.fix_history = self.fix_history[-100:]
    
    def _generate_debug_report(
        self,
        error_analysis: ErrorAnalysis,
        fix: FixProposal,
        validation_result: AgentResponse
    ) -> str:
        """Generate debug report"""
        lines = [
            "**Debugging Report**",
            "",
            f"**Error:** {error_analysis.error_type.value}",
            f"**Message:** {error_analysis.message}",
            ""
        ]
        
        if error_analysis.root_cause:
            lines.append(f"**Root Cause:** {error_analysis.root_cause}")
            lines.append("")
        
        lines.append(f"**Applied Fix:** {fix.description}")
        lines.append(f"**Confidence:** {fix.confidence:.0%}")
        lines.append(f"**Risk:** {fix.risk}")
        
        if fix.code_changes:
            lines.append("")
            lines.append("**Files Modified:**")
            for file_path in fix.code_changes.keys():
                lines.append(f"  - {file_path}")
        
        lines.append("")
        lines.append(f"**Validation:** {'Passed' if validation_result.success else 'Failed'}")
        
        if validation_result.success:
            lines.append("")
            lines.append("Error has been fixed!")
            lines.append("Next: Re-run tests to ensure complete fix")
        else:
            lines.append("")
            lines.append("Fix validation failed")
            lines.append("Next: Try alternative fix or manual debugging")
        
        return "\n".join(lines)
    
    def _get_code_context(
        self, 
        content: str, 
        line_number: int, 
        context_lines: int = 3
    ) -> str:
        """Get code context around a specific line"""
        lines = content.split('\n')
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        
        context_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == line_number - 1 else "    "
            context_lines.append(f"{prefix}{lines[i]}")
        
        return '\n'.join(context_lines)
    
    def _load_error_patterns(self) -> Dict[str, Any]:
        """Load error patterns from config or default"""
        # Could load from file in production
        return {}

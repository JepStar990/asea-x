"""
Developer Agent - Code execution engine
Writes, modifies, and executes code
"""

import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import shutil
import logging

from src.agents.base_agent import BaseAgent, AgentCapability, AgentContext, AgentResponse
from src.execution.command_executor import CommandExecutor
from src.tools.file_manager import FileManager
from src.validation.lint_engine import LintEngine


class DeveloperAgent(BaseAgent):
    """
    Developer Agent - Writes and executes code
    
    Responsibilities:
    1. Write new code
    2. Modify existing code
    3. Run commands
    4. Install dependencies
    5. Test code
    """
    
    def __init__(self):
        super().__init__(
            name="developer",
            capabilities=[
                AgentCapability.CODE_WRITING,
                AgentCapability.CODE_READING,
                AgentCapability.COMMAND_EXECUTION,
                AgentCapability.FILE_OPERATIONS
            ]
        )
        
        self.command_executor = CommandExecutor()
        self.file_manager = FileManager()
        self.lint_engine = LintEngine()
        
        # Execution environment
        self.workdir = Path("./workdir/dev")
        self.workdir.mkdir(exist_ok=True, parents=True)
    
    async def execute(self, context: AgentContext) -> AgentResponse:
        """
        Execute development task
        
        Steps:
        1. Analyze what needs to be built
        2. Check dependencies
        3. Write/modify code
        4. Run tests
        5. Handle errors
        """
        self.logger.info(f"Starting development for task: {context.task_id}")
        
        try:
            # Step 1: Analyze task
            analysis = await self._analyze_task(context)
            
            # Step 2: Check and install dependencies if needed
            if analysis.get("requires_dependencies", False):
                dep_result = await self._handle_dependencies(analysis, context)
                if not dep_result.success:
                    return dep_result
            
            # Step 3: Write or modify code
            if analysis.get("action") == "write":
                result = await self._write_code(analysis, context)
            elif analysis.get("action") == "modify":
                result = await self._modify_code(analysis, context)
            elif analysis.get("action") == "test":
                result = await self._run_tests(analysis, context)
            else:
                result = await self._write_code(analysis, context)
            
            # Step 4: Run linting (but don't fail on it)
            if result.success and result.data.get("code_written", False):
                lint_result = await self._run_linting(context)
                if not lint_result.success:
                    result.metadata["lint_issues"] = lint_result.data.get("issues", [])
                    result.next_step = "Fix linting issues with /lint mode"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Development failed: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                message=f"Development failed: {str(e)}",
                error=str(e),
                requires_human_input=True,
                next_step="Switch to /debug mode to analyze error"
            )
    
    async def _analyze_task(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze what needs to be done"""
        prompt = f"""
        Analyze this development task and determine what needs to be done:
        
        Task: {context.task_description}
        
        Current Files in Context: {list(context.file_context.keys())}
        
        System Mode: {context.mode}
        
        Analyze and return JSON with:
        1. action: "write", "modify", "test", or "debug"
        2. target_file: Which file to work on (if known)
        3. language: Programming language
        4. requirements: What specifically needs to be implemented
        5. requires_dependencies: True/False
        6. test_needed: True/False
        7. estimated_complexity: "simple", "moderate", "complex"
        8. existing_code: Any relevant existing code from context
        """
        
        response = self.llm_client.chat_completion([
            {"role": "system", "content": "You are a code analysis assistant."},
            {"role": "user", "content": prompt}
        ])
        
        try:
            analysis = json.loads(response.content)
            
            # Get existing code if target file exists
            if analysis.get("target_file"):
                existing_code = context.file_context.get(analysis["target_file"])
                if existing_code:
                    analysis["existing_code"] = existing_code
            
            return analysis
        except:
            # Default analysis
            return {
                "action": "write",
                "language": "python",
                "requirements": context.task_description,
                "requires_dependencies": True,
                "test_needed": True,
                "estimated_complexity": "moderate"
            }
    
    async def _handle_dependencies(
        self, 
        analysis: Dict[str, Any], 
        context: AgentContext
    ) -> AgentResponse:
        """Handle dependency installation"""
        self.logger.info("Checking dependencies...")
        
        # Determine dependencies based on language and task
        language = analysis.get("language", "python")
        deps = await self._determine_dependencies(language, analysis)
        
        if not deps:
            return AgentResponse(
                success=True,
                message="No dependencies needed",
                data={"dependencies_installed": []}
            )
        
        # Check safety before installing
        for dep in deps:
            if language == "python":
                cmd = f"pip install {dep}"
            elif language == "node":
                cmd = f"npm install {dep}"
            elif language == "rust":
                cmd = f"cargo add {dep}"
            else:
                continue
            
            safety_check = self.state_manager.safety_system.check_command(cmd)
            if not safety_check.allowed:
                return AgentResponse(
                    success=False,
                    message=f"Cannot install {dep}: {safety_check.reason}",
                    requires_human_input=True,
                    next_step=f"Use /unsafe on to enable installation"
                )
        
        # Install dependencies
        installed = []
        failed = []
        
        for dep in deps:
            try:
                if language == "python":
                    result = subprocess.run(
                        ["pip", "install", dep],
                        capture_output=True,
                        text=True,
                        cwd=self.workdir
                    )
                elif language == "node":
                    result = subprocess.run(
                        ["npm", "install", dep],
                        capture_output=True,
                        text=True,
                        cwd=self.workdir
                    )
                else:
                    self.logger.warning(f"Unknown language for dependency: {language}")
                    continue
                
                if result.returncode == 0:
                    installed.append(dep)
                    
                    # Log installation
                    self.state_manager.log_action(
                        agent=self.name,
                        action="dependency_install",
                        details={
                            "dependency": dep,
                            "language": language,
                            "success": True
                        }
                    )
                else:
                    failed.append(f"{dep}: {result.stderr[:200]}")
                    
            except Exception as e:
                failed.append(f"{dep}: {str(e)}")
        
        if failed:
            return AgentResponse(
                success=False,
                message=f"Failed to install some dependencies: {', '.join(failed)}",
                data={"installed": installed, "failed": failed},
                requires_human_input=True
            )
        
        return AgentResponse(
            success=True,
            message=f"Installed dependencies: {', '.join(installed)}",
            data={"dependencies_installed": installed}
        )
    
    async def _determine_dependencies(
        self, 
        language: str, 
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Determine required dependencies"""
        prompt = f"""
        Determine what dependencies are needed for this task:
        
        Language: {language}
        Task: {analysis.get('requirements')}
        Complexity: {analysis.get('estimated_complexity')}
        
        Return a JSON list of dependency names (just names, no versions).
        Only include essential dependencies for the task.
        For Python, include test frameworks if test_needed is true.
        """
        
        response = self.llm_client.chat_completion([
            {"role": "system", "content": "You are a dependency analysis assistant."},
            {"role": "user", "content": prompt}
        ])
        
        try:
            deps = json.loads(response.content)
            return deps if isinstance(deps, list) else []
        except:
            # Default dependencies
            defaults = {
                "python": ["pytest"] if analysis.get("test_needed") else [],
                "node": [],
                "rust": []
            }
            return defaults.get(language, [])
    
    async def _write_code(self, analysis: Dict[str, Any], context: AgentContext) -> AgentResponse:
        """Write new code"""
        prompt = f"""
        Write code for the following task:
        
        Task: {analysis.get('requirements')}
        Language: {analysis.get('language', 'python')}
        
        Requirements:
        {json.dumps(analysis.get('specific_requirements', {}), indent=2)}
        
        Additional context from system:
        {json.dumps(context.system_state.get('file_context_summary', {}), indent=2)}
        
        Write complete, working code.
        Include proper imports.
        Add docstrings/comments.
        Make it production-ready.
        
        Return JSON with:
        1. code: The complete code
        2. filename: Suggested filename
        3. description: What the code does
        4. imports: List of imports used
        5. functions: List of functions/classes defined
        """
        
        response = self.llm_client.chat_completion([
            {"role": "system", "content": "You are an expert software developer."},
            {"role": "user", "content": prompt}
        ])
        
        try:
            code_info = json.loads(response.content)
            code = code_info.get("code", "")
            filename = code_info.get("filename", "new_code.py")
            
            # Write file
            file_path = self.workdir / filename
            file_path.parent.mkdir(exist_ok=True, parents=True)
            file_path.write_text(code, encoding="utf-8")
            
            # Add to context
            self.state_manager.add_file_context(str(file_path), code)
            
            # Log action
            self.state_manager.log_action(
                agent=self.name,
                action="code_written",
                details={
                    "filename": filename,
                    "size": len(code),
                    "language": analysis.get("language"),
                    "functions": code_info.get("functions", [])
                }
            )
            
            return AgentResponse(
                success=True,
                message=f"Wrote code to {filename}",
                data={
                    "code": code,
                    "filename": filename,
                    "code_written": True,
                    "functions": code_info.get("functions", []),
                    "imports": code_info.get("imports", [])
                },
                next_step="Test the code or switch to /lint for quality check"
            )
            
        except Exception as e:
            self.logger.error(f"Code writing failed: {e}")
            return AgentResponse(
                success=False,
                message=f"Failed to write code: {str(e)}",
                error=str(e)
            )
    
    async def _modify_code(self, analysis: Dict[str, Any], context: AgentContext) -> AgentResponse:
        """Modify existing code"""
        target_file = analysis.get("target_file")
        if not target_file:
            return AgentResponse(
                success=False,
                message="No target file specified for modification",
                requires_human_input=True
            )
        
        existing_code = analysis.get("existing_code")
        if not existing_code:
            existing_code = self.state_manager.get_file_context(target_file)
        
        if not existing_code:
            return AgentResponse(
                success=False,
                message=f"File {target_file} not found in context",
                requires_human_input=True
            )
        
        prompt = f"""
        Modify the following code according to the task:
        
        Task: {analysis.get('requirements')}
        
        Original code ({target_file}):
        ```{analysis.get('language', 'python')}
        {existing_code}
        ```
        
        Modification requirements:
        {json.dumps(analysis.get('modification_details', {}), indent=2)}
        
        Return JSON with:
        1. new_code: The modified code
        2. diff: Unified diff showing changes
        3. changes_made: Description of what was changed
        4. functions_modified: List of functions/classes modified
        5. functions_added: List of functions/classes added
        6. functions_removed: List of functions/classes removed
        """
        
        response = self.llm_client.chat_completion([
            {"role": "system", "content": "You are an expert code modifier."},
            {"role": "user", "content": prompt}
        ])
        
        try:
            modification = json.loads(response.content)
            new_code = modification.get("new_code", existing_code)
            diff = modification.get("diff", "")
            
            # Write file
            file_path = self.workdir / target_file
            file_path.parent.mkdir(exist_ok=True, parents=True)
            file_path.write_text(new_code, encoding="utf-8")
            
            # Update context
            self.state_manager.add_file_context(target_file, new_code)
            
            # Log action
            self.state_manager.log_action(
                agent=self.name,
                action="code_modified",
                details={
                    "filename": target_file,
                    "changes": modification.get("changes_made", ""),
                    "functions_modified": modification.get("functions_modified", []),
                    "functions_added": modification.get("functions_added", [])
                }
            )
            
            return AgentResponse(
                success=True,
                message=f"Modified {target_file}",
                data={
                    "new_code": new_code,
                    "diff": diff,
                    "filename": target_file,
                    "changes_made": modification.get("changes_made", ""),
                    "code_modified": True
                },
                next_step="Test the modifications or run /lint"
            )
            
        except Exception as e:
            self.logger.error(f"Code modification failed: {e}")
            return AgentResponse(
                success=False,
                message=f"Failed to modify code: {str(e)}",
                error=str(e)
            )
    
    async def _run_tests(self, analysis: Dict[str, Any], context: AgentContext) -> AgentResponse:
        """Run tests on code"""
        language = analysis.get("language", "python")
        
        try:
            if language == "python":
                # Look for test files
                test_files = list(self.workdir.glob("test_*.py")) + \
                            list(self.workdir.glob("*_test.py"))
                
                if not test_files:
                    # Create a simple test if none exists
                    test_result = await self._create_and_run_test(analysis, context)
                    return test_result
                
                # Run pytest
                result = subprocess.run(
                    ["python", "-m", "pytest", "-v"],
                    capture_output=True,
                    text=True,
                    cwd=self.workdir,
                    timeout=30
                )
                
                # Parse results
                if result.returncode == 0:
                    message = "✅ All tests passed!"
                    success = True
                else:
                    # Count failures
                    lines = result.stdout.split('\n')
                    failed = [l for l in lines if "FAILED" in l]
                    message = f"❌ Tests failed: {len(failed)} failures"
                    success = False
                
                return AgentResponse(
                    success=success,
                    message=message,
                    data={
                        "test_output": result.stdout,
                        "test_errors": result.stderr,
                        "returncode": result.returncode,
                        "tests_run": True,
                        "tests_passed": success
                    },
                    next_step="Debug failing tests with /debug mode" if not success else "Continue development"
                )
                
            else:
                return AgentResponse(
                    success=False,
                    message=f"Test running not implemented for {language}",
                    requires_human_input=True
                )
                
        except subprocess.TimeoutExpired:
            return AgentResponse(
                success=False,
                message="Tests timed out (30 seconds)",
                error="Timeout",
                requires_human_input=True
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                message=f"Test execution failed: {str(e)}",
                error=str(e)
            )
    
    async def _create_and_run_test(
        self, 
        analysis: Dict[str, Any], 
        context: AgentContext
    ) -> AgentResponse:
        """Create and run a test"""
        # Find Python files to test
        py_files = list(self.workdir.glob("*.py"))
        if not py_files:
            return AgentResponse(
                success=False,
                message="No Python files found to test",
                requires_human_input=True
            )
        
        # Create a simple test
        test_code = """
import pytest

def test_example():
    '''Example test - replace with actual tests'''
    assert True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
        
        test_file = self.workdir / "test_example.py"
        test_file.write_text(test_code)
        
        # Run the test
        result = subprocess.run(
            ["python", "-m", "pytest", str(test_file), "-v"],
            capture_output=True,
            text=True,
            cwd=self.workdir
        )
        
        return AgentResponse(
            success=result.returncode == 0,
            message=f"Ran example test: {'Passed' if result.returncode == 0 else 'Failed'}",
            data={
                "test_output": result.stdout,
                "test_created": True,
                "tests_passed": result.returncode == 0
            },
            next_step="Write actual tests for your code"
        )
    
    async def _run_linting(self, context: AgentContext) -> AgentResponse:
        """Run linting on code"""
        try:
            # Run lint engine
            result = self.lint_engine.lint_directory(self.workdir)
            
            if result["issues"]:
                return AgentResponse(
                    success=False,
                    message=f"Found {len(result['issues'])} linting issues",
                    data=result,
                    next_step="Fix linting issues with /lint mode"
                )
            else:
                return AgentResponse(
                    success=True,
                    message="✅ No linting issues found",
                    data=result
                )
                
        except Exception as e:
            self.logger.error(f"Linting failed: {e}")
            return AgentResponse(
                success=False,
                message=f"Linting failed: {str(e)}",
                error=str(e)
            )

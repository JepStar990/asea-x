"""
Linter Agent - Code quality enforcer
Ensures code meets linting and documentation standards
"""

import subprocess
import tempfile
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from src.agents.base_agent import BaseAgent, AgentCapability, AgentContext, AgentResponse
from src.validation.lint_engine import LintEngine
from src.validation.doc_validator import DocumentationValidator


class LinterAgent(BaseAgent):
    """
    Linter Agent - Enforces code quality standards
    
    Responsibilities:
    1. Run linters on code
    2. Check documentation
    3. Enforce style guides
    4. Fix violations automatically when possible
    """
    
    def __init__(self):
        super().__init__(
            name="linter",
            capabilities=[
                AgentCapability.LINTING,
                AgentCapability.DOCUMENTATION,
                AgentCapability.CODE_READING
            ]
        )
        
        self.lint_engine = LintEngine()
        self.doc_validator = DocumentationValidator()
        
        # Language-specific linter configurations
        self.linter_configs = {
            "python": {
                "linters": ["ruff", "black", "mypy", "pydocstyle"],
                "default_args": {
                    "ruff": ["check", "--fix"],
                    "black": ["--check"],
                    "mypy": [],
                    "pydocstyle": []
                }
            },
            "javascript": {
                "linters": ["eslint"],
                "default_args": ["--fix"]
            },
            "typescript": {
                "linters": ["eslint"],
                "default_args": ["--fix"]
            },
            "java": {
                "linters": ["checkstyle"],
                "default_args": []
            },
            "rust": {
                "linters": ["clippy"],
                "default_args": []
            },
            "go": {
                "linters": ["golangci-lint"],
                "default_args": ["run"]
            }
        }
    
    async def execute(self, context: AgentContext) -> AgentResponse:
        """
        Execute linting and documentation checks
        
        Steps:
        1. Detect language
        2. Run appropriate linters
        3. Check documentation
        4. Attempt automatic fixes
        5. Report violations
        """
        self.logger.info(f"Starting linting for task: {context.task_id}")
        
        try:
            # Step 1: Analyze codebase
            analysis = await self._analyze_codebase(context)
            
            # Step 2: Run linters
            lint_results = await self._run_linters(analysis, context)
            
            # Step 3: Check documentation
            doc_results = await self._check_documentation(analysis, context)
            
            # Step 4: Attempt fixes if violations found
            fix_results = {}
            if lint_results.get("has_issues", False) or doc_results.get("has_issues", False):
                fix_results = await self._attempt_fixes(analysis, lint_results, doc_results, context)
            
            # Step 5: Generate report
            report = await self._generate_report(
                analysis, lint_results, doc_results, fix_results, context
            )
            
            # Determine if we should block commits
            should_block = self._should_block_commit(lint_results, doc_results, fix_results)
            
            return AgentResponse(
                success=not should_block,
                message=report,
                data={
                    "lint_results": lint_results,
                    "doc_results": doc_results,
                    "fix_results": fix_results,
                    "lint_passed": not should_block,
                    "block_commit": should_block,
                    "violations_found": lint_results.get("issue_count", 0) + 
                                      doc_results.get("issue_count", 0)
                },
                next_step="Fix remaining violations before committing" if should_block else 
                         "Code quality checks passed"
            )
            
        except Exception as e:
            self.logger.error(f"Linting failed: {e}", exc_info=True)
            return AgentResponse(
                success=False,
                message=f"Linting failed: {str(e)}",
                error=str(e),
                requires_human_input=True
            )
    
    async def _analyze_codebase(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze codebase to determine languages and files"""
        # Get files from context
        files = list(context.file_context.keys())
        
        if not files:
            # Scan work directory
            workdir = Path("./workdir/dev")
            if workdir.exists():
                files = [str(f) for f in workdir.rglob("*") if f.is_file()]
            else:
                return {"languages": [], "files": []}
        
        # Determine languages
        languages = set()
        for file_path in files:
            lang = self._detect_language(file_path)
            if lang:
                languages.add(lang)
        
        return {
            "languages": list(languages),
            "files": files,
            "file_count": len(files)
        }
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension"""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".rs": "rust",
            ".go": "go",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala"
        }
        
        path = Path(file_path)
        for ext, lang in ext_map.items():
            if path.suffix == ext:
                return lang
        
        return None
    
    async def _run_linters(
        self, 
        analysis: Dict[str, Any], 
        context: AgentContext
    ) -> Dict[str, Any]:
        """Run linters for detected languages"""
        results = {
            "languages": [],
            "issue_count": 0,
            "has_issues": False,
            "details": {}
        }
        
        for language in analysis.get("languages", []):
            if language not in self.linter_configs:
                self.logger.warning(f"No linter config for language: {language}")
                continue
            
            # Get language-specific config
            config = self.linter_configs[language]
            language_results = {
                "linters_run": [],
                "issues": [],
                "fixed": [],
                "error": None
            }
            
            # Run each linter
            for linter in config["linters"]:
                linter_result = await self._run_linter(language, linter, context)
                language_results["linters_run"].append(linter)
                
                if linter_result.get("success"):
                    language_results["issues"].extend(linter_result.get("issues", []))
                    language_results["fixed"].extend(linter_result.get("fixed", []))
                else:
                    language_results["error"] = linter_result.get("error")
            
            # Update totals
            issue_count = len(language_results["issues"])
            results["issue_count"] += issue_count
            results["has_issues"] = results["has_issues"] or (issue_count > 0)
            results["details"][language] = language_results
        
        results["languages"] = list(results["details"].keys())
        return results
    
    async def _run_linter(
        self, 
        language: str, 
        linter: str, 
        context: AgentContext
    ) -> Dict[str, Any]:
        """Run a specific linter"""
        workdir = Path("./workdir/dev")
        
        if linter == "ruff" and language == "python":
            return await self._run_ruff(workdir)
        elif linter == "black" and language == "python":
            return await self._run_black(workdir)
        elif linter == "eslint" and language in ["javascript", "typescript"]:
            return await self._run_eslint(workdir, language)
        else:
            # Generic linter execution
            return await self._run_generic_linter(linter, language, workdir)
    
    async def _run_ruff(self, workdir: Path) -> Dict[str, Any]:
        """Run ruff linter on Python code"""
        try:
            # First check
            result = subprocess.run(
                ["ruff", "check", "--output-format", "json", str(workdir)],
                capture_output=True,
                text=True,
                cwd=workdir
            )
            
            issues = []
            if result.returncode != 0 and result.stdout:
                try:
                    ruff_output = json.loads(result.stdout)
                    for violation in ruff_output:
                        issues.append({
                            "file": violation["filename"],
                            "line": violation["location"]["row"],
                            "column": violation["location"]["column"],
                            "code": violation["code"],
                            "message": violation["message"],
                            "fixable": violation.get("fix", {}).get("applicable", False)
                        })
                except:
                    # Parse text output
                    for line in result.stdout.split('\n'):
                        if line.strip():
                            issues.append({"message": line, "raw": line})
            
            # Attempt to fix
            fixed = []
            if issues:
                fix_result = subprocess.run(
                    ["ruff", "check", "--fix", str(workdir)],
                    capture_output=True,
                    text=True,
                    cwd=workdir
                )
                
                if fix_result.returncode == 0:
                    fixed = [issue for issue in issues if issue.get("fixable", False)]
            
            return {
                "success": True,
                "issues": issues,
                "fixed": fixed,
                "issue_count": len(issues),
                "fixed_count": len(fixed)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "issues": [],
                "fixed": []
            }
    
    async def _run_black(self, workdir: Path) -> Dict[str, Any]:
        """Run black formatter on Python code"""
        try:
            # Check formatting
            result = subprocess.run(
                ["black", "--check", "--diff", str(workdir)],
                capture_output=True,
                text=True,
                cwd=workdir
            )
            
            issues = []
            if result.returncode != 0:
                # Parse diff output
                diff_lines = result.stdout.split('\n')
                current_file = None
                
                for line in diff_lines:
                    if line.startswith('--- '):
                        current_file = line[4:].strip()
                    elif line.startswith('+++ '):
                        continue
                    elif line.startswith('@@ '):
                        if current_file:
                            issues.append({
                                "file": current_file,
                                "diff": line,
                                "message": "Formatting issue"
                            })
            
            # Apply formatting
            fixed = []
            if issues:
                fix_result = subprocess.run(
                    ["black", str(workdir)],
                    capture_output=True,
                    text=True,
                    cwd=workdir
                )
                
                if fix_result.returncode == 0:
                    fixed = issues  # Assume all were fixed
            
            return {
                "success": True,
                "issues": issues,
                "fixed": fixed
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _run_eslint(self, workdir: Path, language: str) -> Dict[str, Any]:
        """Run eslint on JavaScript/TypeScript code"""
        try:
            # Check if eslint is installed
            result = subprocess.run(
                ["npx", "eslint", "--version"],
                capture_output=True,
                text=True,
                cwd=workdir
            )
            
            if result.returncode != 0:
                # Try to install eslint
                install_result = subprocess.run(
                    ["npm", "install", "--save-dev", "eslint"],
                    capture_output=True,
                    text=True,
                    cwd=workdir
                )
                
                if install_result.returncode != 0:
                    return {
                        "success": False,
                        "error": "Failed to install eslint"
                    }
            
            # Run eslint
            result = subprocess.run(
                ["npx", "eslint", "--format", "json", "."],
                capture_output=True,
                text=True,
                cwd=workdir
            )
            
            issues = []
            if result.stdout:
                try:
                    eslint_output = json.loads(result.stdout)
                    for file_result in eslint_output:
                        for message in file_result.get("messages", []):
                            issues.append({
                                "file": file_result["filePath"],
                                "line": message["line"],
                                "column": message["column"],
                                "rule": message.get("ruleId", ""),
                                "message": message["message"],
                                "severity": message["severity"],
                                "fixable": message.get("fix", None) is not None
                            })
                except:
                    pass
            
            # Fix issues
            fixed = []
            if issues:
                fix_result = subprocess.run(
                    ["npx", "eslint", "--fix", "."],
                    capture_output=True,
                    text=True,
                    cwd=workdir
                )
                
                if fix_result.returncode == 0:
                    fixed = [issue for issue in issues if issue.get("fixable", False)]
            
            return {
                "success": True,
                "issues": issues,
                "fixed": fixed
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _run_generic_linter(
        self, 
        linter: str, 
        language: str, 
        workdir: Path
    ) -> Dict[str, Any]:
        """Run generic linter"""
        try:
            # Try to run linter
            result = subprocess.run(
                [linter, "."],
                capture_output=True,
                text=True,
                cwd=workdir
            )
            
            issues = []
            if result.returncode != 0 and result.stderr:
                # Parse error output
                for line in result.stderr.split('\n'):
                    if line.strip():
                        issues.append({
                            "message": line,
                            "raw": line,
                            "linter": linter
                        })
            
            return {
                "success": result.returncode == 0 or issues,
                "issues": issues,
                "raw_output": result.stdout + result.stderr
            }
            
        except FileNotFoundError:
            return {
                "success": False,
                "error": f"Linter '{linter}' not found"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _check_documentation(
        self, 
        analysis: Dict[str, Any], 
        context: AgentContext
    ) -> Dict[str, Any]:
        """Check documentation quality"""
        results = {
            "languages": [],
            "issue_count": 0,
            "has_issues": False,
            "details": {}
        }
        
        for language in analysis.get("languages", []):
            if language == "python":
                lang_results = await self._check_python_docs(analysis, context)
            elif language in ["javascript", "typescript"]:
                lang_results = await self._check_js_docs(analysis, context)
            else:
                lang_results = {"issues": [], "error": f"No doc checker for {language}"}
            
            # Update totals
            issue_count = len(lang_results.get("issues", []))
            results["issue_count"] += issue_count
            results["has_issues"] = results["has_issues"] or (issue_count > 0)
            results["details"][language] = lang_results
        
        results["languages"] = list(results["details"].keys())
        return results
    
    async def _check_python_docs(
        self, 
        analysis: Dict[str, Any], 
        context: AgentContext
    ) -> Dict[str, Any]:
        """Check Python documentation"""
        try:
            # Run pydocstyle
            result = subprocess.run(
                ["pydocstyle", "."],
                capture_output=True,
                text=True,
                cwd="./workdir/dev"
            )
            
            issues = []
            if result.stdout:
                # Parse pydocstyle output
                current_file = None
                for line in result.stdout.split('\n'):
                    if line.strip():
                        if ':' in line:
                            parts = line.split(':', 3)
                            if len(parts) >= 4:
                                issues.append({
                                    "file": parts[0].strip(),
                                    "line": parts[1].strip(),
                                    "code": parts[2].strip(),
                                    "message": parts[3].strip()
                                })
            
            return {
                "success": len(issues) == 0,
                "issues": issues,
                "linter": "pydocstyle"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "issues": []
            }
    
    async def _check_js_docs(
        self, 
        analysis: Dict[str, Any], 
        context: AgentContext
    ) -> Dict[str, Any]:
        """Check JavaScript/TypeScript documentation"""
        # For JS/TS, we'll do a simple check for JSDoc comments
        issues = []
        
        for file_path in analysis.get("files", []):
            if file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
                content = context.file_context.get(file_path)
                if content:
                    # Simple check for function declarations without JSDoc
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if (line.strip().startswith('function ') or 
                            line.strip().startswith('const ') and '=' in line and 
                            ('=>' in line or lines[i+1:i+3] and any('=>' in l for l in lines[i+1:i+3]))):
                            
                            # Check for JSDoc in previous lines
                            has_jsdoc = False
                            for j in range(max(0, i-5), i):
                                if lines[j].strip().startswith('/**'):
                                    has_jsdoc = True
                                    break
                            
                            if not has_jsdoc:
                                issues.append({
                                    "file": file_path,
                                    "line": i + 1,
                                    "message": "Function missing JSDoc comment",
                                    "code": line.strip()[:50]
                                })
        
        return {
            "success": len(issues) == 0,
            "issues": issues,
            "linter": "jsdoc-check"
        }
    
    async def _attempt_fixes(
        self, 
        analysis: Dict[str, Any],
        lint_results: Dict[str, Any],
        doc_results: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """Attempt to automatically fix issues"""
        fixed_issues = []
        remaining_issues = []
        
        # Fix linting issues that have auto-fix
        for language, lang_results in lint_results.get("details", {}).items():
            for issue in lang_results.get("issues", []):
                if issue.get("fixable", False):
                    fix_success = await self._fix_issue(issue, language, context)
                    if fix_success:
                        fixed_issues.append(issue)
                    else:
                        remaining_issues.append(issue)
                else:
                    remaining_issues.append(issue)
        
        # Fix documentation issues
        for language, lang_results in doc_results.get("details", {}).items():
            for issue in lang_results.get("issues", []):
                if language == "python" and "missing" in issue.get("message", "").lower():
                    fix_success = await self._fix_doc_issue(issue, language, context)
                    if fix_success:
                        fixed_issues.append(issue)
                    else:
                        remaining_issues.append(issue)
                else:
                    remaining_issues.append(issue)
        
        return {
            "fixed_issues": fixed_issues,
            "remaining_issues": remaining_issues,
            "fix_count": len(fixed_issues),
            "remaining_count": len(remaining_issues)
        }
    
    async def _fix_issue(
        self, 
        issue: Dict[str, Any], 
        language: str, 
        context: AgentContext
    ) -> bool:
        """Attempt to fix a specific issue"""
        try:
            file_path = issue.get("file")
            if not file_path:
                return False
            
            # Get file content
            content = context.file_context.get(file_path)
            if not content:
                # Try to read from disk
                path = Path(file_path)
                if path.exists():
                    content = path.read_text(encoding='utf-8')
                else:
                    return False
            
            # Parse issue details
            line_num = issue.get("line")
            message = issue.get("message", "")
            
            if line_num and "trailing whitespace" in message.lower():
                # Fix trailing whitespace
                lines = content.split('\n')
                if 1 <= line_num <= len(lines):
                    lines[line_num - 1] = lines[line_num - 1].rstrip()
                    new_content = '\n'.join(lines)
                    
                    # Write back
                    path.write_text(new_content, encoding='utf-8')
                    
                    # Update context
                    self.state_manager.add_file_context(file_path, new_content)
                    return True
            
            # More sophisticated fixes could be added here
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to fix issue: {e}")
            return False
    
    async def _fix_doc_issue(
        self, 
        issue: Dict[str, Any], 
        language: str, 
        context: AgentContext
    ) -> bool:
        """Fix documentation issue"""
        try:
            if language == "python":
                return await self._fix_python_doc_issue(issue, context)
            return False
        except Exception as e:
            self.logger.error(f"Failed to fix doc issue: {e}")
            return False
    
    async def _fix_python_doc_issue(
        self, 
        issue: Dict[str, Any], 
        context: AgentContext
    ) -> bool:
        """Fix Python documentation issue"""
        file_path = issue.get("file")
        line_num = issue.get("line")
        
        if not file_path or not line_num:
            return False
        
        # Get file content
        content = context.file_context.get(file_path)
        if not content:
            return False
        
        lines = content.split('\n')
        if line_num > len(lines):
            return False
        
        # Find the function/class definition
        target_line = lines[line_num - 1]
        
        # Generate docstring using LLM
        prompt = f"""
        Generate a proper Python docstring for this code:
        
        {target_line}
        
        Follow PEP 257 guidelines.
        Return just the docstring (triple-quoted string).
        """
        
        response = self.llm_client.chat_completion([
            {"role": "system", "content": "You are a Python documentation expert."},
            {"role": "user", "content": prompt}
        ])
        
        docstring = response.content.strip()
        
        # Insert docstring
        indent = len(target_line) - len(target_line.lstrip())
        indent_str = ' ' * indent
        
        docstring_lines = docstring.split('\n')
        indented_docstring = '\n'.join([indent_str + line for line in docstring_lines])
        
        lines.insert(line_num, indented_docstring)  # Insert after the definition
        
        new_content = '\n'.join(lines)
        
        # Write back
        path = Path(file_path)
        path.write_text(new_content, encoding='utf-8')
        
        # Update context
        self.state_manager.add_file_context(file_path, new_content)
        
        return True
    
    async def _generate_report(
        self,
        analysis: Dict[str, Any],
        lint_results: Dict[str, Any],
        doc_results: Dict[str, Any],
        fix_results: Dict[str, Any],
        context: AgentContext
    ) -> str:
        """Generate linting report"""
        lines = ["**Code Quality Report**", ""]
        
        # Summary
        total_issues = lint_results.get("issue_count", 0) + doc_results.get("issue_count", 0)
        fixed_count = fix_results.get("fix_count", 0)
        remaining = fix_results.get("remaining_count", 0)
        
        lines.append(f"**Summary:**")
        lines.append(f"Languages analyzed: {', '.join(analysis.get('languages', []))}")
        lines.append(f"Files checked: {analysis.get('file_count', 0)}")
        lines.append(f"Total issues found: {total_issues}")
        lines.append(f"Auto-fixed: {fixed_count}")
        lines.append(f"Remaining issues: {remaining}")
        lines.append("")
        
        # Linting results by language
        if lint_results.get("details"):
            lines.append("**Linting Results:**")
            for language, lang_results in lint_results["details"].items():
                issues = len(lang_results.get("issues", []))
                if issues > 0:
                    lines.append(f"  {language}: {issues} issues")
                    for issue in lang_results["issues"][:3]:  # Show first 3
                        lines.append(f"    - {issue.get('message', 'Unknown issue')}")
            lines.append("")
        
        # Documentation results
        if doc_results.get("details"):
            lines.append("**Documentation Results:**")
            for language, lang_results in doc_results["details"].items():
                issues = len(lang_results.get("issues", []))
                if issues > 0:
                    lines.append(f"  {language}: {issues} documentation issues")
            lines.append("")
        
        # Fix results
        if fixed_count > 0:
            lines.append(f"✅ Auto-fixed {fixed_count} issues")
        
        # Final status
        if remaining == 0:
            lines.append("✅ All code quality checks passed!")
        else:
            lines.append(f"⚠️ {remaining} issues need manual fixing")
            lines.append("  Use /dev mode to fix issues or /lint to re-check")
        
        return "\n".join(lines)
    
    def _should_block_commit(
        self, 
        lint_results: Dict[str, Any],
        doc_results: Dict[str, Any],
        fix_results: Dict[str, Any]
    ) -> bool:
        """Determine if commit should be blocked"""
        # Block if there are remaining issues
        remaining = fix_results.get("remaining_count", 0)
        
        # Also block if there are critical issues (errors, not warnings)
        has_critical_issues = False
        for lang_results in lint_results.get("details", {}).values():
            for issue in lang_results.get("issues", []):
                if issue.get("severity") == 2:  # Error severity
                    has_critical_issues = True
                    break
        
        return remaining > 0 or has_critical_issues

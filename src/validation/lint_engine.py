"""
Lint Engine - Language-specific linting
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

class LintEngine:
    """Language-specific linting engine"""

def __init__(self):
        self.logger = logging.getLogger(__name__)
                # Linter configurations by language
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

    def lint_directory(self, directory_path: str) -> Dict[str, Any]:
        """Lint a directory"""
        path = Path(directory_path)
        
        if not path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {directory_path}",
                "issues": []
            }
        
        # Detect languages in directory
        languages = self._detect_languages(path)
        
        results = {
            "languages": languages,
            "issues": [],
            "linters_run": [],
            "success": True
        }
        
        # Run linters for each detected language
        for language in languages:
            if language in self.linter_configs:
                lang_results = self._lint_language(path, language)
                results["issues"].extend(lang_results.get("issues", []))
                results["linters_run"].extend(lang_results.get("linters_run", []))
                
                if not lang_results.get("success", True):
                    results["success"] = False
        
        return results
    
    def _detect_languages(self, directory: Path) -> List[str]:
        """Detect programming languages in directory"""
        extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.rs': 'rust',
            '.go': 'go',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby'
        }
        
        languages = set()
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix in extensions:
                    languages.add(extensions[suffix])
        
        return list(languages)
    
    def _lint_language(self, directory: Path, language: str) -> Dict[str, Any]:
        """Run linters for a specific language"""
        config = self.linter_configs.get(language, {})
        linters = config.get("linters", [])
        
        results = {
            "language": language,
            "linters_run": [],
            "issues": [],
            "success": True
        }
        
        for linter in linters:
            if self._check_linter_available(linter):
                linter_result = self._run_linter(linter, directory, language)
                results["linters_run"].append(linter)
                results["issues"].extend(linter_result.get("issues", []))
                
                if not linter_result.get("success", True):
                    results["success"] = False
            else:
                self.logger.warning(f"Linter not available: {linter}")
                results["issues"].append({
                    "type": "warning",
                    "message": f"Linter {linter} not available for {language}",
                    "file": "",
                    "line": 0
                })
        
        return results
    
    def _check_linter_available(self, linter: str) -> bool:
        """Check if linter is available in PATH"""
        try:
            result = subprocess.run(
                [linter, "--version"],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False

    def _run_linter(self, linter: str, directory: Path, language: str) -> Dict[str, Any]:
        """Run a specific linter"""
        try:
            if linter == "ruff" and language == "python":
                return self._run_ruff(directory)
            elif linter == "black" and language == "python":
                return self._run_black(directory)
            elif linter == "eslint" and language in ["javascript", "typescript"]:
                return self._run_eslint(directory)
            else:
                # Generic linter execution
                result = subprocess.run(
                    [linter, str(directory)],
                    capture_output=True,
                    text=True,
                    cwd=directory,
                    timeout=30
                )
                
                issues = []
                if result.returncode != 0:
                    for line in result.stderr.split('\n'):
                        if line.strip():
                            issues.append({
                                "type": "error",
                                "message": line,
                                "file": "",
                                "line": 0,
                                "linter": linter
                            })
                
                return {
                    "success": result.returncode == 0,
                    "issues": issues
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "issues": [{
                    "type": "error",
                    "message": f"Linter {linter} timed out",
                    "file": "",
                    "line": 0
                }]
            }
        except Exception as e:
            return {
                "success": False,
                "issues": [{
                    "type": "error",
                    "message": f"Failed to run {linter}: {str(e)}",
                    "file": "",
                    "line": 0
                }]
            }
    
    def _run_ruff(self, directory: Path) -> Dict[str, Any]:
        """Run ruff linter"""
        try:
            # Check for issues
            result = subprocess.run(
                ["ruff", "check", str(directory)],
                capture_output=True,
                text=True,
                cwd=directory,
                timeout=30
            )
            
            issues = []
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        parts = line.split(':')
                        if len(parts) >= 3:
                            issues.append({
                                "type": "error",
                                "file": parts[0],
                                "line": int(parts[1]) if parts[1].isdigit() else 0,
                                "column": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0,
                                "message": ':'.join(parts[3:]) if len(parts) > 3 else line,
                                "linter": "ruff"
                            })
            
            # Try to fix issues
            if issues:
                subprocess.run(
                    ["ruff", "check", "--fix", str(directory)],
                    capture_output=True,
                    text=True,
                    cwd=directory,
                    timeout=30
                )
            
            return {
                "success": result.returncode == 0,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "success": False,
                "issues": [{
                    "type": "error",
                    "message": f"Ruff failed: {str(e)}",
                    "file": "",
                    "line": 0
                }]
            }
    
    def _run_black(self, directory: Path) -> Dict[str, Any]:
        """Run black formatter"""
        try:
            result = subprocess.run(
                ["black", "--check", str(directory)],
                capture_output=True,
                text=True,
                cwd=directory,
                timeout=30
            )
            
            issues = []
            if result.returncode != 0:
                issues.append({
                    "type": "formatting",
                    "message": "Code needs formatting with black",
                    "file": "",
                    "line": 0,
                    "linter": "black"
                })
            
            return {
                "success": result.returncode == 0,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "success": False,
                "issues": [{
                    "type": "error",
                    "message": f"Black failed: {str(e)}",
                    "file": "",
                    "line": 0
                }]
            }
    
    def _run_eslint(self, directory: Path) -> Dict[str, Any]:
        """Run eslint"""
        try:
            # Check if package.json exists
            package_json = directory / "package.json"
            if not package_json.exists():
                return {
                    "success": True,
                    "issues": [{
                        "type": "warning",
                        "message": "No package.json found for eslint",
                        "file": "",
                        "line": 0
                    }]
                }
            
            result = subprocess.run(
                ["npx", "eslint", "."],
                capture_output=True,
                text=True,
                cwd=directory,
                timeout=30
            )
            
            issues = []
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip() and not line.startswith('✔'):
                        issues.append({
                            "type": "error",
                            "message": line,
                            "file": "",
                            "line": 0,
                            "linter": "eslint"
                        })
            
            return {
                "success": result.returncode == 0,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "success": False,
                "issues": [{
                    "type": "error",
                    "message": f"ESLint failed: {str(e)}",
                    "file": "",
                    "line": 0
                }]
            }
    
    def lint_file(self, file_path: str) -> Dict[str, Any]:
        """Lint a single file"""
        path = Path(file_path)
        
        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "issues": []
            }
        
        # Detect language from file extension
        language = self._detect_language_from_file(path)
        
        if not language:
            return {
                "success": True,
                "issues": [],
                "message": f"No linter configured for {path.suffix} files"
            }
        
        # Run linters for this language
        directory = path.parent
        return self._lint_language(directory, language)
    
    def _detect_language_from_file(self, file_path: Path) -> Optional[str]:
        """Detect language from file extension"""
        extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.rs': 'rust',
            '.go': 'go',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby'
        }
        
        suffix = file_path.suffix.lower()
        return extensions.get(suffix)

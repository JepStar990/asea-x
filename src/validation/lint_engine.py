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

    # ----------------------------------------------------------
    # Directory linting
    # ----------------------------------------------------------

    def lint_directory(self, directory_path: str) -> Dict[str, Any]:
        """Lint a directory"""
        path = Path(directory_path)

        if not path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {directory_path}",
                "issues": []
            }

        languages = self._detect_languages(path)

        results = {
            "languages": languages,
            "issues": [],
            "linters_run": [],
            "success": True
        }

        for language in languages:
            if language in self.linter_configs:
                lang_results = self._lint_language(path, language)
                results["issues"].extend(lang_results.get("issues", []))
                results["linters_run"].extend(lang_results.get("linters_run", []))

                if not lang_results.get("success", True):
                    results["success"] = False

        return results

    # ----------------------------------------------------------
    # Language detection
    # ----------------------------------------------------------

    def _detect_languages(self, directory: Path) -> List[str]:
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

    # ----------------------------------------------------------
    # Lint per language
    # ----------------------------------------------------------

    def _lint_language(self, directory: Path, language: str) -> Dict[str, Any]:
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

    # ----------------------------------------------------------
    # Linter availability
    # ----------------------------------------------------------

    def _check_linter_available(self, linter: str) -> bool:
        try:
            result = subprocess.run(
                [linter, "--version"],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.returncode == 0
        except Exception:
            return False

    # ----------------------------------------------------------
    # Running specific linters
    # ----------------------------------------------------------

    def _run_linter(self, linter: str, directory: Path, language: str) -> Dict[str, Any]:
        try:
            if linter == "ruff" and language == "python":
                return self._run_ruff(directory)
            if linter == "black" and language == "python":
                return self._run_black(directory)
            if linter == "eslint" and language in ("javascript", "typescript"):
                return self._run_eslint(directory)

            # Generic linter: command <dir>
            result = subprocess.run(
                [linter, str(directory)],
                capture_output=True,
                text=True,
                cwd=directory,
                timeout=30
            )

            issues = []
            if result.returncode != 0:
                for line in result.stderr.split("\n"):
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
                    "message": f"{linter} timed out",
                    "file": "",
                    "line": 0
                }]
            }

        except Exception as e:
            return {
                "success": False,
                "issues": [{
                    "type": "error",
                    "message": f"{linter} failed: {str(e)}",
                    "file": "",
                    "line": 0
                }]
            }

    # ----------------------------------------------------------
    # Ruff
    # ----------------------------------------------------------

    def _run_ruff(self, directory: Path) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                ["ruff", "check", str(directory)],
                capture_output=True,
                text=True,
                cwd=directory,
                timeout=30
            )

            issues = []
            if result.stdout:
                for line in result.stdout.split("\n"):
                    if ":" in line:
                        parts = line.split(":")
                        if len(parts) >= 3:
                            issues.append({
                                "type": "error",
                                "file": parts[0],
                                "line": int(parts[1]) if parts[1].isdigit() else 0,
                                "column": int(parts[2]) if parts[2].isdigit() else 0,
                                "message": ":".join(parts[3:]),
                                "linter": "ruff"
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
                    "message": f"Ruff failed: {str(e)}",
                    "file": "",
                    "line": 0
                }]
            }

    # ----------------------------------------------------------
    # Black
    # ----------------------------------------------------------

    def _run_black(self, directory: Path) -> Dict[str, Any]:
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

    # ----------------------------------------------------------
    # ESLint
    # ----------------------------------------------------------

    def _run_eslint(self, directory: Path) -> Dict[str, Any]:
        try:
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
                for line in result.stdout.split("\n"):
                    if line.strip() and not line.startswith("✔"):
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

    # ----------------------------------------------------------
    # Single file linting
    # ----------------------------------------------------------

    def lint_file(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "issues": []
            }

        language = self._detect_language_from_file(path)

        if not language:
            return {
                "success": True,
                "issues": [],
                "message": f"No linter configured for {path.suffix} files"
            }

        directory = path.parent
        return self._lint_language(directory, language)

    # ----------------------------------------------------------
    # Detect language from extension
    # ----------------------------------------------------------

    def _detect_language_from_file(self, file_path: Path) -> Optional[str]:
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

        return extensions.get(file_path.suffix.lower())

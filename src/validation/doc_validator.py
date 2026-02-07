"""
Documentation Validator - Checks documentation quality
"""

import ast
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging


class DocumentationValidator:
    """Validates documentation quality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_directory(self, directory_path: str) -> Dict[str, Any]:
        """Validate documentation in directory"""
        path = Path(directory_path)
        
        if not path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {directory_path}",
                "issues": []
            }
        
        # Find Python files
        python_files = list(path.rglob("*.py"))
        
        issues = []
        validated_files = 0
        
        for file_path in python_files:
            file_issues = self.validate_python_file(file_path)
            issues.extend(file_issues)
            
            if not file_issues:
                validated_files += 1
        
        return {
            "success": len(issues) == 0,
            "files_checked": len(python_files),
            "files_without_issues": validated_files,
            "issues": issues
        }
    
    def validate_python_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Validate documentation in a Python file"""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Parse Python file
            try:
                tree = ast.parse(content)
            except SyntaxError:
                # Skip files with syntax errors
                return [{
                    "file": str(file_path),
                    "line": 0,
                    "type": "syntax_error",
                    "message": "File has syntax errors, cannot check documentation"
                }]
            
            # Check for module docstring
            if not self._has_module_docstring(tree):
                issues.append({
                    "file": str(file_path),
                    "line": 1,
                    "type": "missing_docstring",
                    "message": "Module missing docstring"
                })
            
            # Check functions and classes
            for node in ast.walk(tree):
                # Check function docstrings
                if isinstance(node, ast.FunctionDef):
                    if not self._has_docstring(node):
                        issues.append({
                            "file": str(file_path),
                            "line": node.lineno,
                            "type": "missing_docstring",
                            "message": f"Function '{node.name}' missing docstring"
                        })
                    else:
                        # Check docstring quality
                        docstring_issues = self._check_docstring_quality(node)
                        issues.extend(docstring_issues)
                
                # Check class docstrings
                elif isinstance(node, ast.ClassDef):
                    if not self._has_docstring(node):
                        issues.append({
                            "file": str(file_path),
                            "line": node.lineno,
                            "type": "missing_docstring",
                            "message": f"Class '{node.name}' missing docstring"
                        })
                    else:
                        # Check class methods
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                if not self._has_docstring(item):
                                    issues.append({
                                        "file": str(file_path),
                                        "line": item.lineno,
                                        "type": "missing_docstring",
                                        "message": f"Method '{item.name}' in class '{node.name}' missing docstring"
                                    })
            
            # Check for TODO comments without explanations
            todo_issues = self._check_todo_comments(content, file_path)
            issues.extend(todo_issues)
            
        except Exception as e:
            self.logger.error(f"Failed to validate {file_path}: {e}")
            issues.append({
                "file": str(file_path),
                "line": 0,
                "type": "validation_error",
                "message": f"Validation failed: {str(e)}"
            })
        
        return issues
    
    def _has_module_docstring(self, tree: ast.Module) -> bool:
        """Check if module has docstring"""
        if not tree.body:
            return False
        
        first_node = tree.body[0]
        if isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Constant):
            if isinstance(first_node.value.value, str):
                return True
        
        return False
    
    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if node has docstring"""
        if not hasattr(node, 'body') or not node.body:
            return False
        
        first_item = node.body[0]
        if isinstance(first_item, ast.Expr) and isinstance(first_item.value, ast.Constant):
            if isinstance(first_item.value.value, str):
                return True
        
        return False
    
    def _check_docstring_quality(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Check docstring quality"""
        issues = []
        
        if not hasattr(node, 'body') or not node.body:
            return issues
        
        first_item = node.body[0]
        if not (isinstance(first_item, ast.Expr) and isinstance(first_item.value, ast.Constant)):
            return issues
        
        docstring = first_item.value.value
        if not isinstance(docstring, str):
            return issues
        
        # Check for empty docstring
        if not docstring.strip():
            issues.append({
                "file": "",  # Will be filled by caller
                "line": node.lineno,
                "type": "empty_docstring",
                "message": f"Function '{node.name}' has empty docstring"
            })
            return issues
        
        # Check for single-line docstrings that should be multi-line
        lines = docstring.strip().split('\n')
        if len(lines) == 1 and len(docstring) > 50:
            issues.append({
                "file": "",
                "line": node.lineno,
                "type": "docstring_style",
                "message": f"Function '{node.name}' has long single-line docstring, consider multi-line"
            })
        
        # Check for missing parameter documentation
        if node.args.args:
            param_names = [arg.arg for arg in node.args.args]
            for param in param_names:
                if param not in docstring:
                    issues.append({
                        "file": "",
                        "line": node.lineno,
                        "type": "missing_param",
                        "message": f"Function '{node.name}' missing documentation for parameter '{param}'"
                    })
        
        # Check for missing return documentation if function returns something
        returns_something = any(
            isinstance(n, ast.Return) and n.value is not None 
            for n in ast.walk(node)
        )
        
        if returns_something and "return" not in docstring.lower() and "returns" not in docstring.lower():
            issues.append({
                "file": "",
                "line": node.lineno,
                "type": "missing_return",
                "message": f"Function '{node.name}' missing return documentation"
            })
        
        return issues
    
    def _check_todo_comments(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Check TODO comments for explanations"""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Look for TODO comments
            todo_match = re.search(r'#\s*TODO\s*:?\s*(.+)', line, re.IGNORECASE)
            if todo_match:
                explanation = todo_match.group(1).strip()
                if not explanation or len(explanation) < 10:
                    issues.append({
                        "file": str(file_path),
                        "line": i,
                        "type": "vague_todo",
                        "message": "TODO comment without clear explanation"
                    })
        
        return issues
    
    def validate_readme(self, directory_path: str) -> Dict[str, Any]:
        """Validate README file"""
        path = Path(directory_path)
        
        # Look for README files
        readme_files = [
            path / "README.md",
            path / "README.rst",
            path / "README.txt",
            path / "README"
        ]
        
        readme_path = None
        for readme_file in readme_files:
            if readme_file.exists():
                readme_path = readme_file
                break
        
        if not readme_path:
            return {
                "success": False,
                "issues": [{
                    "type": "missing_readme",
                    "message": "No README file found",
                    "file": str(directory_path)
                }]
            }
        
        try:
            content = readme_path.read_text(encoding='utf-8')
            issues = []
            
            # Check for minimum content
            if len(content.strip()) < 100:
                issues.append({
                    "type": "short_readme",
                    "message": "README file is very short",
                    "file": str(readme_path)
                })
            
            # Check for essential sections
            sections = ["description", "installation", "usage", "contributing", "license"]
            content_lower = content.lower()
            
            for section in sections:
                if section not in content_lower:
                    issues.append({
                        "type": "missing_section",
                        "message": f"README missing '{section}' section",
                        "file": str(readme_path)
                    })
            
            return {
                "success": len(issues) == 0,
                "file": str(readme_path),
                "size": len(content),
                "issues": issues
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file": str(readme_path),
                "issues": [{
                    "type": "readme_error",
                    "message": f"Failed to read README: {str(e)}"
                }]
            }

"""
Git Manager - Version control integration
Handles atomic commits, branching, and Git operations
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

import git
from git import Repo, GitCommandError


class CommitType(str, Enum):
    """Conventional commit types"""
    FEAT = "feat"
    FIX = "fix"
    DOCS = "docs"
    STYLE = "style"
    REFACTOR = "refactor"
    TEST = "test"
    CHORE = "chore"
    BUILD = "build"
    CI = "ci"
    PERF = "perf"


@dataclass
class CommitInfo:
    """Information about a commit"""
    hash: str
    author: str
    date: datetime
    message: str
    files_changed: List[str]
    insertions: int
    deletions: int
    type: Optional[CommitType] = None
    scope: Optional[str] = None
    breaking: bool = False


@dataclass
class BranchInfo:
    """Information about a branch"""
    name: str
    is_current: bool
    last_commit: Optional[CommitInfo] = None
    ahead: int = 0
    behind: int = 0


class GitManager:
    """
    Manages Git operations for ASEA-X
    
    Features:
    1. Repository initialization and management
    2. Atomic commits with conventional messages
    3. Branch management
    4. Change tracking
    5. Git hook integration
    6. Conflict resolution assistance
    """
    
    def __init__(self, workdir: Path = Path("./workdir")):
        self.workdir = workdir
        self.repo: Optional[Repo] = None
        self.logger = logging.getLogger(__name__)
        
        # Git configuration
        self.auto_commit = True
        self.branch_prefix = "agent/"
        self.commit_prefix = "feat(agent):"
        
        # Initialize repository
        self._init_repository()
    
    def _init_repository(self):
        """Initialize or load Git repository"""
        git_dir = self.workdir / ".git"
        
        try:
            if git_dir.exists():
                self.repo = Repo(str(self.workdir))
                self.logger.info(f"Loaded existing repository at {self.workdir}")
            else:
                # Initialize new repository
                self.repo = Repo.init(str(self.workdir))
                self.logger.info(f"Initialized new repository at {self.workdir}")
                
                # Create initial commit
                self._create_initial_commit()
                
            # Configure git if needed
            self._configure_git()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Git repository: {e}")
            self.repo = None
    
    def _configure_git(self):
        """Configure Git settings"""
        if not self.repo:
            return
        
        try:
            # Set safe directory (for newer Git versions)
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", str(self.workdir)],
                check=False
            )
            
            # Configure user if not set
            if not self.repo.config_reader().get_value("user", "name"):
                self.repo.git.config("user.name", "ASEA-X Agent")
            
            if not self.repo.config_reader().get_value("user", "email"):
                self.repo.git.config("user.email", "agent@asea-x.local")
            
            # Set default branch name
            self.repo.git.config("init.defaultBranch", "main")
            
        except Exception as e:
            self.logger.warning(f"Git configuration failed: {e}")
    
    def _create_initial_commit(self):
        """Create initial commit"""
        if not self.repo:
            return
        
        try:
            # Create README
            readme_path = self.workdir / "README.md"
            if not readme_path.exists():
                readme_path.write_text("# ASEA-X Project\n\nAutonomous Software Engineering Agent System\n")
            
            # Create .gitignore
            gitignore_path = self.workdir / ".gitignore"
            if not gitignore_path.exists():
                gitignore_content = """# ASEA-X generated
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.dll
*.dylib

# Virtual environments
venv/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
"""
                gitignore_path.write_text(gitignore_content)
            
            # Add and commit
            self.repo.git.add(A=True)
            self.repo.git.commit(m="feat: Initial commit by ASEA-X")
            
            self.logger.info("Created initial commit")
            
        except Exception as e:
            self.logger.error(f"Failed to create initial commit: {e}")
    
    def is_initialized(self) -> bool:
        """Check if Git is initialized"""
        return self.repo is not None and not self.repo.bare
    
    def get_status(self) -> Dict[str, Any]:
        """Get Git repository status"""
        if not self.is_initialized():
            return {"error": "Git not initialized"}
        
        try:
            status = {}
            
            # Current branch
            status["branch"] = self.repo.active_branch.name
            
            # Status summary
            status["clean"] = not self.repo.is_dirty()
            
            # Changed files
            changed_files = []
            for item in self.repo.index.diff(None):
                changed_files.append(item.a_path)
            status["changed_files"] = changed_files
            
            # Staged files
            staged_files = []
            for item in self.repo.index.diff("HEAD"):
                staged_files.append(item.a_path)
            status["staged_files"] = staged_files
            
            # Untracked files
            untracked_files = list(self.repo.untracked_files)
            status["untracked_files"] = untracked_files
            
            # Commit count
            status["commit_count"] = len(list(self.repo.iter_commits()))
            
            # Ahead/Behind (if tracking remote)
            try:
                if self.repo.active_branch.tracking_branch():
                    ahead_behind = self.repo.iter_commits(
                        f"{self.repo.active_branch.tracking_branch().name}..{self.repo.active_branch.name}",
                        max_count=1000
                    )
                    status["ahead"] = len(list(ahead_behind))
                    
                    behind_ahead = self.repo.iter_commits(
                        f"{self.repo.active_branch.name}..{self.repo.active_branch.tracking_branch().name}",
                        max_count=1000
                    )
                    status["behind"] = len(list(behind_ahead))
            except:
                status["ahead"] = 0
                status["behind"] = 0
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get Git status: {e}")
            return {"error": str(e)}
    
    def create_branch(self, branch_name: str, from_branch: Optional[str] = None) -> bool:
        """Create a new branch"""
        if not self.is_initialized():
            return False
        
        try:
            if from_branch:
                # Checkout the source branch first
                self.repo.git.checkout(from_branch)
            
            # Create new branch
            full_branch_name = f"{self.branch_prefix}{branch_name}"
            self.repo.git.checkout(b=self.full_branch_name)
            
            self.logger.info(f"Created branch: {full_branch_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create branch {branch_name}: {e}")
            return False
    
    def switch_branch(self, branch_name: str) -> bool:
        """Switch to an existing branch"""
        if not self.is_initialized():
            return False
        
        try:
            self.repo.git.checkout(branch_name)
            self.logger.info(f"Switched to branch: {branch_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch to branch {branch_name}: {e}")
            return False
    
    def commit_changes(
        self,
        message: str,
        files: Optional[List[str]] = None,
        commit_type: CommitType = CommitType.FEAT,
        scope: Optional[str] = None,
        breaking: bool = False
    ) -> Optional[str]:
        """
        Commit changes with conventional commit message
        
        Args:
            message: Commit message
            files: Specific files to commit (None for all)
            commit_type: Type of commit
            scope: Scope of changes
            breaking: Whether it's a breaking change
            
        Returns:
            Commit hash if successful, None otherwise
        """
        if not self.is_initialized():
            self.logger.error("Git not initialized")
            return None
        
        try:
            # Stage files
            if files:
                for file in files:
                    self.repo.git.add(file)
            else:
                self.repo.git.add(A=True)
            
            # Check if there are any changes to commit
            if not self.repo.index.diff("HEAD"):
                self.logger.info("No changes to commit")
                return None
            
            # Build commit message
            commit_msg = self._build_commit_message(
                message, commit_type, scope, breaking
            )
            
            # Commit
            self.repo.git.commit(m=commit_msg)
            
            commit_hash = self.repo.head.commit.hexsha[:8]
            self.logger.info(f"Committed changes: {commit_hash} - {commit_msg}")
            
            # Log the commit
            self._log_commit(commit_hash, commit_msg, files or [])
            
            return commit_hash
            
        except GitCommandError as e:
            self.logger.error(f"Git commit failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Commit failed: {e}")
            return None
    
    def _build_commit_message(
        self,
        message: str,
        commit_type: CommitType,
        scope: Optional[str],
        breaking: bool
    ) -> str:
        """Build conventional commit message"""
        parts = [commit_type.value]
        
        if scope:
            parts[0] = f"{parts[0]}({scope})"
        
        if breaking:
            parts[0] = f"{parts[0]}!"
        
        parts[0] = f"{parts[0]}:"
        
        # Add message
        parts.append(message)
        
        # Add body if message is long
        if len(message) < 50:
            return " ".join(parts)
        else:
            # Use first line as subject, rest as body
            lines = message.split('\n')
            subject = lines[0]
            body = '\n'.join(lines[1:]) if len(lines) > 1 else ""
            
            commit_msg = f"{parts[0]} {subject}"
            if body:
                commit_msg += f"\n\n{body}"
            
            return commit_msg
    
    def _log_commit(self, commit_hash: str, message: str, files: List[str]):
        """Log commit for auditing"""
        log_file = self.workdir / ".agent.commits.log"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "hash": commit_hash,
            "message": message,
            "files": files,
            "branch": self.repo.active_branch.name if self.repo else "unknown"
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def get_commit_history(self, limit: int = 10) -> List[CommitInfo]:
        """Get commit history"""
        if not self.is_initialized():
            return []
        
        try:
            commits = []
            
            for commit in self.repo.iter_commits(max_count=limit):
                # Get diff stats
                stats = commit.stats.total
                
                # Parse commit message for conventional commits
                commit_type = None
                scope = None
                breaking = False
                
                msg = commit.message
                if ': ' in msg:
                    header = msg.split(': ')[0]
                    if '(' in header and ')' in header:
                        # Has scope
                        type_part = header.split('(')[0]
                        scope = header.split('(')[1].split(')')[0]
                        if type_part.endswith('!'):
                            breaking = True
                            type_part = type_part[:-1]
                        commit_type = CommitType(type_part)
                    elif header.endswith('!'):
                        breaking = True
                        header = header[:-1]
                        commit_type = CommitType(header)
                    else:
                        commit_type = CommitType(header)
                
                commits.append(CommitInfo(
                    hash=commit.hexsha[:8],
                    author=str(commit.author),
                    date=commit.authored_datetime,
                    message=commit.message.strip(),
                    files_changed=list(commit.stats.files.keys()),
                    insertions=stats.get('insertions', 0),
                    deletions=stats.get('deletions', 0),
                    type=commit_type,
                    scope=scope,
                    breaking=breaking
                ))
            
            return commits
            
        except Exception as e:
            self.logger.error(f"Failed to get commit history: {e}")
            return []
    
    def get_branches(self) -> List[BranchInfo]:
        """Get list of branches"""
        if not self.is_initialized():
            return []
        
        try:
            branches = []
            current_branch = self.repo.active_branch.name
            
            for ref in self.repo.branches:
                branch_info = BranchInfo(
                    name=ref.name,
                    is_current=ref.name == current_branch,
                    last_commit=None
                )
                
                # Get last commit info
                try:
                    commit = ref.commit
                    branch_info.last_commit = CommitInfo(
                        hash=commit.hexsha[:8],
                        author=str(commit.author),
                        date=commit.authored_datetime,
                        message=commit.message.split('\n')[0][:50],
                        files_changed=[],
                        insertions=0,
                        deletions=0
                    )
                except:
                    pass
                
                branches.append(branch_info)
            
            return branches
            
        except Exception as e:
            self.logger.error(f"Failed to get branches: {e}")
            return []
    
    def get_diff(self, commit_hash: Optional[str] = None) -> str:
        """Get diff of changes"""
        if not self.is_initialized():
            return ""
        
        try:
            if commit_hash:
                # Diff between specific commit and its parent
                commit = self.repo.commit(commit_hash)
                if commit.parents:
                    return self.repo.git.diff(commit.parents[0], commit)
                else:
                    return self.repo.git.diff(commit, tree=True)
            else:
                # Diff of unstaged changes
                return self.repo.git.diff()
                
        except Exception as e:
            self.logger.error(f"Failed to get diff: {e}")
            return ""
    
    def get_file_history(self, file_path: str, limit: int = 5) -> List[CommitInfo]:
        """Get commit history for a specific file"""
        if not self.is_initialized():
            return []
        
        try:
            commits = []
            
            for commit in self.repo.iter_commits(paths=file_path, max_count=limit):
                # Check if file was modified in this commit
                stats = commit.stats.files.get(file_path)
                if stats:
                    commits.append(CommitInfo(
                        hash=commit.hexsha[:8],
                        author=str(commit.author),
                        date=commit.authored_datetime,
                        message=commit.message.split('\n')[0][:100],
                        files_changed=[file_path],
                        insertions=stats.get('insertions', 0),
                        deletions=stats.get('deletions', 0)
                    ))
            
            return commits
            
        except Exception as e:
            self.logger.error(f"Failed to get file history for {file_path}: {e}")
            return []
    
    def create_tag(self, tag_name: str, message: Optional[str] = None) -> bool:
        """Create a tag"""
        if not self.is_initialized():
            return False
        
        try:
            if message:
                self.repo.create_tag(tag_name, message=message)
            else:
                self.repo.create_tag(tag_name)
            
            self.logger.info(f"Created tag: {tag_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create tag {tag_name}: {e}")
            return False
    
    def reset_changes(self, hard: bool = False) -> bool:
        """Reset changes in working directory"""
        if not self.is_initialized():
            return False
        
        try:
            if hard:
                self.repo.git.reset("--hard", "HEAD")
                self.logger.warning("Performed hard reset")
            else:
                self.repo.git.reset("--soft", "HEAD")
                self.logger.info("Performed soft reset")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reset changes: {e}")
            return False
    
    def stash_changes(self, message: Optional[str] = None) -> bool:
        """Stash current changes"""
        if not self.is_initialized():
            return False
        
        try:
            if message:
                self.repo.git.stash("save", message)
            else:
                self.repo.git.stash("save")
            
            self.logger.info("Stashed changes")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stash changes: {e}")
            return False
    
    def apply_stash(self, stash_id: str = "0") -> bool:
        """Apply stashed changes"""
        if not self.is_initialized():
            return False
        
        try:
            self.repo.git.stash("apply", stash_id)
            self.logger.info(f"Applied stash {stash_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply stash {stash_id}: {e}")
            return False
    
    def merge_branch(self, branch_name: str, strategy: str = "recursive") -> Dict[str, Any]:
        """
        Merge a branch into current branch
        
        Returns:
            Dictionary with merge result
        """
        if not self.is_initialized():
            return {"success": False, "error": "Git not initialized"}
        
        try:
            result = self.repo.git.merge(branch_name, strategy=strategy)
            
            return {
                "success": True,
                "message": f"Merged {branch_name} into {self.repo.active_branch.name}",
                "output": result
            }
            
        except GitCommandError as e:
            if "CONFLICT" in str(e):
                # Handle merge conflict
                conflicts = self._get_merge_conflicts()
                return {
                    "success": False,
                    "error": "Merge conflict",
                    "conflicts": conflicts,
                    "message": "Manual resolution required"
                }
            else:
                return {
                    "success": False,
                    "error": str(e)
                }
    
    def _get_merge_conflicts(self) -> List[str]:
        """Get list of files with merge conflicts"""
        if not self.repo:
            return []
        
        conflicts = []
        for file in self.repo.index.unmerged_blobs():
            conflicts.append(file)
        
        return conflicts
    
    def abort_merge(self) -> bool:
        """Abort a merge in progress"""
        if not self.is_initialized():
            return False
        
        try:
            self.repo.git.merge("--abort")
            self.logger.info("Aborted merge")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to abort merge: {e}")
            return False
    
    def setup_hooks(self) -> bool:
        """Setup Git hooks for ASEA-X"""
        if not self.is_initialized():
            return False
        
        try:
            hooks_dir = self.workdir / ".git" / "hooks"
            hooks_dir.mkdir(exist_ok=True)
            
            # Pre-commit hook
            pre_commit_hook = hooks_dir / "pre-commit"
            pre_commit_content = """#!/bin/bash
# ASEA-X Pre-commit Hook

echo "Running ASEA-X pre-commit checks..."

# Run linting if available
if command -v ruff &> /dev/null; then
    ruff check --fix .
    if [ $? -ne 0 ]; then
        echo "Linting failed. Commit aborted."
        exit 1
    fi
fi

echo "Pre-commit checks passed."
exit 0
"""
            pre_commit_hook.write_text(pre_commit_content)
            pre_commit_hook.chmod(0o755)
            
            # Post-commit hook
            post_commit_hook = hooks_dir / "post-commit"
            post_commit_content = """#!/bin/bash
# ASEA-X Post-commit Hook

echo "Commit completed successfully."
echo "Use 'git log --oneline -5' to see recent commits."
"""
            post_commit_hook.write_text(post_commit_content)
            post_commit_hook.chmod(0o755)
            
            self.logger.info("Git hooks installed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup Git hooks: {e}")
            return False

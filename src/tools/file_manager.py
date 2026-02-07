"""
File Manager - File operations with safety checks
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

from src.core.safety_system import SafetySystem, SafetyCheckResult


class FileManager:
    """
    Manages file operations with safety checks
    
    Features:
    1. Safe file reading/writing
    2. Directory operations
    3. File search and filtering
    4. Backup and restore
    5. Permission checking
    """
    
    def __init__(self, safety_system: Optional[SafetySystem] = None):
        self.safety_system = safety_system
        self.logger = logging.getLogger(__name__)
        
        # Operation history
        self.history: List[Dict[str, Any]] = []
        
        # Backup directory
        self.backup_dir = Path("./backups")
        self.backup_dir.mkdir(exist_ok=True)
    
    def read_file(self, file_path: str, encoding: str = "utf-8") -> Optional[str]:
        """
        Read file with safety check
        
        Args:
            file_path: Path to file
            encoding: File encoding
            
        Returns:
            File content or None if failed
        """
        path = Path(file_path)
        
        # Safety check
        if self.safety_system:
            safety_result = self.safety_system.check_file_operation(
                "read", source=path
            )
            if not safety_result.allowed:
                self.logger.error(f"Cannot read {file_path}: {safety_result.reason}")
                return None
        
        try:
            if not path.exists():
                self.logger.error(f"File not found: {file_path}")
                return None
            
            content = path.read_text(encoding=encoding, errors='ignore')
            
            # Log operation
            self._log_operation("read", file_path, success=True)
            
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to read {file_path}: {e}")
            self._log_operation("read", file_path, success=False, error=str(e))
            return None
    
    def write_file(
        self, 
        file_path: str, 
        content: str, 
        encoding: str = "utf-8",
        create_backup: bool = True
    ) -> bool:
        """
        Write file with safety check and optional backup
        
        Args:
            file_path: Path to file
            content: Content to write
            encoding: File encoding
            create_backup: Whether to create backup
            
        Returns:
            True if successful
        """
        path = Path(file_path)
        
        # Safety check
        if self.safety_system:
            safety_result = self.safety_system.check_file_operation(
                "write", target=path
            )
            if not safety_result.allowed:
                self.logger.error(f"Cannot write {file_path}: {safety_result.reason}")
                return False
        
        try:
            # Create backup if file exists
            if create_backup and path.exists():
                self._create_backup(file_path)
            
            # Create parent directories if needed
            path.parent.mkdir(exist_ok=True, parents=True)
            
            # Write file
            path.write_text(content, encoding=encoding)
            
            # Log operation
            self._log_operation("write", file_path, success=True, size=len(content))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write {file_path}: {e}")
            self._log_operation("write", file_path, success=False, error=str(e))
            return False
    
    def copy_file(self, source: str, destination: str) -> bool:
        """
        Copy file with safety checks
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful
        """
        src_path = Path(source)
        dst_path = Path(destination)
        
        # Safety checks
        if self.safety_system:
            # Check read permission
            read_result = self.safety_system.check_file_operation(
                "read", source=src_path
            )
            if not read_result.allowed:
                self.logger.error(f"Cannot read {source}: {read_result.reason}")
                return False
            
            # Check write permission
            write_result = self.safety_system.check_file_operation(
                "write", target=dst_path
            )
            if not write_result.allowed:
                self.logger.error(f"Cannot write {destination}: {write_result.reason}")
                return False
        
        try:
            if not src_path.exists():
                self.logger.error(f"Source file not found: {source}")
                return False
            
            # Create parent directories
            dst_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Copy file
            shutil.copy2(source, destination)
            
            # Log operation
            self._log_operation("copy", f"{source} -> {destination}", success=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to copy {source} to {destination}: {e}")
            self._log_operation("copy", f"{source} -> {destination}", success=False, error=str(e))
            return False
    
    def delete_file(self, file_path: str, create_backup: bool = True) -> bool:
        """
        Delete file with safety check and optional backup
        
        Args:
            file_path: Path to file
            create_backup: Whether to create backup
            
        Returns:
            True if successful
        """
        path = Path(file_path)
        
        # Safety check
        if self.safety_system:
            safety_result = self.safety_system.check_file_operation(
                "delete", source=path
            )
            if not safety_result.allowed:
                self.logger.error(f"Cannot delete {file_path}: {safety_result.reason}")
                return False
        
        try:
            if not path.exists():
                self.logger.warning(f"File not found for deletion: {file_path}")
                return False
            
            # Create backup if requested
            if create_backup:
                self._create_backup(file_path)
            
            # Delete file
            path.unlink()
            
            # Log operation
            self._log_operation("delete", file_path, success=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete {file_path}: {e}")
            self._log_operation("delete", file_path, success=False, error=str(e))
            return False
    
    def move_file(self, source: str, destination: str, create_backup: bool = True) -> bool:
        """
        Move file with safety checks
        
        Args:
            source: Source file path
            destination: Destination file path
            create_backup: Whether to create backup
            
        Returns:
            True if successful
        """
        src_path = Path(source)
        dst_path = Path(destination)
        
        # Safety checks
        if self.safety_system:
            # Check read permission
            read_result = self.safety_system.check_file_operation(
                "read", source=src_path
            )
            if not read_result.allowed:
                self.logger.error(f"Cannot read {source}: {read_result.reason}")
                return False
            
            # Check write permission for destination
            write_result = self.safety_system.check_file_operation(
                "write", target=dst_path
            )
            if not write_result.allowed:
                self.logger.error(f"Cannot write {destination}: {write_result.reason}")
                return False
            
            # Check delete permission for source (since it will be moved)
            delete_result = self.safety_system.check_file_operation(
                "delete", source=src_path
            )
            if not delete_result.allowed:
                self.logger.error(f"Cannot delete {source}: {delete_result.reason}")
                return False
        
        try:
            if not src_path.exists():
                self.logger.error(f"Source file not found: {source}")
                return False
            
            # Create backup if requested
            if create_backup:
                self._create_backup(source)
            
            # Create parent directories
            dst_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Move file
            shutil.move(source, destination)
            
            # Log operation
            self._log_operation("move", f"{source} -> {destination}", success=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to move {source} to {destination}: {e}")
            self._log_operation("move", f"{source} -> {destination}", success=False, error=str(e))
            return False
    
    def list_directory(
        self, 
        directory_path: str, 
        pattern: str = "*",
        recursive: bool = False
    ) -> List[str]:
        """
        List files in directory
        
        Args:
            directory_path: Directory path
            pattern: Glob pattern
            recursive: Whether to search recursively
            
        Returns:
            List of file paths
        """
        path = Path(directory_path)
        
        if not path.exists():
            self.logger.error(f"Directory not found: {directory_path}")
            return []
        
        if not path.is_dir():
            self.logger.error(f"Not a directory: {directory_path}")
            return []
        
        try:
            if recursive:
                files = [str(p) for p in path.rglob(pattern) if p.is_file()]
            else:
                files = [str(p) for p in path.glob(pattern) if p.is_file()]
            
            # Log operation
            self._log_operation("list", directory_path, success=True, count=len(files))
            
            return files
            
        except Exception as e:
            self.logger.error(f"Failed to list directory {directory_path}: {e}")
            return []
    
    def create_directory(self, directory_path: str) -> bool:
        """
        Create directory
        
        Args:
            directory_path: Directory path
            
        Returns:
            True if successful
        """
        path = Path(directory_path)
        
        try:
            if path.exists():
                if path.is_dir():
                    self.logger.info(f"Directory already exists: {directory_path}")
                    return True
                else:
                    self.logger.error(f"Path exists but is not a directory: {directory_path}")
                    return False
            
            # Create directory
            path.mkdir(parents=True, exist_ok=True)
            
            # Log operation
            self._log_operation("mkdir", directory_path, success=True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create directory {directory_path}: {e}")
            self._log_operation("mkdir", directory_path, success=False, error=str(e))
            return False
    
    def delete_directory(self, directory_path: str, recursive: bool = False) -> bool:
        """
        Delete directory with safety check
        
        Args:
            directory_path: Directory path
            recursive: Whether to delete recursively
            
        Returns:
            True if successful
        """
        path = Path(directory_path)
        
        # Safety check - prevent deletion of important directories
        if self.safety_system:
            # Check if directory is system directory
            system_dirs = ['/', '/etc', '/bin', '/usr', '/lib', '/var', '/home']
            for sys_dir in system_dirs:
                if str(path).startswith(sys_dir) and path != Path.cwd():
                    self.logger.error(f"Cannot delete system directory: {directory_path}")
                    return False
        
        try:
            if not path.exists():
                self.logger.warning(f"Directory not found: {directory_path}")
                return False
            
            if not path.is_dir():
                self.logger.error(f"Not a directory: {directory_path}")
                return False
            
            # Check if directory is empty (if not recursive)
            if not recursive and any(path.iterdir()):
                self.logger.error(f"Directory not empty: {directory_path}")
                return False
            
            # Delete directory
            if recursive:
                shutil.rmtree(directory_path)
            else:
                path.rmdir()
            
            # Log operation
            self._log_operation("rmdir", directory_path, success=True, recursive=recursive)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete directory {directory_path}: {e}")
            self._log_operation("rmdir", directory_path, success=False, error=str(e))
            return False
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get file information
        
        Args:
            file_path: File path
            
        Returns:
            Dictionary with file info or None
        """
        path = Path(file_path)
        
        try:
            if not path.exists():
                return None
            
            stat = path.stat()
            
            return {
                "path": str(path),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "accessed": datetime.fromtimestamp(stat.st_atime),
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "is_symlink": path.is_symlink(),
                "permissions": oct(stat.st_mode)[-3:]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get file info for {file_path}: {e}")
            return None
    
    def search_files(
        self, 
        directory_path: str, 
        pattern: str = "*",
        content_filter: Optional[str] = None,
        recursive: bool = True
    ) -> List[Tuple[str, Optional[int]]]:
        """
        Search files with optional content filtering
        
        Args:
            directory_path: Directory to search
            pattern: Glob pattern
            content_filter: Text to search in file content
            recursive: Whether to search recursively
            
        Returns:
            List of (file_path, line_number) tuples
        """
        files = self.list_directory(directory_path, pattern, recursive)
        results = []
        
        for file_path in files:
            if content_filter:
                try:
                    content = self.read_file(file_path)
                    if content and content_filter in content:
                        # Find line numbers
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if content_filter in line:
                                results.append((file_path, i))
                    elif content and content_filter.lower() in content.lower():
                        # Case-insensitive match
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if content_filter.lower() in line.lower():
                                results.append((file_path, i))
                except:
                    continue
            else:
                results.append((file_path, None))
        
        return results
    
    def _create_backup(self, file_path: str) -> bool:
        """
        Create backup of file
        
        Args:
            file_path: File to backup
            
        Returns:
            True if backup created
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{path.name}.backup_{timestamp}"
            backup_path = self.backup_dir / backup_name
            
            # Copy to backup
            shutil.copy2(file_path, backup_path)
            
            self.logger.info(f"Created backup: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create backup for {file_path}: {e}")
            return False
    
    def _log_operation(
        self, 
        operation: str, 
        target: str, 
        success: bool, 
        **kwargs
    ):
        """Log file operation"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "target": target,
            "success": success,
            **kwargs
        }
        
        self.history.append(log_entry)
        
        # Keep only last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def get_operation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get file operation history"""
        return self.history[-limit:] if self.history else []
    
    def clear_history(self):
        """Clear operation history"""
        self.history.clear()

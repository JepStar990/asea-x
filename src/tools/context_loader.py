"""
File Context Loader with Vector Database
Loads, parses, indexes, and retrieves files
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
import pickle

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document


@dataclass
class FileMetadata:
    """Metadata for loaded files"""
    path: str
    size_bytes: int
    line_count: int
    language: Optional[str] = None
    sha256: Optional[str] = None
    last_modified: float = 0.0
    summary: Optional[str] = None
    tags: List[str] = field(default_factory=list)


class ContextLoader:
    """
    Loads, indexes, and retrieves files using vector database
    
    Features:
    1. File loading with metadata extraction
    2. Text chunking for LLM context
    3. Vector embeddings and similarity search
    4. File summarization
    5. Context-aware retrieval
    """
    
    def __init__(self, workdir: Path = Path("./workdir")):
        self.workdir = workdir
        self.workdir.mkdir(exist_ok=True, parents=True)
        
        self.logger = logging.getLogger(__name__)
        self.files: Dict[str, FileMetadata] = {}
        self.vector_store: Optional[FAISS] = None
        self.embeddings = None
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Load existing context if any
        self._load_context()
    
    def _init_embeddings(self):
        """Initialize embeddings model"""
        try:
            # Try to use OpenAI embeddings
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=None  # Will use environment variable
            )
        except:
            # Fallback to simple embeddings
            self.logger.warning("Using fallback embeddings")
            self.embeddings = None
    
    def _load_context(self):
        """Load existing context from disk"""
        context_file = self.workdir / "file_context.json"
        if context_file.exists():
            try:
                with open(context_file, 'r') as f:
                    data = json.load(f)
                    self.files = {
                        path: FileMetadata(**meta) 
                        for path, meta in data.get("files", {}).items()
                    }
                
                # Load vector store if exists
                vector_store_file = self.workdir / "vector_store.faiss"
                if vector_store_file.exists() and self.embeddings:
                    self.vector_store = FAISS.load_local(
                        str(self.workdir),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                self.logger.info(f"Loaded {len(self.files)} files from context")
            except Exception as e:
                self.logger.error(f"Failed to load context: {e}")
    
    def save_context(self):
        """Save context to disk"""
        try:
            # Save metadata
            context_file = self.workdir / "file_context.json"
            with open(context_file, 'w') as f:
                json.dump({
                    "files": {
                        path: {
                            "path": meta.path,
                            "size_bytes": meta.size_bytes,
                            "line_count": meta.line_count,
                            "language": meta.language,
                            "sha256": meta.sha256,
                            "last_modified": meta.last_modified,
                            "summary": meta.summary,
                            "tags": meta.tags
                        }
                        for path, meta in self.files.items()
                    }
                }, f, indent=2)
            
            # Save vector store
            if self.vector_store and self.embeddings:
                self.vector_store.save_local(str(self.workdir))
            
            self.logger.debug(f"Saved context with {len(self.files)} files")
        except Exception as e:
            self.logger.error(f"Failed to save context: {e}")
    
    def load_file(self, file_path: str) -> Optional[FileMetadata]:
        """
        Load a file into context
        
        Args:
            file_path: Path to the file
            
        Returns:
            FileMetadata if successful, None otherwise
        """
        try:
            path = Path(file_path)
            if not path.exists():
                self.logger.error(f"File not found: {file_path}")
                return None
            
            # Read file
            content = path.read_text(encoding='utf-8', errors='ignore')
            
            # Calculate hash
            sha256 = hashlib.sha256(content.encode()).hexdigest()
            
            # Check if already loaded with same content
            if file_path in self.files and self.files[file_path].sha256 == sha256:
                self.logger.info(f"File already loaded: {file_path}")
                return self.files[file_path]
            
            # Create metadata
            metadata = FileMetadata(
                path=file_path,
                size_bytes=len(content),
                line_count=len(content.splitlines()),
                language=self._detect_language(file_path),
                sha256=sha256,
                last_modified=path.stat().st_mtime,
                summary=self._generate_summary(content, file_path)
            )
            
            # Store in memory
            self.files[file_path] = metadata
            
            # Add to vector store
            self._add_to_vector_store(content, file_path, metadata)
            
            # Save context
            self.save_context()
            
            self.logger.info(f"Loaded file: {file_path} ({metadata.size_bytes} bytes)")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load file {file_path}: {e}")
            return None
    
    def load_directory(self, directory_path: str, pattern: str = "**/*") -> List[FileMetadata]:
        """
        Load all files in a directory
        
        Args:
            directory_path: Directory to load
            pattern: Glob pattern for files
            
        Returns:
            List of loaded file metadata
        """
        results = []
        path = Path(directory_path)
        
        if not path.exists():
            self.logger.error(f"Directory not found: {directory_path}")
            return results
        
        for file_path in path.glob(pattern):
            if file_path.is_file():
                try:
                    metadata = self.load_file(str(file_path))
                    if metadata:
                        results.append(metadata)
                except Exception as e:
                    self.logger.error(f"Failed to load {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(results)} files from {directory_path}")
        return results
    
    def unload_file(self, file_path: str) -> bool:
        """Remove file from context"""
        if file_path in self.files:
            del self.files[file_path]
            
            # Remove from vector store (simplified - would need proper removal)
            # For now, we'll just rebuild if needed
            
            self.save_context()
            self.logger.info(f"Unloaded file: {file_path}")
            return True
        
        return False
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get content of a loaded file"""
        if file_path in self.files:
            try:
                return Path(file_path).read_text(encoding='utf-8', errors='ignore')
            except:
                return None
        return None
    
    def search_files(
        self, 
        query: str, 
        limit: int = 5
    ) -> List[Tuple[str, float, Optional[str]]]:
        """
        Search files using semantic similarity
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of (file_path, similarity_score, snippet)
        """
        if not self.vector_store or not self.embeddings:
            # Fallback to simple text search
            return self._simple_search(query, limit)
        
        try:
            # Perform semantic search
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, 
                k=limit
            )
            
            results = []
            for doc, score in docs_with_scores:
                file_path = doc.metadata.get("source", "")
                snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                results.append((file_path, 1.0 - score, snippet))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return self._simple_search(query, limit)
    
    def _simple_search(self, query: str, limit: int = 5) -> List[Tuple[str, float, Optional[str]]]:
        """Simple text-based search fallback"""
        query_lower = query.lower()
        results = []
        
        for file_path, metadata in self.files.items():
            try:
                content = self.get_file_content(file_path)
                if not content:
                    continue
                
                # Check if query appears in content
                if query_lower in content.lower():
                    # Find snippet around query
                    idx = content.lower().find(query_lower)
                    start = max(0, idx - 50)
                    end = min(len(content), idx + len(query) + 50)
                    snippet = content[start:end]
                    
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(content):
                        snippet = snippet + "..."
                    
                    # Simple relevance score
                    occurrences = content.lower().count(query_lower)
                    score = min(1.0, occurrences * 0.1)
                    
                    results.append((file_path, score, snippet))
                    
                    if len(results) >= limit:
                        break
                        
            except:
                continue
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_relevant_context(
        self, 
        query: str, 
        max_chars: int = 4000
    ) -> str:
        """
        Get relevant context for a query
        
        Args:
            query: The query to find context for
            max_chars: Maximum characters to return
            
        Returns:
            Concatenated relevant context
        """
        search_results = self.search_files(query, limit=10)
        
        context_parts = []
        total_chars = 0
        
        for file_path, score, snippet in search_results:
            if score < 0.3:  # Relevance threshold
                continue
            
            content = self.get_file_content(file_path)
            if not content:
                continue
            
            # Truncate content if needed
            if len(content) > 1000:
                # Try to get most relevant part
                lines = content.split('\n')
                relevant_lines = []
                
                for line in lines:
                    if any(keyword.lower() in line.lower() 
                           for keyword in query.split()):
                        relevant_lines.append(line)
                
                if relevant_lines:
                    content = '\n'.join(relevant_lines[:20])  # First 20 relevant lines
                else:
                    content = content[:1000] + "\n...[truncated]"
            
            # Add to context
            context_parts.append(f"=== File: {file_path} (relevance: {score:.2f}) ===\n{content}")
            total_chars += len(content)
            
            if total_chars > max_chars:
                break
        
        return "\n\n".join(context_parts)
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension"""
        ext_map = {
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
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.html': 'html',
            '.css': 'css',
            '.md': 'markdown',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            '.sh': 'shell',
            '.bash': 'shell',
            '.zsh': 'shell',
            '.sql': 'sql',
            '.dockerfile': 'dockerfile',
            'dockerfile': 'dockerfile'
        }
        
        path = Path(file_path)
        
        # Check extension
        for ext, lang in ext_map.items():
            if path.suffix == ext:
                return lang
        
        # Check filename
        for name, lang in ext_map.items():
            if path.name.lower() == name or path.name.lower().endswith(name):
                return lang
        
        return None
    
    def _generate_summary(self, content: str, file_path: str) -> str:
        """Generate a summary of the file"""
        # Simple summary generation
        lines = content.split('\n')
        
        if len(lines) <= 10:
            return "Small file"
        
        # For code files, try to extract key elements
        language = self._detect_language(file_path)
        
        if language == 'python':
            # Extract imports, functions, classes
            imports = [l for l in lines if l.strip().startswith('import ') or l.strip().startswith('from ')]
            functions = [l for l in lines if l.strip().startswith('def ')]
            classes = [l for l in lines if l.strip().startswith('class ')]
            
            summary_parts = []
            if imports:
                summary_parts.append(f"{len(imports)} imports")
            if classes:
                summary_parts.append(f"{len(classes)} classes")
            if functions:
                summary_parts.append(f"{len(functions)} functions")
            
            if summary_parts:
                return ', '.join(summary_parts)
        
        # Generic summary
        return f"{len(lines)} lines, {len(content)} characters"
    
    def _add_to_vector_store(self, content: str, file_path: str, metadata: FileMetadata):
        """Add file to vector store"""
        if not self.embeddings:
            return
        
        try:
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_text(content)
            
            # Create documents
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path,
                        "language": metadata.language or "unknown",
                        "line_count": metadata.line_count
                    }
                )
                for chunk in chunks
            ]
            
            # Add to vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)
            
            self.logger.debug(f"Added {len(chunks)} chunks from {file_path} to vector store")
            
        except Exception as e:
            self.logger.error(f"Failed to add {file_path} to vector store: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded context"""
        total_size = sum(meta.size_bytes for meta in self.files.values())
        languages = {}
        
        for meta in self.files.values():
            lang = meta.language or "unknown"
            languages[lang] = languages.get(lang, 0) + 1
        
        return {
            "total_files": len(self.files),
            "total_size_bytes": total_size,
            "languages": languages,
            "has_vector_store": self.vector_store is not None
        }
    
    def clear_context(self):
        """Clear all loaded context"""
        self.files.clear()
        self.vector_store = None
        
        # Delete saved files
        context_file = self.workdir / "file_context.json"
        if context_file.exists():
            context_file.unlink()
        
        vector_files = [
            self.workdir / "vector_store.faiss",
            self.workdir / "vector_store.pkl"
        ]
        for file in vector_files:
            if file.exists():
                file.unlink()
        
        self.logger.info("Cleared all file context")

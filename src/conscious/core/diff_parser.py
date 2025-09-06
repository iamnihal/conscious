"""Syntax-aware diff parsing using Difftastic with parallel processing."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from ..parsers.difftastic_parser import DifftasticParser
from ..parsers.tree_sitter_parser import TreeSitterParser
from ..analyzers.call_graph_analyzer import CallGraphAnalyzer, CallGraph

@dataclass
class DiffChange:
    """Represents a single change in a diff."""
    old_start: int
    old_end: int
    new_start: int
    new_end: int
    content: str

@dataclass
class DiffFile:
    """Represents a file in a diff."""
    path: str
    language: str
    changes: List[DiffChange]

class DiffParser:
    """Parses diffs using Difftastic for syntax-aware analysis."""
    
    def __init__(self, diff_path: str, max_workers: Optional[int] = None, cache_enabled: bool = True, cache_size: int = 100):
        self.diff_path = diff_path
        self.difft = DifftasticParser()
        self.max_workers = max_workers or os.cpu_count() or 4

        # Caching configuration
        self.cache_enabled = cache_enabled
        self.cache_size = cache_size

        # Cache storage with LRU tracking
        self._file_content_cache = {}  # Cache for extracted file contents
        self._tree_cache = {}  # Cache for Tree-sitter parse trees
        self._call_graph_cache = {}  # Cache for generated call graphs

        # LRU tracking (simple implementation using insertion order)
        self._cache_access_order = []

        # Lazy-loaded components
        self._tree_sitter = None
        self._analyzer = None

    @property
    def tree_sitter(self):
        """Lazy initialization of Tree-sitter parser."""
        if self._tree_sitter is None:
            self._tree_sitter = TreeSitterParser()
        return self._tree_sitter

    @property
    def analyzer(self):
        """Lazy initialization of call graph analyzer."""
        if self._analyzer is None:
            self._analyzer = CallGraphAnalyzer()
        return self._analyzer

    def _cache_get(self, cache_dict, key):
        """Get item from cache with LRU tracking."""
        if not self.cache_enabled:
            return None

        if key in cache_dict:
            # Move to end for LRU (most recently used)
            if key in self._cache_access_order:
                self._cache_access_order.remove(key)
            self._cache_access_order.append(key)
            return cache_dict[key]
        return None

    def _cache_set(self, cache_dict, key, value):
        """Set item in cache with LRU management."""
        if not self.cache_enabled:
            return

        # Add to cache
        cache_dict[key] = value

        # Update LRU order
        if key in self._cache_access_order:
            self._cache_access_order.remove(key)
        self._cache_access_order.append(key)

        # Maintain cache size limit
        while len(cache_dict) > self.cache_size:
            # Remove least recently used item
            oldest_key = self._cache_access_order.pop(0)
            if oldest_key in cache_dict:
                del cache_dict[oldest_key]

    def clear_cache(self):
        """Clear all caches."""
        self._file_content_cache.clear()
        self._tree_cache.clear()
        self._call_graph_cache.clear()
        self._cache_access_order.clear()

    def get_cache_stats(self):
        """Get cache statistics."""
        return {
            'file_content_cache': len(self._file_content_cache),
            'tree_cache': len(self._tree_cache),
            'call_graph_cache': len(self._call_graph_cache),
            'total_cached_items': len(self._file_content_cache) + len(self._tree_cache) + len(self._call_graph_cache),
            'cache_size_limit': self.cache_size,
            'cache_enabled': self.cache_enabled
        }

    def _extract_file_contents(self, file_path: str, raw_diff: str) -> tuple[str, str]:
        """Extract old and new contents of a file from the diff."""
        old_lines = []
        new_lines = []
        current_lines = []
        in_file = False
        in_hunk = False
        
        # Normalize the file path for comparison
        normalized_path = self._normalize_path(file_path)
        
        for line in raw_diff.splitlines():
            # Check for file headers
            if line.startswith('--- '):
                old_file = self._normalize_path(line[4:].strip())
                if old_file == normalized_path:
                    in_file = True
                    current_lines = []
                else:
                    in_file = False
                continue
            elif line.startswith('+++ '):
                new_file = self._normalize_path(line[4:].strip())
                if new_file == normalized_path:
                    in_file = True
                    if not current_lines:  # Only reset if not already set by ---
                        current_lines = []
                else:
                    in_file = False
                continue
            elif line.startswith('@@'):
                # Only process hunks for the current file
                if in_file:
                    in_hunk = True
                continue
                
            if in_file and in_hunk:
                if line.startswith('-'):
                    old_lines.append(line[1:])
                    current_lines.append(line[1:])
                elif line.startswith('+'):
                    new_lines.append(line[1:])
                    current_lines.append(line[1:])
                else:
                    old_lines.append(line)
                    new_lines.append(line)
                    current_lines.append(line)
        
        # If no changes were found, use current content for both
        if not old_lines and not new_lines and current_lines:
            old_lines = current_lines
            new_lines = current_lines
            
        return '\n'.join(old_lines), '\n'.join(new_lines)

    def _normalize_path(self, file_path: str) -> str:
        """Normalize file path by removing a/ or b/ prefixes."""
        # Handle git-style paths
        if file_path.startswith('a/') or file_path.startswith('b/'):
            return file_path[2:]
        # Handle paths with no prefix
        if file_path.startswith('/'):
            return file_path[1:]
        return file_path

    def _process_file(self, file_path: str, normalized_diff: str) -> Optional[DiffFile]:
        """Process a single file from the diff."""
        try:
            if file_path == '/dev/null':  # New file
                return None
                
            # Get file contents
            old_content, new_content = self._extract_file_contents(
                file_path, normalized_diff)
            
            # Use Difftastic for syntax-aware analysis
            language = self._detect_language(file_path)
            difft_changes = self.difft.parse_diff(
                old_content, new_content, language)
            
            # Convert Difftastic changes to our format
            changes = [
                DiffChange(
                    old_start=change.old_start,
                    old_end=change.old_end,
                    new_start=change.new_start,
                    new_end=change.new_end,
                    content=change.content
                )
                for change in difft_changes
            ]
            
            if changes:  # Only return files that have changes
                return DiffFile(
                    path=self._normalize_path(file_path),
                    language=language,
                    changes=changes
                )
            
            return None
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            return None

    def parse(self) -> List[DiffFile]:
        """Parse the diff file using Difftastic for syntax-aware analysis."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'cp1252']
            raw_diff = None
            
            for encoding in encodings:
                try:
                    with open(self.diff_path, 'r', encoding=encoding) as f:
                        raw_diff = f.read()
                    break
                except UnicodeDecodeError:
                    continue
                    
            if raw_diff is None:
                raise RuntimeError(f"Could not decode {self.diff_path} with any supported encoding")
                
            # Normalize the diff format
            normalized_diff = self._normalize_diff(raw_diff)
            
            # Extract unique file paths (ignoring a/ and b/ prefixes)
            file_paths = set()
            for line in normalized_diff.splitlines():
                if line.startswith('--- '):
                    file_path = line[4:].strip()
                    if file_path != '/dev/null':
                        file_paths.add(self._normalize_path(file_path))
                elif line.startswith('+++ '):
                    file_path = line[4:].strip()
                    if file_path != '/dev/null':
                        file_paths.add(self._normalize_path(file_path))
            
            # Process files in parallel
            files: List[DiffFile] = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                try:
                    # Submit all file processing tasks
                    future_to_file = {
                        executor.submit(self._process_file, file_path, normalized_diff): file_path
                        for file_path in file_paths
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            result = future.result()
                            if result is not None:
                                files.append(result)
                        except Exception as e:
                            print(f"Error processing {file_path}: {str(e)}")
                except KeyboardInterrupt:
                    print("\nInterrupted. Cleaning up...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
            
            return files

        except KeyboardInterrupt:
            print("\nParsing interrupted by user")
            raise

    def generate_call_graphs(self, diff_files: List[DiffFile]) -> List[CallGraph]:
        """Generate call graphs for the parsed diff files with caching."""
        call_graphs = []

        for diff_file in diff_files:
            try:
                # Create cache key for this file
                cache_key = f"{diff_file.path}:{diff_file.language}"

                # Check call graph cache first
                cached_call_graph = self._cache_get(self._call_graph_cache, cache_key)
                if cached_call_graph is not None:
                    call_graphs.append(cached_call_graph)
                    continue

                # Extract old and new content from the diff (with caching)
                content_cache_key = f"{diff_file.path}:content"
                cached_content = self._cache_get(self._file_content_cache, content_cache_key)

                if cached_content is not None:
                    old_content, new_content = cached_content
                else:
                    old_content, new_content = self._extract_file_contents(diff_file.path, self._read_raw_diff())
                    self._cache_set(self._file_content_cache, content_cache_key, (old_content, new_content))

                # Use new content for call graph analysis (assuming we're analyzing current state)
                if new_content.strip():
                    # Check tree cache
                    tree_cache_key = f"{diff_file.path}:{diff_file.language}:tree"
                    cached_tree = self._cache_get(self._tree_cache, tree_cache_key)

                    if cached_tree is not None:
                        tree = cached_tree
                    else:
                        # Parse with Tree-sitter
                        tree = self.tree_sitter.parse_file(new_content, diff_file.language)
                        if tree is not None:
                            self._cache_set(self._tree_cache, tree_cache_key, tree)

                    if tree is not None:
                        # Extract code elements
                        functions = self.tree_sitter.get_functions(tree, diff_file.language)
                        classes = self.tree_sitter.get_classes(tree, diff_file.language)
                        calls = self.tree_sitter.get_calls(tree, diff_file.language)
                        imports = self.tree_sitter.get_imports(tree, diff_file.language)

                        # Build call graph
                        call_graph = self.analyzer.build_graph(
                            functions=functions,
                            calls=calls,
                            classes=classes,
                            imports=imports,
                            file_path=diff_file.path,
                            language=diff_file.language
                        )

                        # Cache the call graph
                        self._cache_set(self._call_graph_cache, cache_key, call_graph)

                        call_graphs.append(call_graph)

            except Exception as e:
                print(f"Error generating call graph for {diff_file.path}: {str(e)}")

        return call_graphs

    def _read_raw_diff(self) -> str:
        """Read the raw diff file content."""
        encodings = ['utf-8', 'latin1', 'cp1252']
        for encoding in encodings:
            try:
                with open(self.diff_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"Could not decode {self.diff_path} with any supported encoding")
        
    def _detect_language(self, file_path: str) -> str:
        """Detect the programming language of a file based on extension."""
        ext = file_path.lower().split('.')[-1]
        
        # Language mapping
        LANGUAGE_MAP = {
            # Python
            'py': 'python',
            'pyi': 'python',
            
            # JavaScript/TypeScript
            'js': 'javascript',
            'jsx': 'javascript',
            'ts': 'typescript',
            'tsx': 'typescript',
            
            # Java
            'java': 'java',
            
            # Config files
            'json': 'json',
            'yaml': 'yaml',
            'yml': 'yaml',
            'toml': 'toml',
        }
        
        return LANGUAGE_MAP.get(ext, 'unknown')
        
    def _normalize_diff(self, raw_diff: str) -> str:
        """Normalize different diff formats to a standard unified diff format."""
        lines = raw_diff.splitlines()
        normalized = []
        
        for line in lines:
            # Skip index lines and other metadata
            if line.startswith('index ') or line.startswith('diff '):
                continue
                
            # Keep file paths, but standardize format
            if line.startswith('--- ') or line.startswith('+++ '):
                normalized.append(line)
                continue
                
            # Keep hunk headers and content
            if line.startswith('@@ '):
                normalized.append(line)
                continue
                
            # Keep actual changes
            if line.startswith('+') or line.startswith('-') or line.startswith(' '):
                normalized.append(line)
        
        return '\n'.join(normalized)

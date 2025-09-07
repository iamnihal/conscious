"""Import Graph Builder - Builds dependency graphs from import statements."""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple, Any
from pathlib import Path
import re
import ast
import os


@dataclass
class ImportNode:
    """Represents a file node in the import graph."""
    file_path: str
    module_name: str
    is_package: bool = False
    imports: List['ImportEdge'] = None
    imported_by: List['ImportEdge'] = None

    def __post_init__(self):
        if self.imports is None:
            self.imports = []
        if self.imported_by is None:
            self.imported_by = []


@dataclass
class ImportEdge:
    """Represents an import relationship between files."""
    from_file: str
    to_file: str
    import_type: str  # 'absolute', 'relative', 'from_import', 'import'
    imported_symbols: List[str] = None
    line_number: Optional[int] = None
    confidence: float = 1.0

    def __post_init__(self):
        if self.imported_symbols is None:
            self.imported_symbols = []


@dataclass
class ImportGraph:
    """Complete import dependency graph."""
    nodes: Dict[str, ImportNode]
    edges: List[ImportEdge]
    cycles: List[List[str]] = None
    root_directory: str = ""

    def __post_init__(self):
        if self.cycles is None:
            self.cycles = []


class ImportGraphBuilder:
    """Builds import dependency graphs from Python files."""

    def __init__(self):
        self.file_contents = {}  # Cache for file contents
        self.module_map = {}  # Maps module names to file paths

    def build_graph(self, file_paths: List[str], root_directory: str = "") -> ImportGraph:
        """Build complete import graph from a list of Python files."""
        self.root_directory = root_directory or os.getcwd()
        self._build_module_map(file_paths)

        nodes = {}
        edges = []

        # Create nodes for all files
        for file_path in file_paths:
            if file_path.endswith('.py'):
                module_name = self._file_path_to_module_name(file_path, root_directory)
                nodes[file_path] = ImportNode(
                    file_path=file_path,
                    module_name=module_name,
                    is_package=self._is_package_directory(file_path)
                )

        # Parse imports and create edges
        for file_path in file_paths:
            if file_path.endswith('.py'):
                file_edges = self._parse_file_imports(file_path, root_directory)
                edges.extend(file_edges)

                # Update node relationships
                for edge in file_edges:
                    if edge.from_file in nodes and edge.to_file in nodes:
                        nodes[edge.from_file].imports.append(edge)
                        nodes[edge.to_file].imported_by.append(edge)

        # Detect cycles
        cycles = self._detect_cycles(nodes)

        return ImportGraph(
            nodes=nodes,
            edges=edges,
            cycles=cycles,
            root_directory=root_directory
        )

    def _build_module_map(self, file_paths: List[str]):
        """Build mapping from module names to file paths."""
        self.module_map = {}

        for file_path in file_paths:
            if file_path.endswith('.py'):
                module_name = self._file_path_to_module_name(file_path, self.root_directory)
                self.module_map[module_name] = file_path

                # Also map without .py extension
                if module_name.endswith('.py'):
                    self.module_map[module_name[:-3]] = file_path

    def _file_path_to_module_name(self, file_path: str, root_directory: str) -> str:
        """Convert file path to Python module name."""
        # Remove root directory and .py extension
        rel_path = os.path.relpath(file_path, root_directory)
        if rel_path.startswith('..'):
            rel_path = file_path  # Use absolute path if outside root

        module_name = rel_path.replace(os.sep, '.')
        if module_name.endswith('.py'):
            module_name = module_name[:-3]

        return module_name

    def _is_package_directory(self, file_path: str) -> bool:
        """Check if file is in a package directory (has __init__.py)."""
        directory = os.path.dirname(file_path)
        init_file = os.path.join(directory, '__init__.py')
        return os.path.exists(init_file)

    def _parse_file_imports(self, file_path: str, root_directory: str) -> List[ImportEdge]:
        """Parse import statements from a Python file."""
        try:
            content = self._get_file_content(file_path)
            tree = ast.parse(content, filename=file_path)
            return self._extract_imports_from_ast(tree, file_path, root_directory)
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
            return []

    def _get_file_content(self, file_path: str) -> str:
        """Get file content with caching."""
        if file_path not in self.file_contents:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.file_contents[file_path] = f.read()
            except (UnicodeDecodeError, FileNotFoundError):
                self.file_contents[file_path] = ""

        return self.file_contents[file_path]

    def _extract_imports_from_ast(self, tree: ast.AST, file_path: str,
                                root_directory: str) -> List[ImportEdge]:
        """Extract import statements from AST."""
        edges = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # import module
                for alias in node.names:
                    edge = self._resolve_import(
                        file_path, alias.name, [], 'import',
                        root_directory, node.lineno
                    )
                    if edge:
                        edges.append(edge)

            elif isinstance(node, ast.ImportFrom):
                # from module import symbol
                module_name = node.module or ''
                # Handle relative imports by adding dots based on level
                if node.level > 0:
                    module_name = '.' * node.level + module_name
                symbols = [alias.name for alias in node.names]

                edge = self._resolve_import(
                    file_path, module_name, symbols, 'from_import',
                    root_directory, node.lineno
                )
                if edge:
                    edges.append(edge)

        return edges

    def _resolve_import(self, from_file: str, module_name: str, symbols: List[str],
                       import_type: str, root_directory: str, line_number: int) -> Optional[ImportEdge]:
        """Resolve import to target file path."""
        if not module_name:
            return None

        # Try to find the module in our module map
        target_file = self._find_module_file(module_name, from_file, root_directory)

        if target_file:
            return ImportEdge(
                from_file=from_file,
                to_file=target_file,
                import_type=import_type,
                imported_symbols=symbols,
                line_number=line_number,
                confidence=0.9 if import_type == 'from_import' else 0.8
            )

        return None

    def _find_module_file(self, module_name: str, from_file: str, root_directory: str) -> Optional[str]:
        """Find the file path for a given module name."""
        # Direct lookup in module map
        if module_name in self.module_map:
            return self.module_map[module_name]

        # Try relative imports
        if module_name.startswith('.'):
            return self._resolve_relative_import(module_name, from_file, root_directory)

        # Try absolute imports with different extensions
        for ext in ['', '.py']:
            full_module = f"{module_name}{ext}"
            if full_module in self.module_map:
                return self.module_map[full_module]

        # Try package resolution (look for __init__.py)
        package_path = self._find_package_init(module_name, root_directory)
        if package_path:
            return package_path

        return None

    def _resolve_relative_import(self, module_name: str, from_file: str, root_directory: str) -> Optional[str]:
        """Resolve relative import to absolute path."""
        from_dir = os.path.dirname(from_file)
        dots_count = len(module_name) - len(module_name.lstrip('.'))
        module_part = module_name[dots_count:]

        # For relative imports:
        # - 1 dot (.subpackage) means relative to current directory
        # - 2 dots (..subpackage) means relative to parent directory
        # So we go up (dots_count - 1) levels
        current_dir = from_dir
        for _ in range(dots_count - 1):  # -1 because first dot is relative to current dir
            current_dir = os.path.dirname(current_dir)

        if module_part:
            # Replace dots with path separators and join with current directory
            module_path = module_part.replace('.', os.sep)
            target_path = os.path.join(current_dir, module_path)
        else:
            # Import from current/parent directory (e.g., "from . import something")
            target_path = current_dir

        # Look for Python file or package
        for ext in ['.py', '/__init__.py']:
            candidate = f"{target_path}{ext}"
            if os.path.exists(candidate):
                return candidate

        # Check if it's a directory with __init__.py
        if os.path.isdir(target_path):
            init_file = os.path.join(target_path, '__init__.py')
            if os.path.exists(init_file):
                return init_file

        return None

    def _find_package_init(self, module_name: str, root_directory: str) -> Optional[str]:
        """Find __init__.py file for a package."""
        module_path = module_name.replace('.', os.sep)
        init_path = os.path.join(root_directory, module_path, '__init__.py')

        if os.path.exists(init_path):
            return init_path

        return None

    def _detect_cycles(self, nodes: Dict[str, ImportNode]) -> List[List[str]]:
        """Detect import cycles in the graph."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node_path: str, path: List[str]):
            if node_path in rec_stack:
                # Found cycle
                cycle_start = path.index(node_path)
                cycles.append(path[cycle_start:] + [node_path])
                return

            if node_path in visited:
                return

            visited.add(node_path)
            rec_stack.add(node_path)
            path.append(node_path)

            if node_path in nodes:
                for edge in nodes[node_path].imports:
                    dfs(edge.to_file, path)

            path.pop()
            rec_stack.remove(node_path)

        for node_path in nodes:
            if node_path not in visited:
                dfs(node_path, [])

        return cycles

    def get_dependencies(self, graph: ImportGraph, file_path: str) -> List[str]:
        """Get all files that the given file depends on."""
        if file_path not in graph.nodes:
            return []

        dependencies = set()
        to_visit = [file_path]
        visited = set()

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue

            visited.add(current)

            if current in graph.nodes:
                for edge in graph.nodes[current].imports:
                    if edge.to_file not in dependencies:
                        dependencies.add(edge.to_file)
                        to_visit.append(edge.to_file)

        return list(dependencies)

    def get_dependents(self, graph: ImportGraph, file_path: str) -> List[str]:
        """Get all files that depend on the given file."""
        if file_path not in graph.nodes:
            return []

        dependents = set()
        to_visit = [file_path]
        visited = set()

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue

            visited.add(current)

            if current in graph.nodes:
                for edge in graph.nodes[current].imported_by:
                    if edge.from_file not in dependents:
                        dependents.add(edge.from_file)
                        to_visit.append(edge.from_file)

        return list(dependents)

    def get_import_summary(self, graph: ImportGraph) -> Dict:
        """Get summary statistics of the import graph."""
        return {
            'total_files': len(graph.nodes),
            'total_imports': len(graph.edges),
            'cycles_detected': len(graph.cycles),
            'import_types': self._count_import_types(graph.edges),
            'most_imported': self._get_most_imported(graph),
            'most_importing': self._get_most_importing(graph)
        }

    def _count_import_types(self, edges: List[ImportEdge]) -> Dict[str, int]:
        """Count different types of imports."""
        counts = {}
        for edge in edges:
            counts[edge.import_type] = counts.get(edge.import_type, 0) + 1
        return counts

    def _get_most_imported(self, graph: ImportGraph) -> List[Tuple[str, int]]:
        """Get files that are most imported by others."""
        import_counts = {}
        for node in graph.nodes.values():
            import_counts[node.file_path] = len(node.imported_by)

        return sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    def _get_most_importing(self, graph: ImportGraph) -> List[Tuple[str, int]]:
        """Get files that import the most other files."""
        import_counts = {}
        for node in graph.nodes.values():
            import_counts[node.file_path] = len(node.imports)

        return sorted(import_counts.items(), key=lambda x: x[1], reverse=True)[:10]

"""Call graph generation and analysis."""

from typing import Dict, List, Set, Optional, Tuple
import networkx as nx
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Node:
    """Represents a node in the call graph."""
    id: str
    name: str
    type: str  # function, class, or module
    file: str
    language: str
    start_line: int
    end_line: int

@dataclass
class Edge:
    """Represents an edge in the call graph."""
    source: str
    target: str
    type: str  # calls, imports, inherits, implements
    line_number: int
    file: str

@dataclass
class CallGraph:
    """Represents a complete call graph with metadata."""
    graph: nx.DiGraph
    nodes: Dict[str, Node]
    files: Set[str]
    languages: Set[str]

class CallGraphAnalyzer:
    """Analyzes function calls and dependencies."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, Node] = {}
        self.node_counter = 0

    def build_graph(self, functions: List, calls: List, classes: List,
                   imports: List, file_path: str, language: str) -> CallGraph:
        """Build a call graph from parsed code elements."""
        # Clear existing graph
        self.graph.clear()
        self.nodes.clear()
        self.node_counter = 0

        # Add function nodes
        for func in functions:
            node_id = self._create_node_id(file_path, func.name, "function")
            node = Node(
                id=node_id,
                name=func.name,
                type="function",
                file=file_path,
                language=language,
                start_line=func.start_line,
                end_line=func.end_line
            )
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **node.__dict__)

        # Add class nodes
        for cls in classes:
            node_id = self._create_node_id(file_path, cls.name, "class")
            node = Node(
                id=node_id,
                name=cls.name,
                type="class",
                file=file_path,
                language=language,
                start_line=cls.start_line,
                end_line=cls.end_line
            )
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **node.__dict__)

        # Add call edges
        for call in calls:
            # Find source function (caller)
            source_node = self._find_containing_function(call.start_line, functions, file_path)

            # Find target function (callee)
            target_node = self._find_function_by_name(call.name, functions, file_path)

            if source_node and target_node:
                source_id = self._create_node_id(file_path, source_node.name, "function")
                target_id = self._create_node_id(file_path, target_node.name, "function")

                edge = Edge(
                    source=source_id,
                    target=target_id,
                    type="calls",
                    line_number=call.start_line,
                    file=file_path
                )

                self.graph.add_edge(source_id, target_id, **edge.__dict__)

        # Add import edges
        for imp in imports:
            # Create module node for imports
            module_id = self._create_node_id("", imp.module, "module")
            if module_id not in self.nodes:
                module_node = Node(
                    id=module_id,
                    name=imp.module,
                    type="module",
                    file="",
                    language="",
                    start_line=imp.start_line,
                    end_line=imp.start_line
                )
                self.nodes[module_id] = module_node
                self.graph.add_node(module_id, **module_node.__dict__)

            # Connect file to imported module
            file_id = self._create_node_id(file_path, "", "file")
            if file_id not in self.nodes:
                file_node = Node(
                    id=file_id,
                    name=file_path,
                    type="file",
                    file=file_path,
                    language=language,
                    start_line=0,
                    end_line=0
                )
                self.nodes[file_id] = file_node
                self.graph.add_node(file_id, **file_node.__dict__)

            edge = Edge(
                source=file_id,
                target=module_id,
                type="imports",
                line_number=imp.start_line,
                file=file_path
            )
            self.graph.add_edge(file_id, module_id, **edge.__dict__)

        return CallGraph(
            graph=self.graph,
            nodes=self.nodes,
            files={file_path},
            languages={language}
        )

    def find_impacted_nodes(self, changed_nodes: Set[str]) -> Set[str]:
        """Find nodes impacted by changes using graph traversal."""
        impacted = set(changed_nodes)  # Start with changed nodes

        # For each changed node, find all predecessors (callers) and successors (callees)
        for node in changed_nodes:
            if node in self.graph:
                # Find all nodes that call this node (backward edges)
                predecessors = set(nx.ancestors(self.graph, node))
                predecessors.add(node)

                # Find all nodes called by this node (forward edges)
                successors = set(nx.descendants(self.graph, node))
                successors.add(node)

                impacted.update(predecessors)
                impacted.update(successors)

        return impacted

    def get_call_hierarchy(self, function_id: str) -> Dict:
        """Get the call hierarchy for a given function."""
        if function_id not in self.graph:
            return {"error": f"Function {function_id} not found in call graph"}

        # Get callers (functions that call this function)
        callers = list(self.graph.predecessors(function_id))

        # Get callees (functions called by this function)
        callees = list(self.graph.successors(function_id))

        return {
            "function": function_id,
            "callers": callers,
            "callees": callees,
            "call_count": len(callers) + len(callees)
        }

    def merge_call_graphs(self, graphs: List[CallGraph]) -> CallGraph:
        """Merge multiple call graphs into one."""
        if not graphs:
            return CallGraph(nx.DiGraph(), {}, set(), set())

        merged_graph = graphs[0].graph.copy()
        merged_nodes = graphs[0].nodes.copy()
        all_files = graphs[0].files.copy()
        all_languages = graphs[0].languages.copy()

        for cg in graphs[1:]:
            # Add nodes from current graph
            merged_graph.add_nodes_from(cg.graph.nodes(data=True))
            merged_graph.add_edges_from(cg.graph.edges(data=True))

            # Merge metadata
            merged_nodes.update(cg.nodes)
            all_files.update(cg.files)
            all_languages.update(cg.languages)

        return CallGraph(
            graph=merged_graph,
            nodes=merged_nodes,
            files=all_files,
            languages=all_languages
        )

    def _create_node_id(self, file_path: str, name: str, node_type: str) -> str:
        """Create a unique node ID."""
        if node_type == "file":
            return f"file:{file_path}"
        elif node_type == "module":
            return f"module:{name}"
        else:
            return f"{file_path}:{name}:{node_type}"

    def _find_containing_function(self, line_number: int, functions: List,
                                 file_path: str) -> Optional:
        """Find the function that contains the given line number."""
        for func in functions:
            if func.start_line <= line_number <= func.end_line:
                return func
        return None

    def _find_function_by_name(self, name: str, functions: List,
                              file_path: str) -> Optional:
        """Find a function by name in the current file."""
        for func in functions:
            if func.name == name:
                return func
        return None

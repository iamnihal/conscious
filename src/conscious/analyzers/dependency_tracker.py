"""Dependency Tracker - Tracks indirect and transitive dependencies across the codebase."""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict, deque
import networkx as nx

from .import_graph_builder import ImportGraph, ImportGraphBuilder
from .usage_analyzer import UsageAnalysis, UsageAnalyzer


@dataclass
class DependencyChain:
    """Represents a chain of dependencies."""
    start_node: str
    end_node: str
    path: List[str]
    length: int
    dependency_type: str  # 'import', 'usage', 'combined'


@dataclass
class DependencyMetrics:
    """Metrics for dependency analysis."""
    total_files: int
    direct_dependencies: int
    indirect_dependencies: int
    longest_chain: int
    average_chain_length: float
    circular_dependencies: int
    most_dependent_file: Optional[str] = None
    most_depended_on_file: Optional[str] = None


@dataclass
class ImpactScope:
    """Scope of impact for a change."""
    direct_impacts: List[str]
    indirect_impacts: List[str]
    total_impacts: int
    impact_depth: int
    affected_files: Set[str]
    critical_paths: List[DependencyChain]


class DependencyTracker:
    """Tracks and analyzes transitive dependencies across the codebase."""

    def __init__(self):
        self.import_graph_builder = ImportGraphBuilder()
        self.usage_analyzer = UsageAnalyzer()

    def analyze_dependencies(self, file_paths: List[str], root_directory: str = "",
                           include_usage_deps: bool = True) -> Dict[str, Any]:
        """Perform comprehensive dependency analysis."""

        # Build import graph
        import_graph = self.import_graph_builder.build_graph(file_paths, root_directory)

        # Build usage analysis if requested
        usage_analysis = None
        if include_usage_deps:
            usage_analysis = self.usage_analyzer.analyze_usages(file_paths)

        # Build combined dependency graph
        combined_graph = self._build_combined_dependency_graph(import_graph, usage_analysis)

        # Analyze dependency chains
        dependency_chains = self._find_dependency_chains(combined_graph)

        # Calculate metrics
        metrics = self._calculate_dependency_metrics(combined_graph, dependency_chains)

        # Find circular dependencies
        circular_deps = self._detect_circular_dependencies(combined_graph)

        return {
            'import_graph': import_graph,
            'usage_analysis': usage_analysis,
            'combined_graph': combined_graph,
            'dependency_chains': dependency_chains,
            'metrics': metrics,
            'circular_dependencies': circular_deps
        }

    def _build_combined_dependency_graph(self, import_graph: ImportGraph,
                                       usage_analysis: Optional[UsageAnalysis]) -> nx.DiGraph:
        """Build a combined graph from import and usage dependencies."""
        graph = nx.DiGraph()

        # Add import dependencies
        for edge in import_graph.edges:
            graph.add_edge(edge.from_file, edge.to_file,
                          type='import', weight=1.0, confidence=edge.confidence)

        # Add usage-based dependencies if available
        if usage_analysis:
            for symbol_name, usage_result in usage_analysis.results.items():
                # Create dependencies based on usage locations
                for usage in usage_result.usages:
                    if usage.file_path != usage_result.definition_file:
                        graph.add_edge(usage.file_path, usage_result.definition_file,
                                     type='usage', weight=0.8, confidence=usage.confidence)

        return graph

    def _find_dependency_chains(self, graph: nx.DiGraph) -> List[DependencyChain]:
        """Find all dependency chains in the graph."""
        chains = []

        # Find all simple paths between nodes
        for start_node in graph.nodes():
            for end_node in graph.nodes():
                if start_node != end_node:
                    try:
                        # Find all simple paths (avoid cycles)
                        paths = list(nx.all_simple_paths(graph, start_node, end_node, cutoff=10))
                        for path in paths:
                            if len(path) > 2:  # Only chains with intermediate nodes
                                chain = DependencyChain(
                                    start_node=start_node,
                                    end_node=end_node,
                                    path=path,
                                    length=len(path) - 1,
                                    dependency_type='combined'
                                )
                                chains.append(chain)
                    except nx.NetworkXError:
                        # No path exists
                        continue

        return chains

    def _calculate_dependency_metrics(self, graph: nx.DiGraph,
                                    chains: List[DependencyChain]) -> DependencyMetrics:
        """Calculate comprehensive dependency metrics."""
        total_files = len(graph.nodes())

        # Count direct vs indirect dependencies
        direct_deps = 0
        indirect_deps = 0

        for node in graph.nodes():
            successors = list(graph.successors(node))
            direct_deps += len(successors)

            # Count indirect dependencies (transitive closure)
            reachable = set(nx.descendants(graph, node))
            indirect_deps += len(reachable - set(successors))

        # Chain statistics
        chain_lengths = [chain.length for chain in chains]
        longest_chain = max(chain_lengths) if chain_lengths else 0
        average_chain_length = sum(chain_lengths) / len(chain_lengths) if chain_lengths else 0

        # Find most dependent files
        in_degrees = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())

        most_depended_on = max(in_degrees.items(), key=lambda x: x[1]) if in_degrees else (None, 0)
        most_dependent = max(out_degrees.items(), key=lambda x: x[1]) if out_degrees else (None, 0)

        return DependencyMetrics(
            total_files=total_files,
            direct_dependencies=direct_deps,
            indirect_dependencies=indirect_deps,
            longest_chain=longest_chain,
            average_chain_length=average_chain_length,
            circular_dependencies=0,  # Will be set by circular detection
            most_depended_on_file=most_depended_on[0],
            most_dependent_file=most_dependent[0]
        )

    def _detect_circular_dependencies(self, graph: nx.DiGraph) -> List[List[str]]:
        """Detect circular dependencies in the graph."""
        cycles = []

        try:
            # Find all simple cycles
            cycles = list(nx.simple_cycles(graph))
        except nx.NetworkXError:
            # Graph might have issues
            pass

        return cycles

    def calculate_impact_scope(self, changed_files: List[str],
                             analysis_result: Dict[str, Any]) -> ImpactScope:
        """Calculate the full impact scope of changes to given files."""
        combined_graph = analysis_result['combined_graph']

        direct_impacts = set()
        indirect_impacts = set()
        affected_files = set(changed_files)
        visited = set()

        # Use BFS to find all impacted files
        queue = deque(changed_files)
        depth = 0
        max_depth = 0

        while queue and depth < 10:  # Limit depth to prevent infinite loops
            level_size = len(queue)
            depth += 1

            for _ in range(level_size):
                current_file = queue.popleft()

                if current_file in visited:
                    continue

                visited.add(current_file)

                # Find files that depend on this file (reverse dependencies)
                dependents = []
                for node in combined_graph.nodes():
                    if combined_graph.has_edge(node, current_file):
                        dependents.append(node)

                for dependent in dependents:
                    if dependent not in affected_files:
                        if depth == 1:
                            direct_impacts.add(dependent)
                        else:
                            indirect_impacts.add(dependent)

                        affected_files.add(dependent)
                        queue.append(dependent)

            max_depth = depth

        # Find critical paths
        critical_paths = self._find_critical_paths(changed_files, affected_files,
                                                 analysis_result['dependency_chains'])

        return ImpactScope(
            direct_impacts=list(direct_impacts),
            indirect_impacts=list(indirect_impacts),
            total_impacts=len(affected_files) - len(changed_files),
            impact_depth=max_depth,
            affected_files=affected_files,
            critical_paths=critical_paths
        )

    def _find_critical_paths(self, changed_files: List[str], affected_files: Set[str],
                           dependency_chains: List[DependencyChain]) -> List[DependencyChain]:
        """Find critical dependency paths from changed files to affected files."""
        critical_paths = []

        for chain in dependency_chains:
            # Check if chain starts with a changed file and ends with an affected file
            if (chain.start_node in changed_files and
                chain.end_node in affected_files and
                chain.end_node != chain.start_node):

                # Check if all intermediate files are also affected
                intermediate_affected = all(node in affected_files for node in chain.path[1:-1])
                if intermediate_affected:
                    critical_paths.append(chain)

        # Sort by length (longest first)
        critical_paths.sort(key=lambda x: x.length, reverse=True)

        return critical_paths[:10]  # Return top 10 critical paths

    def find_dependency_hotspots(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Identify files that are dependency hotspots (highly depended upon)."""
        combined_graph = analysis_result['combined_graph']

        # Calculate centrality measures
        try:
            betweenness = nx.betweenness_centrality(combined_graph)
            in_degree = dict(combined_graph.in_degree())
            out_degree = dict(combined_graph.out_degree())

            # Find top hotspots
            top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
            top_in_degree = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]
            top_out_degree = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]

            return {
                'betweenness_centrality': dict(top_betweenness),
                'most_depended_upon': dict(top_in_degree),
                'most_dependent': dict(top_out_degree),
                'high_risk_files': [file for file, centrality in top_betweenness
                                  if centrality > 0.1][:5]  # Files with high centrality
            }
        except:
            # Return basic degree analysis if centrality calculation fails
            in_degree = dict(combined_graph.in_degree())
            out_degree = dict(combined_graph.out_degree())

            return {
                'most_depended_upon': dict(sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]),
                'most_dependent': dict(sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]),
                'high_risk_files': []
            }

    def get_dependency_report(self, analysis_result: Dict[str, Any]) -> str:
        """Generate a comprehensive dependency report."""
        metrics = analysis_result['metrics']
        circular_deps = analysis_result['circular_dependencies']

        report = f"""
Dependency Analysis Report
{'='*50}

📊 Overall Metrics:
   Total Files: {metrics.total_files}
   Direct Dependencies: {metrics.direct_dependencies}
   Indirect Dependencies: {metrics.indirect_dependencies}
   Longest Dependency Chain: {metrics.longest_chain}
   Average Chain Length: {metrics.average_chain_length:.1f}
   Circular Dependencies: {len(circular_deps)}

📁 Key Files:
   Most Depended Upon: {metrics.most_depended_on_file or 'N/A'}
   Most Dependent: {metrics.most_dependent_file or 'N/A'}

🔄 Circular Dependencies:
"""

        if circular_deps:
            for i, cycle in enumerate(circular_deps[:5]):  # Show first 5
                report += f"   {i+1}. {' → '.join(cycle)}\n"
        else:
            report += "   None detected\n"

        # Add hotspot analysis
        hotspots = self.find_dependency_hotspots(analysis_result)
        if hotspots['most_depended_upon']:
            report += f"\n🎯 Dependency Hotspots:\n"
            for file, count in list(hotspots['most_depended_upon'].items())[:5]:
                report += f"   {file}: {count} incoming dependencies\n"

        return report

    def get_file_dependencies(self, file_path: str, analysis_result: Dict[str, Any],
                            depth: int = 3) -> Dict[str, List[str]]:
        """Get detailed dependencies for a specific file."""
        combined_graph = analysis_result['combined_graph']

        if file_path not in combined_graph.nodes():
            return {'direct': [], 'indirect': [], 'all': []}

        # Get direct dependencies
        direct = list(combined_graph.successors(file_path))

        # Get indirect dependencies using BFS
        indirect = []
        visited = set(direct + [file_path])
        queue = deque(direct)

        current_depth = 0
        while queue and current_depth < depth:
            level_size = len(queue)
            current_depth += 1

            for _ in range(level_size):
                current = queue.popleft()
                for successor in combined_graph.successors(current):
                    if successor not in visited:
                        visited.add(successor)
                        indirect.append(successor)
                        queue.append(successor)

        return {
            'direct': direct,
            'indirect': indirect,
            'all': direct + indirect
        }

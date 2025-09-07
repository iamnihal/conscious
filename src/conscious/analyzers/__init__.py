"""Code analysis and impact detection."""

from .call_graph_analyzer import CallGraphAnalyzer, Node, Edge, CallGraph
from .semantic_diff_parser import SemanticDiffParser, SemanticChange, ChangeType
from .change_element_mapper import ChangeElementMapper, MappedElement, ElementMapping
from .change_classifier import (
    ChangeClassifier, ChangeClassification, ClassificationSummary,
    ChangeCategory, ImpactSeverity, BreakingChange
)
from .import_graph_builder import (
    ImportGraphBuilder, ImportGraph, ImportNode, ImportEdge
)
from .usage_analyzer import (
    UsageAnalyzer, UsageAnalysis, UsageResult, UsageLocation
)
from .dependency_tracker import (
    DependencyTracker, DependencyChain, DependencyMetrics, ImpactScope
)
# REMOVED: ImpactPropagator imports - Pure heuristic analysis removed
from .advanced_impact_analyzer import (
    AdvancedImpactAnalyzer, AnalysisConfiguration, AnalysisProgress,
    ComprehensiveAnalysisResult
)

__all__ = [
    "CallGraphAnalyzer", "Node", "Edge", "CallGraph",
    "SemanticDiffParser", "SemanticChange", "ChangeType",
    "ChangeElementMapper", "MappedElement", "ElementMapping",
    "ChangeClassifier", "ChangeClassification", "ClassificationSummary",
    "ChangeCategory", "ImpactSeverity", "BreakingChange",
    "ImportGraphBuilder", "ImportGraph", "ImportNode", "ImportEdge",
    "UsageAnalyzer", "UsageAnalysis", "UsageResult", "UsageLocation",
    "DependencyTracker", "DependencyChain", "DependencyMetrics", "ImpactScope",
# REMOVED: ImpactPropagator exports - Pure heuristic analysis removed
    "AdvancedImpactAnalyzer", "AnalysisConfiguration", "AnalysisProgress",
    "ComprehensiveAnalysisResult"
]

"""Advanced Impact Analyzer - Orchestrates comprehensive impact analysis across the entire codebase."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Any, Union
import time
import os
from pathlib import Path

from .semantic_diff_parser import SemanticDiffParser
from .change_element_mapper import ChangeElementMapper
from .change_classifier import ChangeClassifier
from .import_graph_builder import ImportGraphBuilder
from .usage_analyzer import UsageAnalyzer
from .dependency_tracker import DependencyTracker, ImpactScope
# REMOVED: ImpactPropagator - Pure heuristic logic removed


@dataclass
class AnalysisConfiguration:
    """Configuration for impact analysis."""
    include_usage_analysis: bool = True
    include_dependency_tracking: bool = True
    test_coverage_data: Optional[Dict[str, float]] = None
    analysis_depth: int = 3  # How deep to analyze dependencies
    confidence_threshold: float = 0.4  # Minimum confidence for recommendations
    enable_caching: bool = True
    cache_directory: Optional[str] = None


@dataclass
class AnalysisProgress:
    """Tracks progress through analysis phases."""
    current_phase: str
    completed_phases: List[str]
    total_phases: int
    start_time: float
    phase_start_time: float

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def phase_elapsed_time(self) -> float:
        return time.time() - self.phase_start_time


@dataclass
class ComprehensiveAnalysisResult:
    """Complete result from advanced impact analysis."""
    # Core analysis results
    semantic_changes: List[Any]
    change_classifications: List[Any]
    element_mappings: List[Any]
    configuration: AnalysisConfiguration
    progress: AnalysisProgress
    analysis_summary: Dict[str, Any]
    performance_metrics: Dict[str, float]

    # Graph and dependency analysis
    import_graph: Optional[Any] = None
    usage_analysis: Optional[Any] = None
    dependency_analysis: Optional[Dict[str, Any]] = None

    # Impact analysis
    impact_scope: Optional[ImpactScope] = None
# REMOVED: impact_propagation - Pure heuristic analysis removed


class AdvancedImpactAnalyzer:
    """Main orchestrator for comprehensive impact analysis."""

    def __init__(self, configuration: Optional[AnalysisConfiguration] = None):
        self.config = configuration or AnalysisConfiguration()

        # Initialize all analysis components
        self.semantic_parser = SemanticDiffParser()
        self.element_mapper = ChangeElementMapper()
        self.change_classifier = ChangeClassifier()
        self.import_builder = ImportGraphBuilder()
        self.usage_analyzer = UsageAnalyzer()
        self.dependency_tracker = DependencyTracker()
# REMOVED: impact_propagator - Pure heuristic analysis removed

        # Analysis state
        self._progress = None
        self._cache = {}

    def analyze_impact(self, diff_content: str, file_paths: List[str],
                      root_directory: str = "") -> ComprehensiveAnalysisResult:
        """Perform comprehensive impact analysis on code changes."""

        start_time = time.time()
        self._progress = AnalysisProgress(
            current_phase="initialization",
            completed_phases=[],
            total_phases=7,  # Estimate total phases
            start_time=start_time,
            phase_start_time=start_time
        )

        try:
            # Phase 1: Parse semantic changes
            self._update_progress("semantic_parsing")
            # Parse semantic changes for each file in the diff
            semantic_changes = []
            for file_path in file_paths:
                if file_path.endswith('.py'):
                    file_changes = self.semantic_parser.parse_semantic_changes(
                        diff_content, file_path
                    )
                    semantic_changes.extend(file_changes)

            # Phase 2: Map changes to elements
            self._update_progress("element_mapping")
            file_contents = self._load_file_contents(file_paths)
            element_mappings = self.element_mapper.map_changes_to_elements(
                semantic_changes, file_contents
            )

            # Phase 3: Classify changes
            self._update_progress("change_classification")
            change_classifications = self.change_classifier.classify_changes(element_mappings)

            # Phase 4: Build import graph
            self._update_progress("import_graph_building")
            import_graph = self.import_builder.build_graph(file_paths, root_directory)

            # Phase 5: Usage analysis (optional)
            usage_analysis = None
            if self.config.include_usage_analysis:
                self._update_progress("usage_analysis")
                usage_analysis = self.usage_analyzer.analyze_usages(file_paths)

            # Phase 6: Dependency tracking (optional)
            dependency_analysis = None
            if self.config.include_dependency_tracking:
                self._update_progress("dependency_tracking")
                dependency_analysis = self.dependency_tracker.analyze_dependencies(
                    file_paths, root_directory, self.config.include_usage_analysis
                )

            # Phase 7: Impact propagation
            self._update_progress("impact_propagation")
            impact_scope = None
            if dependency_analysis and 'combined_graph' in dependency_analysis:
                # Identify changed files from semantic changes
                changed_files = self._extract_changed_files(semantic_changes, file_paths)
                impact_scope = self.dependency_tracker.calculate_impact_scope(
                    changed_files, dependency_analysis
                )

# REMOVED: Impact propagation - Pure heuristic analysis removed

            # Generate final result
            self._update_progress("finalizing")
            result = ComprehensiveAnalysisResult(
                semantic_changes=semantic_changes,
                change_classifications=change_classifications,
                element_mappings=element_mappings,
                import_graph=import_graph,
                usage_analysis=usage_analysis,
                dependency_analysis=dependency_analysis,
                impact_scope=impact_scope,
# REMOVED: impact_propagation parameter - Pure heuristic analysis removed
                configuration=self.config,
                progress=self._progress,
                analysis_summary=self._generate_analysis_summary(
                    semantic_changes, change_classifications
                ),
                performance_metrics=self._calculate_performance_metrics(start_time)
            )

            return result

        except Exception as e:
            # Handle errors gracefully
            self._update_progress("error")
            return self._create_error_result(str(e), start_time)

    def _update_progress(self, phase: str):
        """Update analysis progress."""
        if self._progress:
            if self._progress.current_phase != "error":
                self._progress.completed_phases.append(self._progress.current_phase)
            self._progress.current_phase = phase
            self._progress.phase_start_time = time.time()

    def _load_file_contents(self, file_paths: List[str]) -> Dict[str, str]:
        """Load contents of files for analysis."""
        contents = {}
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        contents[file_path] = f.read()
                except (UnicodeDecodeError, IOError):
                    contents[file_path] = ""
        return contents

    def _extract_changed_files(self, semantic_changes: List[Any],
                             file_paths: List[str]) -> List[str]:
        """Extract list of files that have changes."""
        changed_files = set()

        # This is a simplified implementation - in practice, you'd
        # extract file paths from the semantic changes
        for change in semantic_changes:
            if hasattr(change, 'file_path') and change.file_path:
                changed_files.add(change.file_path)

        return list(changed_files)

    def _generate_analysis_summary(self, semantic_changes: List[Any],
                                 change_classifications: List[Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis summary."""

        summary = {
            'total_semantic_changes': len(semantic_changes),
            'total_change_classifications': len(change_classifications),
            'change_types_breakdown': {},
            'severity_breakdown': {},
            'breaking_changes': 0,
            'non_breaking_changes': 0
        }

        # Analyze change classifications
        for classification in change_classifications:
            # Count change types
            change_type = classification.category.value
            summary['change_types_breakdown'][change_type] = \
                summary['change_types_breakdown'].get(change_type, 0) + 1

# REMOVED: Severity and breaking change classification - Pure heuristic logic removed

# REMOVED: Impact propagation summary - Pure heuristic analysis removed

        return summary

    def _calculate_performance_metrics(self, start_time: float) -> Dict[str, float]:
        """Calculate performance metrics for the analysis."""
        end_time = time.time()
        total_time = end_time - start_time

        return {
            'total_analysis_time': total_time,
            'average_time_per_change': total_time / max(1, len(self._progress.completed_phases)),
            'phases_completed': len(self._progress.completed_phases)
        }

    def _create_error_result(self, error_message: str,
                           start_time: float) -> ComprehensiveAnalysisResult:
        """Create a result object for error cases."""
        return ComprehensiveAnalysisResult(
            semantic_changes=[],
            change_classifications=[],
            element_mappings=[],
            configuration=self.config,
            progress=self._progress,
            analysis_summary={'error': error_message},
            performance_metrics=self._calculate_performance_metrics(start_time)
        )

    def generate_comprehensive_report(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate a comprehensive human-readable report."""
        report_lines = []

        # Header
        report_lines.append("=" * 80)
        report_lines.append("ADVANCED IMPACT ANALYSIS REPORT")
        report_lines.append("=" * 80)

        # Executive Summary
        report_lines.append("\n📊 EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)

        summary = result.analysis_summary
        report_lines.append(f"Total Changes Analyzed: {summary.get('total_semantic_changes', 0)}")
        report_lines.append(f"Breaking Changes: {summary.get('breaking_changes', 0)}")
        report_lines.append(f"Non-Breaking Changes: {summary.get('non_breaking_changes', 0)}")

        if 'overall_severity' in summary:
            report_lines.append(f"Overall Severity: {summary['overall_severity'].upper()}")
            report_lines.append(f"Overall Confidence: {summary['overall_confidence'].upper()}")

        # Performance Metrics
        report_lines.append("\n⚡ PERFORMANCE METRICS")
        report_lines.append("-" * 40)
        perf = result.performance_metrics
        report_lines.append(".2f")
        report_lines.append(f"Phases Completed: {perf.get('phases_completed', 0)}")

        # Change Analysis
        report_lines.append("\n🔍 CHANGE ANALYSIS")
        report_lines.append("-" * 40)

        if result.change_classifications:
            report_lines.append(f"Changes by Type: {summary.get('change_types_breakdown', {})}")
            report_lines.append(f"Changes by Severity: {summary.get('severity_breakdown', {})}")

        # Impact Analysis
        if result.impact_propagation:
            report_lines.append("\n🎯 IMPACT ANALYSIS")
            report_lines.append("-" * 40)

            propagation = result.impact_propagation
            report_lines.append(f"Critical Changes: {propagation.critical_changes}")
            report_lines.append(f"Major Changes: {propagation.major_changes}")
            report_lines.append(f"Minor Changes: {propagation.minor_changes}")
            report_lines.append(f"Patch Changes: {propagation.patch_changes}")

            if propagation.risk_assessment:
                report_lines.append(f"\nRisk Assessment: {propagation.risk_assessment}")

            # Recommendations
            if propagation.deployment_recommendations:
                report_lines.append("\n🚀 DEPLOYMENT RECOMMENDATIONS")
                report_lines.append("-" * 40)
                for rec in propagation.deployment_recommendations:
                    report_lines.append(f"• {rec}")

            if propagation.coordination_needed:
                report_lines.append("\n👥 COORDINATION REQUIRED")
                report_lines.append("-" * 40)
                for coord in propagation.coordination_needed:
                    report_lines.append(f"• {coord}")

        # Detailed Changes
        if result.change_classifications:
            report_lines.append("\n📝 DETAILED CHANGE ANALYSIS")
            report_lines.append("-" * 40)

            for i, classification in enumerate(result.change_classifications[:10]):  # Show first 10
                report_lines.append(f"\n{i+1}. {classification.category.value.upper()}")
                report_lines.append(f"   Severity: {classification.severity.value.upper()}")
                report_lines.append(f"   Breaking: {classification.breaking_change.value.upper()}")
                if classification.suggested_actions:
                    report_lines.append(f"   Actions: {classification.suggested_actions[0]}")

        # Footer
        report_lines.append("\n" + "=" * 80)
        report_lines.append("Analysis completed successfully")
        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def export_analysis_data(self, result: ComprehensiveAnalysisResult,
                           output_format: str = "json") -> Union[str, Dict]:
        """Export analysis results in various formats."""

        if output_format == "json":
            # Convert to JSON-serializable format
            export_data = {
                'summary': result.analysis_summary,
                'performance': result.performance_metrics,
                'changes': {
                    'total': len(result.semantic_changes),
                    'types': result.analysis_summary.get('change_types_breakdown', {}),
                    'severities': result.analysis_summary.get('severity_breakdown', {})
                }
            }

            if result.impact_propagation:
                export_data['impact'] = {
                    'severity': result.impact_propagation.overall_severity.value,
                    'confidence': result.impact_propagation.overall_confidence.value,
                    'critical_changes': result.impact_propagation.critical_changes,
                    'major_changes': result.impact_propagation.major_changes,
                    'risk_assessment': result.impact_propagation.risk_assessment
                }

            return export_data

        elif output_format == "text":
            return self.generate_comprehensive_report(result)

        else:
            raise ValueError(f"Unsupported export format: {output_format}")

    def get_analysis_status(self) -> Optional[AnalysisProgress]:
        """Get current analysis progress."""
        return self._progress

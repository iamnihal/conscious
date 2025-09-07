"""Impact Propagator - Propagates impacts with intelligent rules and confidence scoring."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Any, Tuple
from enum import Enum

from .change_classifier import ChangeClassification, ChangeCategory, ImpactSeverity, BreakingChange
from .usage_analyzer import UsageAnalysis, UsageResult
from .dependency_tracker import ImpactScope, DependencyChain
from .semantic_diff_parser import SemanticChange, ChangeType


class PropagationSeverity(Enum):
    """Severity levels for impact propagation."""
    CRITICAL = "critical"      # Breaking changes requiring coordination
    MAJOR = "major"           # Significant changes needing attention
    MINOR = "minor"           # Small changes, routine deployment
    PATCH = "patch"           # Safe changes, minimal risk


class ConfidenceLevel(Enum):
    """Confidence levels for impact predictions."""
    HIGH = "high"             # 80-100% confidence
    MEDIUM = "medium"         # 60-79% confidence
    LOW = "low"              # 40-59% confidence
    UNCERTAIN = "uncertain"   # <40% confidence


@dataclass
class PropagationRule:
    """A rule for propagating impact from a change."""
    name: str
    change_category: ChangeCategory
    breaking_change: BreakingChange
    severity: PropagationSeverity
    confidence_multiplier: float
    description: str
    conditions: List[str]
    recommendations: List[str]


@dataclass
class ImpactPrediction:
    """Prediction of impact for a specific change."""
    change_classification: ChangeClassification
    propagated_severity: PropagationSeverity
    confidence_level: ConfidenceLevel
    confidence_score: float
    affected_files: Set[str]
    affected_symbols: Set[str]
    propagation_rules: List[PropagationRule]
    risk_factors: List[str]
    recommendations: List[str]
    mitigation_steps: List[str]


@dataclass
class ImpactAnalysisResult:
    """Complete impact analysis result."""
    predictions: List[ImpactPrediction]
    overall_severity: PropagationSeverity
    overall_confidence: ConfidenceLevel
    total_affected_files: int
    total_affected_symbols: int
    critical_changes: int
    major_changes: int
    minor_changes: int
    patch_changes: int
    risk_assessment: str
    deployment_recommendations: List[str]
    coordination_needed: List[str]


class ImpactPropagationRules:
    """Defines rules for propagating impact from different types of changes."""

    def __init__(self):
        self.rules = self._build_propagation_rules()

    def _build_propagation_rules(self) -> List[PropagationRule]:
        """Build the complete set of propagation rules."""
        return [
            # Function signature changes
            PropagationRule(
                name="breaking_function_signature",
                change_category=ChangeCategory.SIGNATURE_CHANGE,
                breaking_change=BreakingChange.BREAKING,
                severity=PropagationSeverity.CRITICAL,
                confidence_multiplier=0.9,
                description="Breaking change to function signature",
                conditions=["Parameter removal", "Required parameter addition", "Return type change"],
                recommendations=[
                    "Update all function calls",
                    "Coordinate with all dependent teams",
                    "Schedule maintenance window",
                    "Prepare rollback plan"
                ]
            ),

            PropagationRule(
                name="backward_compatible_signature",
                change_category=ChangeCategory.SIGNATURE_CHANGE,
                breaking_change=BreakingChange.NON_BREAKING,
                severity=PropagationSeverity.MINOR,
                confidence_multiplier=0.8,
                description="Backward compatible signature change",
                conditions=["Optional parameter addition", "Parameter renaming with default"],
                recommendations=[
                    "Update API documentation",
                    "Test with both old and new signatures",
                    "Gradual rollout recommended"
                ]
            ),

            # Class definition changes
            PropagationRule(
                name="breaking_class_change",
                change_category=ChangeCategory.STRUCTURE_CHANGE,
                breaking_change=BreakingChange.BREAKING,
                severity=PropagationSeverity.CRITICAL,
                confidence_multiplier=0.95,
                description="Breaking change to class definition",
                conditions=["Method removal", "Property removal", "Interface change"],
                recommendations=[
                    "Update all instantiations and usages",
                    "Coordinate with all dependent teams",
                    "Consider API versioning",
                    "Extensive testing required"
                ]
            ),

            PropagationRule(
                name="class_addition",
                change_category=ChangeCategory.STRUCTURE_CHANGE,
                breaking_change=BreakingChange.NON_BREAKING,
                severity=PropagationSeverity.PATCH,
                confidence_multiplier=0.9,
                description="Safe class addition",
                conditions=["New class", "New method", "New property"],
                recommendations=[
                    "No immediate action required",
                    "Update documentation when convenient"
                ]
            ),

            # Import changes
            PropagationRule(
                name="import_removal",
                change_category=ChangeCategory.IMPORT_CHANGE,
                breaking_change=BreakingChange.BREAKING,
                severity=PropagationSeverity.MAJOR,
                confidence_multiplier=0.85,
                description="Import removal breaks dependencies",
                conditions=["Import statement removed"],
                recommendations=[
                    "Verify import is truly unused",
                    "Check for dynamic imports",
                    "Update dependent files"
                ]
            ),

            PropagationRule(
                name="import_addition",
                change_category=ChangeCategory.IMPORT_CHANGE,
                breaking_change=BreakingChange.NON_BREAKING,
                severity=PropagationSeverity.PATCH,
                confidence_multiplier=0.9,
                description="Safe import addition",
                conditions=["New import added"],
                recommendations=[
                    "Ensure dependency is available",
                    "Check import compatibility"
                ]
            ),

            # Logic changes
            PropagationRule(
                name="significant_logic_change",
                change_category=ChangeCategory.LOGIC_CHANGE,
                breaking_change=BreakingChange.NON_BREAKING,
                severity=PropagationSeverity.MAJOR,
                confidence_multiplier=0.75,
                description="Significant logic modification",
                conditions=["Complex logic changes", "Algorithm modifications"],
                recommendations=[
                    "Review logic changes thoroughly",
                    "Update tests for new behavior",
                    "Monitor for behavioral changes"
                ]
            ),

            PropagationRule(
                name="minor_logic_change",
                change_category=ChangeCategory.LOGIC_CHANGE,
                breaking_change=BreakingChange.NON_BREAKING,
                severity=PropagationSeverity.MINOR,
                confidence_multiplier=0.7,
                description="Minor logic modification",
                conditions=["Small fixes", "Code cleanup"],
                recommendations=[
                    "Standard testing procedures",
                    "Code review for correctness"
                ]
            ),

            # Configuration changes
            PropagationRule(
                name="configuration_removal",
                change_category=ChangeCategory.CONFIGURATION_CHANGE,
                breaking_change=BreakingChange.BREAKING,
                severity=PropagationSeverity.MAJOR,
                confidence_multiplier=0.8,
                description="Configuration removal",
                conditions=["Constant removal", "Setting removal"],
                recommendations=[
                    "Update configuration files",
                    "Notify operations team",
                    "Test configuration changes"
                ]
            )
        ]

    def find_applicable_rules(self, change_classification: ChangeClassification) -> List[PropagationRule]:
        """Find rules that apply to a given change classification."""
        applicable = []

        for rule in self.rules:
            if (rule.change_category == change_classification.category and
                rule.breaking_change == change_classification.breaking_change):
                applicable.append(rule)

        # If no specific rules match, use category-only matching
        if not applicable:
            for rule in self.rules:
                if rule.change_category == change_classification.category:
                    applicable.append(rule)

        return applicable


class ConfidenceScorer:
    """Calculates confidence scores for impact predictions."""

    def __init__(self):
        self.factors = {
            'semantic_parsing_quality': 0.25,
            'usage_analysis_completeness': 0.3,
            'dependency_analysis_depth': 0.15,
            'change_type_clarity': 0.15,
            'historical_patterns': 0.1,
            'test_coverage': 0.05
        }

    def calculate_confidence(self, change_classification: ChangeClassification,
                           usage_result: Optional[UsageResult] = None,
                           dependency_depth: int = 0,
                           test_coverage: float = 0.0) -> Tuple[float, ConfidenceLevel]:
        """Calculate overall confidence score for an impact prediction."""

        confidence_score = 0.0

        # Semantic parsing quality
        semantic_quality = self._assess_semantic_quality(change_classification)
        confidence_score += semantic_quality * self.factors['semantic_parsing_quality']

        # Usage analysis completeness
        usage_completeness = self._assess_usage_completeness(usage_result)
        confidence_score += usage_completeness * self.factors['usage_analysis_completeness']

        # Dependency analysis depth
        dependency_quality = min(dependency_depth / 5.0, 1.0)  # Normalize to 0-1
        confidence_score += dependency_quality * self.factors['dependency_analysis_depth']

        # Change type clarity
        change_clarity = self._assess_change_clarity(change_classification)
        confidence_score += change_clarity * self.factors['change_type_clarity']

        # Test coverage
        if test_coverage is not None:
            confidence_score += test_coverage * self.factors['test_coverage']

        # Historical patterns (simplified)
        historical_factor = 0.7  # Could be improved with ML model
        confidence_score += historical_factor * self.factors['historical_patterns']

        # Convert to confidence level
        if confidence_score >= 0.6:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence_score >= 0.4:
            confidence_level = ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.2:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.UNCERTAIN

        return confidence_score, confidence_level

    def _assess_semantic_quality(self, change_classification: ChangeClassification) -> float:
        """Assess quality of semantic change parsing."""
        # Higher confidence for well-defined changes
        if change_classification.category == ChangeCategory.SIGNATURE_CHANGE:
            return 0.9
        elif change_classification.category == ChangeCategory.STRUCTURE_CHANGE:
            return 0.85
        elif change_classification.category == ChangeCategory.LOGIC_CHANGE:
            return 0.7
        else:
            return 0.6

    def _assess_usage_completeness(self, usage_result: Optional[UsageResult]) -> float:
        """Assess completeness of usage analysis."""
        if not usage_result:
            return 0.3

        # Higher confidence with more usage data
        files_coverage = min(usage_result.unique_files / 10.0, 1.0)  # Normalize
        total_usages = min(usage_result.total_usages / 50.0, 1.0)   # Normalize

        return (files_coverage + total_usages) / 2.0

    def _assess_change_clarity(self, change_classification: ChangeClassification) -> float:
        """Assess clarity of the change type."""
        # Clear breaking changes have higher confidence
        if change_classification.breaking_change == BreakingChange.BREAKING:
            return 0.9
        elif change_classification.breaking_change == BreakingChange.NON_BREAKING:
            return 0.8
        else:
            return 0.5


class ActionRecommender:
    """Provides actionable recommendations based on impact analysis."""

    def __init__(self):
        self.recommendation_templates = self._build_recommendation_templates()

    def _build_recommendation_templates(self) -> Dict[str, List[str]]:
        """Build templates for different types of recommendations."""
        return {
            'deployment': [
                "Schedule deployment during low-traffic hours",
                "Prepare rollback plan and test it",
                "Have on-call engineer available during deployment",
                "Monitor error rates and performance metrics closely",
                "Consider canary deployment for high-risk changes"
            ],

            'coordination': [
                "Notify all teams that depend on this component",
                "Schedule coordination meeting with stakeholders",
                "Update API documentation and versioning",
                "Communicate breaking changes to external consumers",
                "Consider feature flags for gradual rollout"
            ],

            'testing': [
                "Run full test suite before deployment",
                "Execute integration tests for affected components",
                "Perform manual testing of critical user flows",
                "Validate with production-like data",
                "Test rollback procedures thoroughly"
            ],

            'monitoring': [
                "Set up additional monitoring for affected services",
                "Monitor error rates and latency metrics",
                "Watch for unexpected behavioral changes",
                "Have alerting rules for potential issues",
                "Prepare incident response procedures"
            ]
        }

    def generate_recommendations(self, predictions: List[ImpactPrediction],
                               impact_scope: ImpactScope) -> Dict[str, List[str]]:
        """Generate comprehensive recommendations based on impact analysis."""

        recommendations = {
            'immediate_actions': [],
            'coordination_needed': [],
            'testing_requirements': [],
            'deployment_strategy': [],
            'monitoring_setup': [],
            'risk_mitigation': []
        }

        # Analyze severity distribution
        severity_counts = {}
        for prediction in predictions:
            severity_counts[prediction.propagated_severity] = \
                severity_counts.get(prediction.propagated_severity, 0) + 1

        # Generate recommendations based on severity
        if severity_counts.get(PropagationSeverity.CRITICAL, 0) > 0:
            recommendations['immediate_actions'].extend([
                "STOP deployment and reassess",
                "Schedule emergency coordination meeting",
                "Prepare detailed impact analysis report"
            ])
            recommendations['coordination_needed'].extend(
                self.recommendation_templates['coordination']
            )
            recommendations['deployment_strategy'].append(
                "High-risk changes require special approval process"
            )

        elif severity_counts.get(PropagationSeverity.MAJOR, 0) > 0:
            recommendations['coordination_needed'].extend([
                "Notify dependent teams",
                "Schedule brief coordination call"
            ])
            recommendations['testing_requirements'].extend(
                self.recommendation_templates['testing']
            )

        # Impact scope based recommendations
        if impact_scope.total_impacts > 10:
            recommendations['deployment_strategy'].append(
                "Large impact scope - consider phased deployment"
            )
            recommendations['monitoring_setup'].extend(
                self.recommendation_templates['monitoring']
            )

        if impact_scope.impact_depth > 3:
            recommendations['risk_mitigation'].append(
                "Deep impact propagation - prepare detailed rollback plan"
            )

        # Confidence-based recommendations
        low_confidence_predictions = [
            p for p in predictions
            if p.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.UNCERTAIN]
        ]

        if low_confidence_predictions:
            recommendations['testing_requirements'].append(
                "Low confidence predictions - extensive testing required"
            )
            recommendations['deployment_strategy'].append(
                "Consider feature flags for uncertain impacts"
            )

        return recommendations


class ImpactPropagator:
    """Main orchestrator for impact propagation analysis."""

    def __init__(self):
        self.rules_engine = ImpactPropagationRules()
        self.confidence_scorer = ConfidenceScorer()
        self.action_recommender = ActionRecommender()

    def propagate_impacts(self, change_classifications: List[ChangeClassification],
                         usage_analysis: Optional[UsageAnalysis] = None,
                         impact_scope: Optional[ImpactScope] = None,
                         test_coverage: float = 0.0) -> ImpactAnalysisResult:
        """Propagate impacts for a set of change classifications."""

        predictions = []

        for change_classification in change_classifications:
            prediction = self._analyze_single_change(
                change_classification, usage_analysis, test_coverage
            )
            predictions.append(prediction)

        # Calculate overall assessment
        overall_severity = self._calculate_overall_severity(predictions)
        overall_confidence = self._calculate_overall_confidence(predictions)

        # Aggregate statistics
        total_affected_files = len(set().union(*[p.affected_files for p in predictions]))
        total_affected_symbols = len(set().union(*[p.affected_symbols for p in predictions]))

        severity_counts = self._count_severities(predictions)

        # Generate risk assessment
        risk_assessment = self._generate_risk_assessment(
            predictions, impact_scope, overall_severity, overall_confidence
        )

        # Generate recommendations
        if impact_scope:
            recommendations = self.action_recommender.generate_recommendations(
                predictions, impact_scope
            )
        else:
            recommendations = {'general': ['Conduct full impact analysis with dependency tracking']}

        return ImpactAnalysisResult(
            predictions=predictions,
            overall_severity=overall_severity,
            overall_confidence=overall_confidence,
            total_affected_files=total_affected_files,
            total_affected_symbols=total_affected_symbols,
            critical_changes=severity_counts.get(PropagationSeverity.CRITICAL, 0),
            major_changes=severity_counts.get(PropagationSeverity.MAJOR, 0),
            minor_changes=severity_counts.get(PropagationSeverity.MINOR, 0),
            patch_changes=severity_counts.get(PropagationSeverity.PATCH, 0),
            risk_assessment=risk_assessment,
            deployment_recommendations=recommendations.get('deployment_strategy', []),
            coordination_needed=recommendations.get('coordination_needed', [])
        )

    def _analyze_single_change(self, change_classification: ChangeClassification,
                             usage_analysis: Optional[UsageAnalysis],
                             test_coverage: float) -> ImpactPrediction:
        """Analyze impact for a single change."""

        # Find applicable propagation rules
        applicable_rules = self.rules_engine.find_applicable_rules(change_classification)

        # Determine propagated severity
        propagated_severity = self._determine_propagated_severity(
            change_classification, applicable_rules
        )

        # Calculate confidence
        usage_result = None
        if usage_analysis and change_classification.affected_elements:
            # Try to find usage data for any of the affected elements
            for element in change_classification.affected_elements:
                if element in usage_analysis.results:
                    usage_result = usage_analysis.results[element]
                    break

        confidence_score, confidence_level = self.confidence_scorer.calculate_confidence(
            change_classification, usage_result, 0, test_coverage
        )

        # Determine affected files and symbols
        affected_files, affected_symbols = self._determine_affected_entities(
            change_classification, usage_result
        )

        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            change_classification, propagated_severity, confidence_level
        )

        # Generate recommendations
        recommendations = []
        mitigation_steps = []

        for rule in applicable_rules:
            recommendations.extend(rule.recommendations)

        return ImpactPrediction(
            change_classification=change_classification,
            propagated_severity=propagated_severity,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            affected_files=affected_files,
            affected_symbols=affected_symbols,
            propagation_rules=applicable_rules,
            risk_factors=risk_factors,
            recommendations=list(set(recommendations)),  # Remove duplicates
            mitigation_steps=mitigation_steps
        )

    def _determine_propagated_severity(self, change_classification: ChangeClassification,
                                     applicable_rules: List[PropagationRule]) -> PropagationSeverity:
        """Determine the propagated severity based on rules."""
        if not applicable_rules:
            # Default mapping from basic severity
            severity_map = {
                ImpactSeverity.HIGH: PropagationSeverity.MAJOR,
                ImpactSeverity.MEDIUM: PropagationSeverity.MINOR,
                ImpactSeverity.LOW: PropagationSeverity.PATCH
            }
            return severity_map.get(change_classification.severity, PropagationSeverity.MINOR)

        # Use the highest severity rule
        severities = [rule.severity for rule in applicable_rules]
        severity_order = [PropagationSeverity.PATCH, PropagationSeverity.MINOR,
                         PropagationSeverity.MAJOR, PropagationSeverity.CRITICAL]

        for severity in reversed(severity_order):
            if severity in severities:
                return severity

        return PropagationSeverity.MINOR

    def _determine_affected_entities(self, change_classification: ChangeClassification,
                                   usage_result: Optional[UsageResult]) -> Tuple[Set[str], Set[str]]:
        """Determine affected files and symbols."""
        affected_files = set()
        affected_symbols = set()

        # Add files that use the changed symbol
        if usage_result:
            for usage in usage_result.usages:
                affected_files.add(usage.file_path)
                if usage.confidence > 0.7:  # High confidence usages
                    affected_symbols.add(f"{usage.file_path}:{usage.line_number}")

        # If no usage data, assume at least the affected elements are impacted
        if not affected_files and change_classification.affected_elements:
            # We don't have file info, so we'll work with what we have
            affected_symbols.update(change_classification.affected_elements)

        return affected_files, affected_symbols

    def _identify_risk_factors(self, change_classification: ChangeClassification,
                             severity: PropagationSeverity,
                             confidence: ConfidenceLevel) -> List[str]:
        """Identify risk factors for the change."""
        risk_factors = []

        if severity == PropagationSeverity.CRITICAL:
            risk_factors.append("Critical severity - requires special handling")

        if confidence in [ConfidenceLevel.LOW, ConfidenceLevel.UNCERTAIN]:
            risk_factors.append("Low confidence in impact prediction")

        if change_classification.breaking_change == BreakingChange.BREAKING:
            risk_factors.append("Breaking change detected")

        if change_classification.category == ChangeCategory.SIGNATURE_CHANGE:
            risk_factors.append("API signature change - affects all callers")

        return risk_factors

    def _calculate_overall_severity(self, predictions: List[ImpactPrediction]) -> PropagationSeverity:
        """Calculate overall severity from all predictions."""
        if not predictions:
            return PropagationSeverity.PATCH

        severities = [p.propagated_severity for p in predictions]

        # Return the highest severity
        severity_order = [PropagationSeverity.PATCH, PropagationSeverity.MINOR,
                         PropagationSeverity.MAJOR, PropagationSeverity.CRITICAL]

        for severity in reversed(severity_order):
            if severity in severities:
                return severity

        return PropagationSeverity.PATCH

    def _calculate_overall_confidence(self, predictions: List[ImpactPrediction]) -> ConfidenceLevel:
        """Calculate overall confidence from all predictions."""
        if not predictions:
            return ConfidenceLevel.UNCERTAIN

        confidence_levels = [p.confidence_level for p in predictions]

        # Return the lowest confidence level
        confidence_order = [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM,
                           ConfidenceLevel.LOW, ConfidenceLevel.UNCERTAIN]

        for confidence in confidence_order:
            if confidence in confidence_levels:
                return confidence

        return ConfidenceLevel.UNCERTAIN

    def _count_severities(self, predictions: List[ImpactPrediction]) -> Dict[PropagationSeverity, int]:
        """Count occurrences of each severity level."""
        counts = {}
        for prediction in predictions:
            counts[prediction.propagated_severity] = \
                counts.get(prediction.propagated_severity, 0) + 1
        return counts

    def _generate_risk_assessment(self, predictions: List[ImpactPrediction],
                                impact_scope: Optional[ImpactScope],
                                overall_severity: PropagationSeverity,
                                overall_confidence: ConfidenceLevel) -> str:
        """Generate a comprehensive risk assessment."""

        assessment_parts = []

        # Severity assessment
        if overall_severity == PropagationSeverity.CRITICAL:
            assessment_parts.append("CRITICAL RISK: Breaking changes require emergency coordination")
        elif overall_severity == PropagationSeverity.MAJOR:
            assessment_parts.append("MAJOR RISK: Significant changes need careful coordination")
        elif overall_severity == PropagationSeverity.MINOR:
            assessment_parts.append("MINOR RISK: Small changes, standard deployment process")
        else:
            assessment_parts.append("LOW RISK: Safe changes, routine deployment")

        # Confidence assessment
        if overall_confidence == ConfidenceLevel.HIGH:
            assessment_parts.append("High confidence in impact predictions")
        elif overall_confidence == ConfidenceLevel.MEDIUM:
            assessment_parts.append("Medium confidence - additional testing recommended")
        elif overall_confidence == ConfidenceLevel.LOW:
            assessment_parts.append("Low confidence - extensive testing required")
        else:
            assessment_parts.append("Uncertain impact - full analysis needed")

        # Impact scope assessment
        if impact_scope:
            if impact_scope.total_impacts > 20:
                assessment_parts.append("Large impact scope - phased deployment recommended")
            elif impact_scope.total_impacts > 5:
                assessment_parts.append("Moderate impact scope - coordinated deployment needed")

            if impact_scope.impact_depth > 4:
                assessment_parts.append("Deep impact propagation - detailed rollback plan required")

        return " | ".join(assessment_parts)

"""Change Classifier - Classifies semantic changes by type and impact."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Union, Any
from enum import Enum

from .semantic_diff_parser import SemanticChange, ChangeType
from .change_element_mapper import ElementMapping, MappedElement


class ChangeCategory(Enum):
    """Categories of code changes."""
    SIGNATURE_CHANGE = "signature_change"
    LOGIC_CHANGE = "logic_change"
    IMPORT_CHANGE = "import_change"
    TYPE_CHANGE = "type_change"
    STRUCTURE_CHANGE = "structure_change"
    CONFIGURATION_CHANGE = "configuration_change"


class ImpactSeverity(Enum):
    """Severity levels for change impact."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class BreakingChange(Enum):
    """Whether a change is breaking or non-breaking."""
    BREAKING = "breaking"
    NON_BREAKING = "non_breaking"
    UNKNOWN = "unknown"


@dataclass
class ChangeClassification:
    """Classification result for a semantic change."""
    element_mapping: Any  # ElementMapping
    category: ChangeCategory
    severity: ImpactSeverity
    breaking_change: BreakingChange
    reason: str
    confidence: float
    affected_elements: List[str] = None
    suggested_actions: List[str] = None

    def __post_init__(self):
        if self.affected_elements is None:
            self.affected_elements = []
        if self.suggested_actions is None:
            self.suggested_actions = []


@dataclass
class ClassificationSummary:
    """Summary of classification results."""
    total_changes: int
    breaking_changes: int
    non_breaking_changes: int
    high_severity_changes: int
    medium_severity_changes: int
    low_severity_changes: int
    categories_breakdown: Dict[str, int]
    average_confidence: float
    risk_assessment: str


class ChangeClassifier:
    """Classifies semantic changes by type and impact severity."""

    def __init__(self):
        pass

    def classify_changes(self, element_mappings: List[ElementMapping]) -> List[ChangeClassification]:
        """Classify a list of element mappings."""
        classifications = []

        for mapping in element_mappings:
            try:
                classification = self._classify_single_change(mapping)
                classifications.append(classification)
            except Exception as e:
                # Create a fallback classification
                classification = ChangeClassification(
                    element_mapping=mapping,
                    category=ChangeCategory.LOGIC_CHANGE,
                    severity=ImpactSeverity.UNKNOWN,
                    breaking_change=BreakingChange.UNKNOWN,
                    reason=f"Classification failed: {str(e)}",
                    confidence=0.0
                )
                classifications.append(classification)

        return classifications

    def _classify_single_change(self, mapping: ElementMapping) -> ChangeClassification:
        """Classify a single change based on its type and context."""

        change = mapping.semantic_change
        elements = mapping.mapped_elements

        # Apply classification rules based on change type
        if change.change_type == ChangeType.FUNCTION_SIGNATURE:
            return self._classify_function_signature_change(mapping)
        elif change.change_type == ChangeType.FUNCTION_LOGIC:
            return self._classify_function_logic_change(mapping)
        elif change.change_type == ChangeType.CLASS_DEFINITION:
            return self._classify_class_change(mapping)
        elif change.change_type in [ChangeType.IMPORT_ADDED, ChangeType.IMPORT_REMOVED]:
            return self._classify_import_change(mapping)
        elif change.change_type in [ChangeType.VARIABLE_ADDED, ChangeType.VARIABLE_REMOVED]:
            return self._classify_variable_change(mapping)
        else:
            return self._classify_generic_change(mapping)

    def _classify_function_signature_change(self, mapping: ElementMapping) -> ChangeClassification:
        """Classify function signature changes."""
        change = mapping.semantic_change

        # Analyze the signature change
        old_content = change.old_content or ""
        new_content = change.new_content or ""

        # Check for parameter additions/removals
        if self._has_parameter_changes(old_content, new_content):
            # Parameter changes are typically breaking
            severity = ImpactSeverity.HIGH
            breaking = BreakingChange.BREAKING
            reason = "Function signature changed with parameter modifications"
            suggested_actions = [
                "Update all function calls to match new signature",
                "Check for default parameter compatibility",
                "Update documentation and type hints"
            ]
        elif self._has_return_type_changes(old_content, new_content):
            # Return type changes might be breaking
            severity = ImpactSeverity.MEDIUM
            breaking = BreakingChange.BREAKING
            reason = "Function return type changed"
            suggested_actions = [
                "Update return type annotations",
                "Verify callers handle new return type"
            ]
        else:
            # Minor signature changes (e.g., whitespace, formatting)
            severity = ImpactSeverity.LOW
            breaking = BreakingChange.NON_BREAKING
            reason = "Minor function signature formatting changes"
            suggested_actions = ["Review for consistency"]

        return ChangeClassification(
            element_mapping=mapping,
            category=ChangeCategory.SIGNATURE_CHANGE,
            severity=severity,
            breaking_change=breaking,
            reason=reason,
            confidence=0.9,
            affected_elements=[change.element_name],
            suggested_actions=suggested_actions
        )

    def _classify_function_logic_change(self, mapping: ElementMapping) -> ChangeClassification:
        """Classify function logic changes."""
        change = mapping.semantic_change

        # Analyze the logic change impact
        old_content = change.old_content or ""
        new_content = change.new_content or ""

        # Check if it's a significant logic change
        if self._is_significant_logic_change(old_content, new_content):
            severity = ImpactSeverity.MEDIUM
            breaking = BreakingChange.NON_BREAKING  # Usually not breaking unless behavior changes drastically
            reason = "Function logic significantly modified"
            suggested_actions = [
                "Review logic changes for correctness",
                "Update tests to cover new behavior",
                "Check for side effect changes"
            ]
        else:
            # Minor logic changes
            severity = ImpactSeverity.LOW
            breaking = BreakingChange.NON_BREAKING
            reason = "Minor function logic modifications"
            suggested_actions = ["Verify functionality remains intact"]

        return ChangeClassification(
            element_mapping=mapping,
            category=ChangeCategory.LOGIC_CHANGE,
            severity=severity,
            breaking_change=breaking,
            reason=reason,
            confidence=0.8,
            affected_elements=[change.element_name],
            suggested_actions=suggested_actions
        )

    def _classify_class_change(self, mapping: ElementMapping) -> ChangeClassification:
        """Classify class definition changes."""
        change = mapping.semantic_change

        # Class changes are typically significant
        severity = ImpactSeverity.HIGH
        breaking = BreakingChange.BREAKING
        reason = "Class definition modified"

        suggested_actions = [
            "Update class instantiations",
            "Review inheritance and interfaces",
            "Update documentation",
            "Check for breaking API changes"
        ]

        return ChangeClassification(
            element_mapping=mapping,
            category=ChangeCategory.STRUCTURE_CHANGE,
            severity=severity,
            breaking_change=breaking,
            reason=reason,
            confidence=0.9,
            affected_elements=[change.element_name],
            suggested_actions=suggested_actions
        )

    def _classify_import_change(self, mapping: ElementMapping) -> ChangeClassification:
        """Classify import changes."""
        change = mapping.semantic_change

        if change.change_type == ChangeType.IMPORT_ADDED:
            severity = ImpactSeverity.LOW
            breaking = BreakingChange.NON_BREAKING
            reason = "New import added"
            suggested_actions = ["Ensure dependency is available", "Check import compatibility"]
        else:  # IMPORT_REMOVED
            severity = ImpactSeverity.MEDIUM
            breaking = BreakingChange.BREAKING
            reason = "Import removed"
            suggested_actions = [
                "Verify import is no longer needed",
                "Check for unused import warnings",
                "Update requirements if external dependency"
            ]

        return ChangeClassification(
            element_mapping=mapping,
            category=ChangeCategory.IMPORT_CHANGE,
            severity=severity,
            breaking_change=breaking,
            reason=reason,
            confidence=0.8,
            affected_elements=[change.element_name],
            suggested_actions=suggested_actions
        )

    def _classify_variable_change(self, mapping: ElementMapping) -> ChangeClassification:
        """Classify variable changes."""
        change = mapping.semantic_change

        if change.change_type == ChangeType.VARIABLE_ADDED:
            severity = ImpactSeverity.LOW
            breaking = BreakingChange.NON_BREAKING
            reason = "New variable added"
            suggested_actions = ["Review variable purpose", "Check naming conventions"]
        else:  # VARIABLE_REMOVED
            severity = ImpactSeverity.MEDIUM
            breaking = BreakingChange.BREAKING
            reason = "Variable removed"
            suggested_actions = [
                "Find all usages and update",
                "Check for configuration impacts",
                "Update documentation"
            ]

        return ChangeClassification(
            element_mapping=mapping,
            category=ChangeCategory.CONFIGURATION_CHANGE,
            severity=severity,
            breaking_change=breaking,
            reason=reason,
            confidence=0.7,
            affected_elements=[change.element_name],
            suggested_actions=suggested_actions
        )

    def _classify_generic_change(self, mapping: ElementMapping) -> ChangeClassification:
        """Classify generic/unknown changes."""
        change = mapping.semantic_change

        return ChangeClassification(
            element_mapping=mapping,
            category=ChangeCategory.LOGIC_CHANGE,
            severity=ImpactSeverity.MEDIUM,
            breaking_change=BreakingChange.UNKNOWN,
            reason=f"Generic {change.change_type.value} change",
            confidence=0.5,
            affected_elements=[change.element_name] if change.element_name else [],
            suggested_actions=["Review change manually", "Test functionality"]
        )

    def _has_parameter_changes(self, old_content: str, new_content: str) -> bool:
        """Check if function parameters changed."""
        # Simple heuristic: look for parameter patterns
        import re

        old_params = re.findall(r'def\s+\w+\s*\(([^)]*)\)', old_content)
        new_params = re.findall(r'def\s+\w+\s*\(([^)]*)\)', new_content)

        if old_params and new_params:
            return old_params[0].strip() != new_params[0].strip()

        return False

    def _has_return_type_changes(self, old_content: str, new_content: str) -> bool:
        """Check if return type annotations changed."""
        # Look for type annotations after parameters
        import re

        old_return = re.findall(r'\)\s*->\s*([^:]+)', old_content)
        new_return = re.findall(r'\)\s*->\s*([^:]+)', new_content)

        if old_return and new_return:
            return old_return[0].strip() != new_return[0].strip()

        return False

    def _is_significant_logic_change(self, old_content: str, new_content: str) -> bool:
        """Determine if a logic change is significant."""
        # Heuristics for significant changes:
        # - Large line count difference
        # - Addition/removal of control structures
        # - Complex expressions

        old_lines = len(old_content.split('\n'))
        new_lines = len(new_content.split('\n'))

        # Significant line count change
        if abs(old_lines - new_lines) > 2:
            return True

        # Check for control structures
        control_keywords = ['if ', 'for ', 'while ', 'try:', 'except:', 'with ']
        old_controls = sum(1 for keyword in control_keywords if keyword in old_content)
        new_controls = sum(1 for keyword in control_keywords if keyword in new_content)

        if abs(old_controls - new_controls) > 0:
            return True

        return False

    def get_classification_summary(self, classifications: List[ChangeClassification]) -> ClassificationSummary:
        """Generate a summary of classification results."""
        if not classifications:
            return ClassificationSummary(
                total_changes=0,
                breaking_changes=0,
                non_breaking_changes=0,
                high_severity_changes=0,
                medium_severity_changes=0,
                low_severity_changes=0,
                categories_breakdown={},
                average_confidence=0.0,
                risk_assessment="No changes to analyze"
            )

        total_changes = len(classifications)
        breaking_changes = sum(1 for c in classifications if c.breaking_change == BreakingChange.BREAKING)
        non_breaking_changes = sum(1 for c in classifications if c.breaking_change == BreakingChange.NON_BREAKING)

        high_severity = sum(1 for c in classifications if c.severity == ImpactSeverity.HIGH)
        medium_severity = sum(1 for c in classifications if c.severity == ImpactSeverity.MEDIUM)
        low_severity = sum(1 for c in classifications if c.severity == ImpactSeverity.LOW)

        categories_breakdown = {}
        for classification in classifications:
            cat_name = classification.category.value
            categories_breakdown[cat_name] = categories_breakdown.get(cat_name, 0) + 1

        average_confidence = sum(c.confidence for c in classifications) / total_changes

        risk_assessment = self._assess_overall_risk(classifications)

        return ClassificationSummary(
            total_changes=total_changes,
            breaking_changes=breaking_changes,
            non_breaking_changes=non_breaking_changes,
            high_severity_changes=high_severity,
            medium_severity_changes=medium_severity,
            low_severity_changes=low_severity,
            categories_breakdown=categories_breakdown,
            average_confidence=average_confidence,
            risk_assessment=risk_assessment
        )

    def _assess_overall_risk(self, classifications: List[ChangeClassification]) -> str:
        """Assess the overall risk level of the changes."""
        if not classifications:
            return "No risk"

        high_count = sum(1 for c in classifications if c.severity == ImpactSeverity.HIGH)
        breaking_count = sum(1 for c in classifications if c.breaking_change == BreakingChange.BREAKING)

        total_changes = len(classifications)
        high_percentage = high_count / total_changes
        breaking_percentage = breaking_count / total_changes

        if high_percentage > 0.5 or breaking_percentage > 0.7:
            return "HIGH RISK - Major breaking changes detected"
        elif high_percentage > 0.3 or breaking_percentage > 0.5:
            return "MEDIUM RISK - Some breaking changes present"
        elif breaking_count > 0:
            return "LOW RISK - Minor breaking changes"
        else:
            return "VERY LOW RISK - Non-breaking changes only"

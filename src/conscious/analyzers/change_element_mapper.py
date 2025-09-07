"""Change Element Mapper - Maps semantic changes to specific AST elements."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path

from .semantic_diff_parser import SemanticChange, ChangeType
from ..parsers.tree_sitter_parser import TreeSitterParser, Function, Class


@dataclass
class MappedElement:
    """Represents a code element mapped from a semantic change."""
    element_type: str  # 'function', 'class', 'method', 'variable'
    name: str
    file_path: str
    start_line: int
    end_line: int
    ast_node: Optional[object] = None  # Tree-sitter AST node
    confidence: float = 1.0
    context: Optional[str] = None


@dataclass
class ElementMapping:
    """Maps a semantic change to specific code elements."""
    semantic_change: SemanticChange
    mapped_elements: List[MappedElement]
    mapping_confidence: float
    mapping_reason: str


class ChangeElementMapper:
    """Maps semantic changes to specific AST elements."""

    def __init__(self):
        self.tree_sitter = TreeSitterParser()
        self._file_cache = {}  # Cache for parsed file content

    def map_changes_to_elements(self, semantic_changes: List[SemanticChange],
                               file_contents: Dict[str, str]) -> List[ElementMapping]:
        """Map semantic changes to specific AST elements."""
        mappings = []

        for change in semantic_changes:
            try:
                mapping = self._map_single_change(change, file_contents)
                if mapping:
                    mappings.append(mapping)
            except Exception as e:
                # Create a mapping with low confidence for failed mappings
                mapping = ElementMapping(
                    semantic_change=change,
                    mapped_elements=[],
                    mapping_confidence=0.0,
                    mapping_reason=f"Mapping failed: {str(e)}"
                )
                mappings.append(mapping)

        return mappings

    def _map_single_change(self, change: SemanticChange,
                          file_contents: Dict[str, str]) -> Optional[ElementMapping]:
        """Map a single semantic change to AST elements."""

        # Get file content
        content = file_contents.get(change.file_path)
        if not content:
            return None

        # Parse with Tree-sitter
        try:
            tree = self.tree_sitter.parse_file(content, self._detect_language(change.file_path))
            if not tree:
                return None
        except Exception:
            return None

        mapped_elements = []

        # Map based on change type
        if change.change_type == ChangeType.FUNCTION_SIGNATURE:
            elements = self._map_function_change(change, tree, content)
            mapped_elements.extend(elements)

        elif change.change_type == ChangeType.FUNCTION_LOGIC:
            elements = self._map_function_logic_change(change, tree, content)
            mapped_elements.extend(elements)

        elif change.change_type == ChangeType.CLASS_DEFINITION:
            elements = self._map_class_change(change, tree, content)
            mapped_elements.extend(elements)

        elif change.change_type in [ChangeType.IMPORT_ADDED, ChangeType.IMPORT_REMOVED]:
            elements = self._map_import_change(change, tree, content)
            mapped_elements.extend(elements)

        elif change.change_type in [ChangeType.VARIABLE_ADDED, ChangeType.VARIABLE_REMOVED]:
            elements = self._map_variable_change(change, tree, content)
            mapped_elements.extend(elements)

        # Calculate mapping confidence
        confidence = self._calculate_mapping_confidence(mapped_elements, change)

        return ElementMapping(
            semantic_change=change,
            mapped_elements=mapped_elements,
            mapping_confidence=confidence,
            mapping_reason=self._get_mapping_reason(mapped_elements, change)
        )

    def _map_function_change(self, change: SemanticChange, tree, content: str) -> List[MappedElement]:
        """Map function signature changes to AST elements."""
        elements = []

        # Extract function name from change
        func_name = change.element_name

        # Parse the AST to find functions
        functions = self.tree_sitter.get_functions(tree, self._detect_language(change.file_path))

        # Find matching functions
        for func in functions:
            if func.name == func_name:
                element = MappedElement(
                    element_type="function",
                    name=func.name,
                    file_path=change.file_path,
                    start_line=func.start_line,
                    end_line=func.end_line,
                    ast_node=None,  # Could store actual AST node here
                    confidence=0.9,
                    context=f"Function signature changed: {func.name}"
                )
                elements.append(element)

        return elements

    def _map_function_logic_change(self, change: SemanticChange, tree, content: str) -> List[MappedElement]:
        """Map function logic changes to AST elements."""
        elements = []

        # For logic changes, we need to infer the function from context
        # This is more complex as the change might not contain the function name
        func_name = change.element_name or self._infer_function_from_context(change, content)

        if func_name:
            functions = self.tree_sitter.get_functions(tree, self._detect_language(change.file_path))

            for func in functions:
                if func.name == func_name:
                    element = MappedElement(
                        element_type="function",
                        name=func.name,
                        file_path=change.file_path,
                        start_line=func.start_line,
                        end_line=func.end_line,
                        confidence=0.7,  # Lower confidence for inferred mappings
                        context=f"Function logic changed: {func.name}"
                    )
                    elements.append(element)

        return elements

    def _map_class_change(self, change: SemanticChange, tree, content: str) -> List[MappedElement]:
        """Map class definition changes to AST elements."""
        elements = []

        class_name = change.element_name
        classes = self.tree_sitter.get_classes(tree, self._detect_language(change.file_path))

        for cls in classes:
            if cls.name == class_name:
                element = MappedElement(
                    element_type="class",
                    name=cls.name,
                    file_path=change.file_path,
                    start_line=cls.start_line,
                    end_line=cls.end_line,
                    confidence=0.9,
                    context=f"Class definition changed: {cls.name}"
                )
                elements.append(element)

        return elements

    def _map_import_change(self, change: SemanticChange, tree, content: str) -> List[MappedElement]:
        """Map import changes to AST elements."""
        # For imports, we create a synthetic element since they're not in the AST
        element = MappedElement(
            element_type="import",
            name=change.element_name,
            file_path=change.file_path,
            start_line=change.line_number,
            end_line=change.line_number,
            confidence=0.8,
            context=f"Import changed: {change.element_name}"
        )
        return [element]

    def _map_variable_change(self, change: SemanticChange, tree, content: str) -> List[MappedElement]:
        """Map variable changes to AST elements."""
        # For variables, we create a synthetic element
        element = MappedElement(
            element_type="variable",
            name=change.element_name,
            file_path=change.file_path,
            start_line=change.line_number,
            end_line=change.line_number,
            confidence=0.6,  # Lower confidence for variables
            context=f"Variable changed: {change.element_name}"
        )
        return [element]

    def _infer_function_from_context(self, change: SemanticChange, content: str) -> Optional[str]:
        """Try to infer function name from change context."""
        # Look for function definitions in the vicinity of the change
        lines = content.split('\n')
        start_line = max(0, change.line_number - 10)  # Look 10 lines before
        end_line = min(len(lines), change.line_number + 10)  # Look 10 lines after

        for i in range(start_line, end_line):
            line = lines[i].strip()
            if line.startswith('def '):
                # Extract function name
                import re
                match = re.match(r'def\s+(\w+)', line)
                if match:
                    return match.group(1)

        return None

    def _calculate_mapping_confidence(self, elements: List[MappedElement],
                                    change: SemanticChange) -> float:
        """Calculate confidence score for the mapping."""
        if not elements:
            return 0.0

        # Average confidence of all mapped elements
        total_confidence = sum(element.confidence for element in elements)
        avg_confidence = total_confidence / len(elements)

        # Adjust based on change type reliability
        type_multipliers = {
            ChangeType.FUNCTION_SIGNATURE: 1.0,
            ChangeType.CLASS_DEFINITION: 1.0,
            ChangeType.IMPORT_ADDED: 0.9,
            ChangeType.IMPORT_REMOVED: 0.9,
            ChangeType.FUNCTION_LOGIC: 0.8,
            ChangeType.VARIABLE_ADDED: 0.7,
            ChangeType.VARIABLE_REMOVED: 0.7,
        }

        type_multiplier = type_multipliers.get(change.change_type, 0.5)
        return avg_confidence * type_multiplier

    def _get_mapping_reason(self, elements: List[MappedElement],
                           change: SemanticChange) -> str:
        """Get a human-readable reason for the mapping."""
        if not elements:
            return f"No elements mapped for {change.change_type.value}"

        element_names = [e.name for e in elements]
        return f"Mapped to {len(elements)} element(s): {', '.join(element_names)}"

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = file_path.lower().split('.')[-1]
        language_map = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'java': 'java'
        }
        return language_map.get(ext, 'python')

    def get_mapping_summary(self, mappings: List[ElementMapping]) -> Dict:
        """Get a summary of element mappings."""
        summary = {
            'total_mappings': len(mappings),
            'successful_mappings': 0,
            'failed_mappings': 0,
            'average_confidence': 0.0,
            'element_types': {},
            'change_types': {}
        }

        total_confidence = 0.0

        for mapping in mappings:
            if mapping.mapping_confidence > 0:
                summary['successful_mappings'] += 1
            else:
                summary['failed_mappings'] += 1

            total_confidence += mapping.mapping_confidence

            # Count element types
            for element in mapping.mapped_elements:
                summary['element_types'][element.element_type] = \
                    summary['element_types'].get(element.element_type, 0) + 1

            # Count change types
            change_type = mapping.semantic_change.change_type.value
            summary['change_types'][change_type] = \
                summary['change_types'].get(change_type, 0) + 1

        if mappings:
            summary['average_confidence'] = total_confidence / len(mappings)

        return summary

"""Semantic diff parser - understands WHAT changed, not just that something changed."""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum

from ..parsers.tree_sitter_parser import TreeSitterParser, Function, Class, Import


class ChangeType(Enum):
    """Types of semantic changes."""
    FUNCTION_SIGNATURE = "function_signature"
    FUNCTION_LOGIC = "function_logic"
    CLASS_DEFINITION = "class_definition"
    CLASS_METHOD = "class_method"
    IMPORT_ADDED = "import_added"
    IMPORT_REMOVED = "import_removed"
    VARIABLE_ADDED = "variable_added"
    VARIABLE_REMOVED = "variable_removed"
    TYPE_ANNOTATION = "type_annotation"
    UNKNOWN = "unknown"


@dataclass
class SemanticChange:
    """Represents a semantic change with context."""
    change_type: ChangeType
    element_name: str
    element_type: str  # function, class, variable, import
    file_path: str
    line_number: int
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    context: Optional[str] = None
    confidence: float = 1.0


@dataclass
class DiffHunk:
    """Represents a diff hunk with semantic information."""
    file_path: str
    start_line: int
    end_line: int
    changes: List[str]
    semantic_changes: List[SemanticChange]


class SemanticDiffParser:
    """Parses diffs semantically to understand what actually changed."""

    def __init__(self):
        self.tree_sitter = TreeSitterParser()

        # Patterns for detecting semantic changes
        self.function_pattern = re.compile(r'^\s*def\s+(\w+)\s*\(')
        self.class_pattern = re.compile(r'^\s*class\s+(\w+)')
        self.import_pattern = re.compile(r'^\s*(?:import|from)\s+')
        self.variable_pattern = re.compile(r'^\s*(\w+)\s*=\s*')

    def parse_semantic_changes(self, diff_content: str, file_path: str) -> List[SemanticChange]:
        """Parse diff content and extract semantic changes."""
        changes = []

        # Split diff into hunks
        hunks = self._split_into_hunks(diff_content)

        for hunk in hunks:
            hunk_changes = self._analyze_hunk(hunk, file_path)
            changes.extend(hunk_changes)

        return changes

    def _split_into_hunks(self, diff_content: str) -> List[DiffHunk]:
        """Split diff content into logical hunks."""
        hunks = []
        lines = diff_content.split('\n')

        current_hunk = None
        current_file = None

        for i, line in enumerate(lines):
            if line.startswith('+++ b/'):
                current_file = line[6:].strip()  # Remove '+++ b/' prefix
            elif line.startswith('@@'):
                # New hunk
                if current_hunk:
                    hunks.append(current_hunk)

                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                match = re.match(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@', line)
                if match:
                    new_start = int(match.group(3))
                    current_hunk = DiffHunk(
                        file_path=current_file or "unknown",
                        start_line=new_start,
                        end_line=new_start,
                        changes=[],
                        semantic_changes=[]
                    )
            elif current_hunk and line.startswith(('+', '-')):
                current_hunk.changes.append(line)
                current_hunk.end_line += 1

        # Add final hunk
        if current_hunk:
            hunks.append(current_hunk)

        return hunks

    def _analyze_hunk(self, hunk: DiffHunk, file_path: str) -> List[SemanticChange]:
        """Analyze a diff hunk for semantic changes."""
        changes = []

        # Group changes by type
        added_lines = [line[1:] for line in hunk.changes if line.startswith('+')]
        removed_lines = [line[1:] for line in hunk.changes if line.startswith('-')]

        # Analyze added content
        for line_num, line in enumerate(added_lines, hunk.start_line):
            change = self._classify_line(line, line_num, file_path, "added")
            if change:
                changes.append(change)

        # Analyze removed content
        for line_num, line in enumerate(removed_lines, hunk.start_line):
            change = self._classify_line(line, line_num, file_path, "removed")
            if change:
                changes.append(change)

        # Look for function signature changes
        func_changes = self._analyze_function_changes(added_lines, removed_lines, hunk.start_line, file_path)
        changes.extend(func_changes)

        # If no specific function changes found but we have changes, classify as function logic changes
        if not func_changes and added_lines and removed_lines:
            # Try to infer function context from the diff
            context_func = self._infer_function_context(added_lines + removed_lines)
            if context_func:
                changes.append(SemanticChange(
                    change_type=ChangeType.FUNCTION_LOGIC,
                    element_name=context_func,
                    element_type="function",
                    file_path=file_path,
                    line_number=hunk.start_line,
                    context="Function logic modified"
                ))

        return changes

    def _infer_function_context(self, lines: List[str]) -> Optional[str]:
        """Try to infer function name from diff context."""
        # Look for function definitions in the context
        for line in lines:
            if func_match := self.function_pattern.match(line.strip()):
                return func_match.group(1)
        return None

    def _classify_line(self, line: str, line_num: int, file_path: str, change_type: str) -> Optional[SemanticChange]:
        """Classify a single line change."""
        line = line.strip()

        # Function definition
        if func_match := self.function_pattern.match(line):
            func_name = func_match.group(1)
            return SemanticChange(
                change_type=ChangeType.FUNCTION_SIGNATURE if change_type == "added" else ChangeType.FUNCTION_LOGIC,
                element_name=func_name,
                element_type="function",
                file_path=file_path,
                line_number=line_num,
                new_content=line if change_type == "added" else None,
                old_content=line if change_type == "removed" else None
            )

        # Class definition
        elif class_match := self.class_pattern.match(line):
            class_name = class_match.group(1)
            return SemanticChange(
                change_type=ChangeType.CLASS_DEFINITION,
                element_name=class_name,
                element_type="class",
                file_path=file_path,
                line_number=line_num,
                new_content=line if change_type == "added" else None,
                old_content=line if change_type == "removed" else None
            )

        # Import statement
        elif self.import_pattern.match(line):
            return SemanticChange(
                change_type=ChangeType.IMPORT_ADDED if change_type == "added" else ChangeType.IMPORT_REMOVED,
                element_name=line,
                element_type="import",
                file_path=file_path,
                line_number=line_num,
                new_content=line if change_type == "added" else None,
                old_content=line if change_type == "removed" else None
            )

        # Variable assignment
        elif var_match := self.variable_pattern.match(line):
            var_name = var_match.group(1)
            return SemanticChange(
                change_type=ChangeType.VARIABLE_ADDED if change_type == "added" else ChangeType.VARIABLE_REMOVED,
                element_name=var_name,
                element_type="variable",
                file_path=file_path,
                line_number=line_num,
                new_content=line if change_type == "added" else None,
                old_content=line if change_type == "removed" else None
            )

        return None

    def _analyze_function_changes(self, added_lines: List[str], removed_lines: List[str],
                                 start_line: int, file_path: str) -> List[SemanticChange]:
        """Analyze function signature changes between added and removed lines."""
        changes = []

        # Find function definitions in both added and removed
        added_funcs = self._extract_functions_from_lines(added_lines)
        removed_funcs = self._extract_functions_from_lines(removed_lines)

        # Compare functions with same name (signature changes)
        for added_func in added_funcs:
            for removed_func in removed_funcs:
                if added_func['name'] == removed_func['name']:
                    # Compare signatures
                    if added_func['signature'] != removed_func['signature']:
                        changes.append(SemanticChange(
                            change_type=ChangeType.FUNCTION_SIGNATURE,
                            element_name=added_func['name'],
                            element_type="function",
                            file_path=file_path,
                            line_number=start_line,
                            old_content=removed_func['signature'],
                            new_content=added_func['signature'],
                            context="Function signature changed"
                        ))

        return changes

    def _analyze_function_logic_changes(self, added_lines: List[str], removed_lines: List[str],
                                       start_line: int, file_path: str) -> List[SemanticChange]:
        """Analyze function logic changes (changes within function bodies)."""
        changes = []

        # If we have both added and removed lines, and they're not just function definitions,
        # classify as function logic changes
        if added_lines and removed_lines:
            # Check if we're dealing with function body changes
            has_function_context = any(self.function_pattern.match(line.strip()) for line in added_lines + removed_lines)

            if has_function_context or (len(added_lines) > 0 and len(removed_lines) > 0):
                # Look for the function name in context
                all_lines = added_lines + removed_lines
                func_names = []
                for line in all_lines:
                    if func_match := self.function_pattern.match(line.strip()):
                        func_names.append(func_match.group(1))

                if func_names:
                    # Use the first function name found
                    func_name = func_names[0]
                    changes.append(SemanticChange(
                        change_type=ChangeType.FUNCTION_LOGIC,
                        element_name=func_name,
                        element_type="function",
                        file_path=file_path,
                        line_number=start_line,
                        context="Function logic changed"
                    ))

        return changes

    def _extract_functions_from_lines(self, lines: List[str]) -> List[Dict]:
        """Extract function definitions from a list of lines."""
        functions = []

        for line in lines:
            if func_match := self.function_pattern.match(line.strip()):
                func_name = func_match.group(1)
                # Extract function signature (simplified)
                signature = line.strip()
                functions.append({
                    'name': func_name,
                    'signature': signature
                })

        return functions

    def get_change_summary(self, changes: List[SemanticChange]) -> Dict:
        """Get a summary of semantic changes."""
        summary = {
            'total_changes': len(changes),
            'change_types': {},
            'affected_elements': set(),
            'files_affected': set()
        }

        for change in changes:
            # Count change types
            change_type_str = change.change_type.value
            summary['change_types'][change_type_str] = summary['change_types'].get(change_type_str, 0) + 1

            # Track affected elements
            summary['affected_elements'].add(f"{change.element_type}:{change.element_name}")

            # Track affected files
            summary['files_affected'].add(change.file_path)

        summary['affected_elements'] = len(summary['affected_elements'])
        summary['files_affected'] = len(summary['files_affected'])

        return summary

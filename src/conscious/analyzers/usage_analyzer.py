"""Usage Analyzer - Finds all usages of functions, classes, and variables across the codebase."""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple, Any
from pathlib import Path
import re
import ast

from ..parsers.tree_sitter_parser import TreeSitterParser
from .call_graph_analyzer import CallGraphAnalyzer
from .import_graph_builder import ImportGraph


@dataclass
class UsageLocation:
    """Represents a single usage location."""
    file_path: str
    line_number: int
    column_start: int
    column_end: int
    context: str  # Surrounding code context
    usage_type: str  # 'call', 'attribute', 'inheritance', 'import', 'assignment'
    confidence: float = 1.0


@dataclass
class UsageResult:
    """Complete usage analysis result for a symbol."""
    symbol_name: str
    symbol_type: str  # 'function', 'class', 'variable', 'method'
    definition_file: str
    definition_line: int
    usages: List[UsageLocation]
    total_usages: int
    unique_files: int
    most_used_in: Optional[str] = None


@dataclass
class UsageAnalysis:
    """Complete usage analysis for multiple symbols."""
    results: Dict[str, UsageResult]
    cross_references: Dict[str, List[str]]  # symbol -> symbols it references
    referenced_by: Dict[str, List[str]]    # symbol -> symbols that reference it
    analysis_summary: Dict[str, Any]


class UsageAnalyzer:
    """Analyzes usage patterns of functions, classes, and variables across the codebase."""

    def __init__(self):
        self.tree_sitter = TreeSitterParser()
        self.call_graph_analyzer = CallGraphAnalyzer()

    def analyze_usages(self, file_paths: List[str], target_symbols: Optional[List[str]] = None,
                      import_graph: Optional[ImportGraph] = None) -> UsageAnalysis:
        """Analyze usages of specified symbols across all files."""

        # Build comprehensive symbol index
        symbol_index = self._build_symbol_index(file_paths)

        # If no target symbols specified, analyze all symbols
        if target_symbols is None:
            target_symbols = list(symbol_index.keys())

        results = {}
        cross_references = {}
        referenced_by = {}

        for symbol in target_symbols:
            if symbol in symbol_index:
                usage_result = self._analyze_symbol_usage(symbol, symbol_index, file_paths)
                if usage_result:
                    results[symbol] = usage_result

                    # Build cross-reference data
                    self._build_cross_references(symbol, usage_result, cross_references, referenced_by)

        # Generate analysis summary
        summary = self._generate_analysis_summary(results, file_paths)

        return UsageAnalysis(
            results=results,
            cross_references=cross_references,
            referenced_by=referenced_by,
            analysis_summary=summary
        )

    def _build_symbol_index(self, file_paths: List[str]) -> Dict[str, Dict]:
        """Build an index of all symbols (functions, classes, variables) in the codebase."""
        symbol_index = {}

        for file_path in file_paths:
            if not file_path.endswith('.py'):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse with AST to find definitions
                tree = ast.parse(content, filename=file_path)
                self._extract_symbol_definitions(tree, file_path, symbol_index)

            except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
                continue

        return symbol_index

    def _extract_symbol_definitions(self, tree: ast.AST, file_path: str,
                                  symbol_index: Dict[str, Dict]):
        """Extract symbol definitions from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                symbol_index[node.name] = {
                    'type': 'function',
                    'file': file_path,
                    'line': node.lineno,
                    'node': node
                }
            elif isinstance(node, ast.ClassDef):
                symbol_index[node.name] = {
                    'type': 'class',
                    'file': file_path,
                    'line': node.lineno,
                    'node': node
                }
            elif isinstance(node, ast.Assign):
                # Extract variable assignments
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        symbol_index[target.id] = {
                            'type': 'variable',
                            'file': file_path,
                            'line': node.lineno,
                            'node': node
                        }

    def _analyze_symbol_usage(self, symbol: str, symbol_index: Dict[str, Dict],
                            file_paths: List[str]) -> Optional[UsageResult]:
        """Analyze usage of a specific symbol across all files."""
        if symbol not in symbol_index:
            return None

        symbol_info = symbol_index[symbol]
        usages = []

        for file_path in file_paths:
            if not file_path.endswith('.py'):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n')

                # Find usages in this file
                file_usages = self._find_symbol_usages_in_file(
                    symbol, content, lines, file_path, symbol_info['type']
                )
                usages.extend(file_usages)

            except (UnicodeDecodeError, FileNotFoundError):
                continue

        if not usages:
            return None

        # Calculate statistics
        unique_files = len(set(usage.file_path for usage in usages))
        most_used_file = max(
            set((usage.file_path, sum(1 for u in usages if u.file_path == usage.file_path))
                for usage in usages),
            key=lambda x: x[1]
        )[0] if usages else None

        return UsageResult(
            symbol_name=symbol,
            symbol_type=symbol_info['type'],
            definition_file=symbol_info['file'],
            definition_line=symbol_info['line'],
            usages=usages,
            total_usages=len(usages),
            unique_files=unique_files,
            most_used_in=most_used_file
        )

    def _find_symbol_usages_in_file(self, symbol: str, content: str, lines: List[str],
                                  file_path: str, symbol_type: str) -> List[UsageLocation]:
        """Find all usages of a symbol in a specific file."""
        usages = []

        # Use regex patterns to find symbol usages
        patterns = self._get_usage_patterns(symbol, symbol_type)

        for pattern_name, pattern in patterns.items():
            for match in re.finditer(pattern, content, re.MULTILINE):
                line_number = content[:match.start()].count('\n') + 1
                column_start = match.start() - content.rfind('\n', 0, match.start())
                column_end = match.end() - content.rfind('\n', 0, match.end())

                # Get context (surrounding lines)
                context_start = max(0, line_number - 2)
                context_end = min(len(lines), line_number + 2)
                context_lines = lines[context_start:context_end]
                context = '\n'.join(f"{i+context_start+1:3d}: {line}"
                                  for i, line in enumerate(context_lines))

                usage = UsageLocation(
                    file_path=file_path,
                    line_number=line_number,
                    column_start=column_start,
                    column_end=column_end,
                    context=context,
                    usage_type=pattern_name,
                    confidence=self._calculate_usage_confidence(pattern_name, match.group())
                )
                usages.append(usage)

        return usages

    def _get_usage_patterns(self, symbol: str, symbol_type: str) -> Dict[str, str]:
        """Get regex patterns for finding symbol usages."""
        # Escape special regex characters
        escaped_symbol = re.escape(symbol)

        patterns = {}

        if symbol_type in ['function', 'method']:
            # Function calls: symbol( or obj.symbol(
            patterns['call'] = rf'(?<!\w){escaped_symbol}\s*\('
            patterns['attribute_call'] = rf'\.\s*{escaped_symbol}\s*\('

        if symbol_type == 'class':
            # Class instantiation: symbol( or inheritance
            patterns['instantiation'] = rf'(?<!\w){escaped_symbol}\s*\('
            patterns['inheritance'] = rf'class\s+\w+\s*\([^)]*{escaped_symbol}[^)]*\)'

        # General attribute access
        patterns['attribute'] = rf'\.\s*{escaped_symbol}(?!\w)'

        # Variable usage (non-attribute)
        patterns['variable'] = rf'(?<!\w){escaped_symbol}(?!\w)'

        # Import statements
        patterns['import'] = rf'import\s+.*{escaped_symbol}|from\s+.*import\s+.*{escaped_symbol}'

        return patterns

    def _calculate_usage_confidence(self, pattern_name: str, match_text: str) -> float:
        """Calculate confidence score for a usage match."""
        confidence_map = {
            'call': 0.9,
            'attribute_call': 0.9,
            'instantiation': 0.9,
            'inheritance': 0.95,
            'attribute': 0.7,
            'variable': 0.6,
            'import': 0.8
        }
        return confidence_map.get(pattern_name, 0.5)

    def _build_cross_references(self, symbol: str, usage_result: UsageResult,
                              cross_references: Dict[str, List[str]],
                              referenced_by: Dict[str, List[str]]):
        """Build cross-reference relationships between symbols."""
        # This is a simplified version - in a full implementation,
        # you'd analyze the AST to find actual symbol relationships
        cross_references[symbol] = []
        referenced_by[symbol] = []

        # For now, we'll use the usage locations to infer some relationships
        for usage in usage_result.usages:
            if usage.usage_type in ['call', 'attribute_call', 'instantiation']:
                # This symbol is referenced by the file containing the usage
                file_symbols = referenced_by.get(symbol, [])
                if usage.file_path not in file_symbols:
                    file_symbols.append(usage.file_path)
                    referenced_by[symbol] = file_symbols

    def _generate_analysis_summary(self, results: Dict[str, UsageResult],
                                 file_paths: List[str]) -> Dict[str, Any]:
        """Generate summary statistics for the usage analysis."""
        total_symbols = len(results)
        total_usages = sum(result.total_usages for result in results.values())
        avg_usages_per_symbol = total_usages / total_symbols if total_symbols > 0 else 0

        symbol_types = {}
        for result in results.values():
            symbol_types[result.symbol_type] = symbol_types.get(result.symbol_type, 0) + 1

        # Find most used symbols
        most_used = sorted(results.items(),
                          key=lambda x: x[1].total_usages,
                          reverse=True)[:10]

        # Find symbols with broadest usage (most files)
        broadest_usage = sorted(results.items(),
                               key=lambda x: x[1].unique_files,
                               reverse=True)[:10]

        return {
            'total_symbols_analyzed': total_symbols,
            'total_usages_found': total_usages,
            'average_usages_per_symbol': avg_usages_per_symbol,
            'symbol_types_breakdown': symbol_types,
            'most_used_symbols': [(name, result.total_usages) for name, result in most_used],
            'broadest_usage_symbols': [(name, result.unique_files) for name, result in broadest_usage],
            'files_analyzed': len([f for f in file_paths if f.endswith('.py')])
        }

    def find_impacted_symbols(self, changed_symbols: List[str],
                            usage_analysis: UsageAnalysis) -> List[str]:
        """Find all symbols that could be impacted by changes to the given symbols."""
        impacted = set(changed_symbols)
        to_analyze = list(changed_symbols)

        # Use breadth-first search to find all impacted symbols
        while to_analyze:
            current_symbol = to_analyze.pop(0)

            # Find symbols that use this symbol
            if current_symbol in usage_analysis.referenced_by:
                for referencing_symbol in usage_analysis.referenced_by[current_symbol]:
                    if referencing_symbol not in impacted:
                        impacted.add(referencing_symbol)
                        to_analyze.append(referencing_symbol)

        return list(impacted)

    def get_usage_report(self, symbol: str, usage_result: UsageResult) -> str:
        """Generate a human-readable usage report for a symbol."""
        if not usage_result:
            return f"No usage data found for symbol: {symbol}"

        report = f"""
Usage Report for: {symbol} ({usage_result.symbol_type})
{'='*50}
Defined in: {usage_result.definition_file}:{usage_result.definition_line}
Total usages: {usage_result.total_usages}
Files using symbol: {usage_result.unique_files}
Most used in: {usage_result.most_used_in or 'N/A'}

Detailed Usages:
"""

        # Group usages by file
        usages_by_file = {}
        for usage in usage_result.usages:
            if usage.file_path not in usages_by_file:
                usages_by_file[usage.file_path] = []
            usages_by_file[usage.file_path].append(usage)

        for file_path, file_usages in usages_by_file.items():
            report += f"\n📁 {file_path} ({len(file_usages)} usages):\n"
            for usage in file_usages[:5]:  # Show first 5 usages per file
                report += f"  Line {usage.line_number}: {usage.usage_type} ({usage.confidence:.1f})\n"

            if len(file_usages) > 5:
                report += f"  ... and {len(file_usages) - 5} more usages\n"

        return report

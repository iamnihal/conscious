"""Command-line interface for the conscious tool."""

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.traceback import install
from pathlib import Path
import json
import time
from typing import Optional

from .core import DiffParser
from .analyzers import CallGraphAnalyzer, AdvancedImpactAnalyzer, AnalysisConfiguration
from .analyzers.advanced_impact_analyzer import ComprehensiveAnalysisResult

# Set up rich error handling
install()
console = Console()

def cli():
    """Conscious - Advanced Impact Analysis for Code Changes

    A comprehensive code change analyzer that understands the semantic impact
    of your changes across the entire codebase.

    USAGE:
        conscious diff.patch                    # Analyze with advanced impact analysis
        conscious diff.patch --basic           # Use basic analysis mode
        conscious diff.patch --format json     # JSON output
        conscious diff.patch --output report.html --format html  # HTML report

    FEATURES:
    • Semantic change detection (function signatures, logic changes)
    • Impact propagation with confidence scoring
    • Dependency analysis and usage tracking
    • Risk assessment and deployment recommendations
    • Breaking change detection
    """
    analyze()

@click.command()
@click.argument('diff_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for analysis results')
@click.option('--format', '-f', type=click.Choice(['text', 'json', 'html']), default='text', help='Output format')
@click.option('--call-graph/--no-call-graph', default=True, help='Generate call graph analysis')
@click.option('--impact-analysis', '-i', is_flag=True, help='Show impact analysis for changes')
@click.option('--advanced/--basic', default=True, help='Use advanced impact analysis system (default: advanced)')
@click.option('--include-usage/--no-usage', default=True, help='Include usage analysis (advanced mode only)')
@click.option('--include-dependencies/--no-dependencies', default=True, help='Include dependency tracking (advanced mode only)')
@click.option('--analysis-depth', type=int, default=3, help='Analysis depth for dependency tracking (advanced mode only)')
@click.option('--confidence-threshold', type=float, default=0.4, help='Minimum confidence threshold (advanced mode only)')
@click.option('--cache/--no-cache', default=True, help='Enable caching for performance')
@click.option('--cache-size', type=int, default=100, help='Maximum cache size (number of items)')
@click.option('--root-dir', type=click.Path(exists=True), help='Root directory for file resolution (advanced mode only)')
def analyze(diff_file, output, format, call_graph, impact_analysis, advanced, include_usage,
            include_dependencies, analysis_depth, confidence_threshold, cache, cache_size, root_dir):
    """Analyze changes in a diff file with syntax-aware impact assessment.

    BASIC MODE (--basic):
    Traditional analysis with call graphs and basic impact detection.
    Fast and suitable for quick reviews.

    ADVANCED MODE (--advanced):
    Comprehensive semantic analysis with:
    • Semantic change detection (function signatures, logic changes)
    • Impact propagation with confidence scoring
    • Dependency analysis and usage tracking
    • Risk assessment and deployment recommendations
    • Breaking change detection

    The advanced mode provides detailed insights into how your changes
    affect the entire codebase with actionable intelligence for deployment.
    """
    try:
        start_time = time.time()

        # Determine analysis mode
        if advanced:
            return _analyze_impact_advanced(diff_file, output, format, include_usage,
                                          include_dependencies, analysis_depth,
                                          confidence_threshold, cache, cache_size, root_dir)
        else:
            return _analyze_impact_basic(diff_file, output, format, call_graph,
                                       impact_analysis, cache, cache_size)

    except Exception as e:
        console.print(f"[red]❌ Error:[/red] {str(e)}")
        raise click.Abort()

def _analyze_impact_basic(diff_file, output, format, call_graph, impact_analysis, cache, cache_size):
    """Basic analysis mode using the original call graph analyzer."""
    console.print("🔍 [bold]Basic Analysis Mode[/bold]")
    console.print("📄 Parsing diff...")

    parser = DiffParser(str(diff_file), cache_enabled=cache, cache_size=cache_size)
    diff_files = parser.parse()

    console.print(f"✅ Parsed {len(diff_files)} files")

    results = {
        'files': [],
        'call_graphs': [],
        'impact_analysis': {},
        'analysis_mode': 'basic',
        'timestamp': time.time()
    }

    # Process each file
    for file in diff_files:
        file_info = {
            'path': file.path,
            'language': file.language,
            'changes': len(file.changes),
            'change_details': []
        }

        for change in file.changes:
            file_info['change_details'].append({
                'old_start': change.old_start,
                'old_end': change.old_end,
                'new_start': change.new_start,
                'new_end': change.new_end,
                'content': change.content
            })

        results['files'].append(file_info)

    # Generate call graphs if requested
    if call_graph:
        with console.status("[bold green]📊 Generating call graphs...[/bold green]") as status:
            call_graphs = parser.generate_call_graphs(diff_files)

        console.print(f"📊 Generated {len(call_graphs)} call graphs")

        for cg in call_graphs:
            cg_info = {
                'file': list(cg.files)[0] if cg.files else None,
                'nodes': len(cg.nodes),
                'edges': len(cg.graph.edges),
                'languages': list(cg.languages),
                'functions': [],
                'relationships': []
            }

            # Extract function information
            for node_id, node in cg.nodes.items():
                if node.type == 'function':
                    cg_info['functions'].append({
                        'name': node.name,
                        'file': node.file,
                        'start_line': node.start_line,
                        'end_line': node.end_line
                    })

            # Extract relationships
            for source, target, data in cg.graph.edges(data=True):
                source_node = cg.nodes[source]
                target_node = cg.nodes[target]
                cg_info['relationships'].append({
                    'source': source_node.name,
                    'target': target_node.name,
                    'type': data['type'],
                    'line': data['line_number']
                })

            results['call_graphs'].append(cg_info)

    # Perform basic impact analysis if requested
    if impact_analysis and call_graph and results['call_graphs']:
        console.print("🎯 Performing basic impact analysis...")
        analyzer = CallGraphAnalyzer()

        # Identify changed functions (simplified)
        changed_functions = set()
        for file in diff_files:
            for change in file.changes:
                changed_functions.add(f"{file.path}:function")

        if changed_functions:
            impact_info = {
                'changed_functions': list(changed_functions),
                'note': 'Basic impact analysis - for advanced analysis use --advanced flag'
            }
            results['impact_analysis'] = impact_info

    # Output results
    _output_results(results, format, output, diff_files)
    return results

def _analyze_impact_advanced(diff_file, output, format, include_usage, include_dependencies,
                             analysis_depth, confidence_threshold, cache, cache_size, root_dir):
    """Advanced analysis mode using the comprehensive impact analysis system."""
    if format != 'json':
        console.print("🚀 [bold]Advanced Impact Analysis Mode[/bold]")
        console.print("🔬 Using comprehensive analysis system...")

    # Read the diff content
    with open(diff_file, 'r') as f:
        diff_content = f.read()

    # Determine file paths from diff and root directory
    file_paths = _extract_file_paths_from_diff(diff_content)

    if root_dir:
        # Convert relative paths to absolute paths based on root_dir
        import os
        file_paths = [os.path.join(root_dir, fp) if not os.path.isabs(fp) else fp for fp in file_paths]

    # Normalize file paths for semantic parsing (remove ./ prefix to match diff format)
    normalized_paths = []
    for fp in file_paths:
        # Remove leading ./ to match the format in diffs
        if fp.startswith('./'):
            normalized_paths.append(fp[2:])
        else:
            normalized_paths.append(fp)

    # Filter to only Python files that exist (use original paths for file existence checks)
    python_files = [fp for fp in file_paths if fp.endswith('.py') and (not root_dir or os.path.exists(fp))]
    python_files_normalized = [normalized_paths[i] for i, fp in enumerate(file_paths) if fp.endswith('.py') and (not root_dir or os.path.exists(fp))]

    if not python_files:
        if format != 'json':
            console.print("[yellow]⚠️  No Python files found in diff or they don't exist at specified paths[/yellow]")
            console.print("💡 Try using --root-dir to specify the project root directory")
        return {}

    if format != 'json':
        console.print(f"📁 Analyzing {len(python_files)} Python files")

    # Create analysis configuration
    config = AnalysisConfiguration(
        include_usage_analysis=include_usage,
        include_dependency_tracking=include_dependencies,
        analysis_depth=analysis_depth,
        confidence_threshold=confidence_threshold,
        enable_caching=cache
    )

    # Initialize advanced analyzer
    analyzer = AdvancedImpactAnalyzer(config)

    # Perform comprehensive analysis
    if format == 'json':
        # Run without progress display for clean JSON output
        try:
            result = analyzer.analyze_impact(
                diff_content=diff_content,
                file_paths=python_files_normalized,  # Use normalized paths for semantic parsing
                root_directory=root_dir or ""
            )
        except Exception as e:
            raise
    else:
        # Run with progress display for other formats
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            task = progress.add_task("🔬 Running comprehensive impact analysis...", total=None)

            try:
                result = analyzer.analyze_impact(
                    diff_content=diff_content,
                    file_paths=python_files_normalized,  # Use normalized paths for semantic parsing
                    root_directory=root_dir or ""
                )

                progress.update(task, completed=True, description="✅ Analysis complete!")

            except Exception as e:
                progress.update(task, description=f"❌ Analysis failed: {str(e)}")
                raise

    # Display progress information
    if result.progress and format != 'json':
        elapsed = result.progress.elapsed_time
        phases = len(result.progress.completed_phases)
        console.print(".2f")

    # Prepare results for output
    results = {
        'analysis_mode': 'advanced',
        'timestamp': time.time(),
        'configuration': {
            'include_usage': include_usage,
            'include_dependencies': include_dependencies,
            'analysis_depth': analysis_depth,
            'confidence_threshold': confidence_threshold
        },
        'summary': result.analysis_summary if result.analysis_summary else {},
        'performance': result.performance_metrics if result.performance_metrics else {},
        'semantic_changes': len(result.semantic_changes) if result.semantic_changes else 0,
        'change_classifications': len(result.change_classifications) if result.change_classifications else 0,
# REMOVED: impact_propagation section - Pure heuristic analysis removed
        'import_graph': {
            'nodes': len(result.import_graph.nodes) if result.import_graph else 0,
            'edges': len(result.import_graph.edges) if result.import_graph else 0
        } if result.import_graph else {},
        'usage_analysis': {
            'total_usages': len(result.usage_analysis.results) if result.usage_analysis else 0
        } if result.usage_analysis else {},
        'dependency_analysis': result.dependency_analysis if result.dependency_analysis else {}
    }

    # Output results
    if format == 'json':
        _output_pure_facts_json(result, output)
    else:
        _display_advanced_text_results(result)
    return result

# Simplified helper functions for clean JSON extraction

def _get_imported_symbols_from_graph(import_graph, module_name, file_path):
    """Extract imported symbols for a specific module from the import graph."""
    if not import_graph or not hasattr(import_graph, 'edges'):
        return []

    # Look for edges where this module is imported in the specified file
    for edge in import_graph.edges:
        if (getattr(edge, 'from_file', '') == file_path and
            getattr(edge, 'to_file', '').replace('.py', '').replace('/', '.') == module_name):
            return getattr(edge, 'imported_symbols', [])

    return []

def _get_call_chains_from_graph(result):
    """Extract call chains from dependency analysis."""
    call_dependencies = []

    # If we have call graph data, extract basic call relationships
    if hasattr(result, 'call_graph') and result.call_graph:
        for edge in getattr(result.call_graph, 'edges', []):
            call_dependencies.append({
                "caller": getattr(edge, 'source', ''),
                "callee": getattr(edge, 'target', ''),
                "call_chain": [getattr(edge, 'source', ''), getattr(edge, 'target', '')],
                "chain_depth": 1
            })

    return call_dependencies[:10]  # Limit for safety

def _get_class_instantiations_from_usage(usage_analysis):
    """Extract class instantiation data from usage analysis."""
    if not usage_analysis or not hasattr(usage_analysis, 'results'):
        return []

    instantiations = []
    for symbol_name, usage_result in usage_analysis.results.items():
        if hasattr(usage_result, 'usages'):
            for usage in usage_result.usages:
                # Look for class definitions (they appear as 'variable' type in usage analysis)
                usage_type = getattr(usage, 'usage_type', '')
                context = getattr(usage, 'context', '')

                # Check if this is a class definition by looking for 'class ' in context
                if usage_type == 'variable' and 'class ' in context and symbol_name in context:
                    instantiations.append({
                        "class_name": symbol_name,
                        "file": getattr(usage, 'file_path', ''),
                        "line_number": getattr(usage, 'line_number', 0),
                        "assigned_to_variable": "",  # Not applicable for class definitions
                        "constructor_called": False,  # This is class definition, not instantiation
                        "context": _safe_json_string(context, 150)
                    })

    return instantiations[:10]  # Limit for safety

def _get_attribute_access_from_usage(usage_analysis):
    """Extract attribute access data from usage analysis."""
    if not usage_analysis or not hasattr(usage_analysis, 'results'):
        return []

    access_patterns = []
    for symbol_name, usage_result in usage_analysis.results.items():
        if hasattr(usage_result, 'usages'):
            for usage in usage_result.usages:
                usage_type = getattr(usage, 'usage_type', '')
                context = getattr(usage, 'context', '')

                # Look for method calls or attribute access patterns
                if usage_type == 'call' and ('self.' in context or '.' in symbol_name):
                    access_patterns.append({
                        "object_name": "self" if 'self.' in context else "unknown",
                        "attribute_name": symbol_name,
                        "file": getattr(usage, 'file_path', ''),
                        "line_number": getattr(usage, 'line_number', 0),
                        "access_type": "method_call",
                        "context": _safe_json_string(context, 150)
                    })

    return access_patterns[:15]  # Limit for safety

def _get_inheritance_relationships_from_mappings(element_mappings):
    """Extract inheritance relationships from element mappings."""
    if not element_mappings:
        return []

    relationships = []
    for mapping in element_mappings:
        if hasattr(mapping, 'mapped_elements'):
            for elem in mapping.mapped_elements:
                if getattr(elem, 'element_type', '') == 'class':
                    # Look for parent class information in AST node if available
                    parent_class = None
                    if hasattr(elem, 'ast_node') and elem.ast_node:
                        # Try to extract parent class from AST (this is basic implementation)
                        parent_class = getattr(elem, 'parent_class', None)

                    if parent_class:
                        relationships.append({
                            "parent_class": parent_class,
                            "child_class": getattr(elem, 'name', ''),
                            "file": getattr(elem, 'file_path', ''),
                            "line_number": getattr(elem, 'start_line', 0),
                            "methods_inherited": []  # Would need deeper AST analysis
                        })

    return relationships[:10]  # Limit for safety

def _get_parent_class_relationships(element_mappings):
    """Extract parent-child class relationships from element mappings."""
    relationships = {}

    if element_mappings:
        for mapping in element_mappings:
            if hasattr(mapping, 'mapped_elements'):
                for elem in mapping.mapped_elements:
                    if getattr(elem, 'element_type', '') == 'class':
                        class_name = getattr(elem, 'name', '')
                        file_path = getattr(elem, 'file_path', '')

                        # Basic parent detection (would need AST analysis for accuracy)
                        # For now, we'll use a simple heuristic or leave as None
                        relationships[class_name] = None

    return relationships

def _safe_json_string(text, max_length=500):
    """Safely prepare a string for JSON serialization."""
    if not isinstance(text, str):
        return ""

    # Remove null bytes that can break JSON
    text = text.replace('\x00', '')

    # Normalize excessive whitespace
    import re
    text = re.sub(r'\s+', ' ', text.strip())

    # Limit length to prevent bloated JSON
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return text

def _extract_parameters_safely(signature, other_signature):
    """Safely extract parameters that were added/removed between signatures."""
    if not signature or not other_signature:
        return []

    try:
        import re

        # Extract parameter list from function signature
        params_match = re.search(r'def\s+\w+\s*\((.*?)\)', signature.strip())
        other_params_match = re.search(r'def\s+\w+\s*\((.*?)\)', other_signature.strip())

        if not params_match:
            return []

        current_params = params_match.group(1).strip()
        other_params = other_params_match.group(1).strip() if other_params_match else ""

        # Parse parameter names (basic implementation)
        current_param_names = _parse_parameter_names(current_params)
        other_param_names = _parse_parameter_names(other_params)

        # Find differences
        added = [p for p in current_param_names if p not in other_param_names]
        return added

    except Exception:
        # If parsing fails, return empty list (safe fallback)
        return []

def _parse_parameter_names(params_str):
    """Parse parameter names from a parameter string."""
    if not params_str or params_str.strip() == "":
        return []

    try:
        import re

        # Split by comma and extract parameter names
        param_list = []
        for param in params_str.split(','):
            param = param.strip()
            if param:
                # Remove type hints and default values
                param_name = re.split(r'[=:]', param)[0].strip()
                if param_name and not param_name.startswith('*'):
                    param_list.append(param_name)

        return param_list

    except Exception:
        return []

def _output_pure_facts_json(result: ComprehensiveAnalysisResult, output):
    """Output comprehensive factual data in LLM-optimized JSON format following output_format.json structure."""
    import json
    import time

    # Build comprehensive JSON structure based on output_format.json
    facts_data = {
        "analysis_metadata": {
            "tool_version": "2.0",
            "timestamp": int(time.time()),
            "processing_duration_ms": result.performance_metrics.get('total_analysis_time', 0) * 1000 if result.performance_metrics else 0,
            "files_analyzed": len(set(getattr(c, 'file_path', '') for c in (result.semantic_changes or []) if getattr(c, 'file_path', ''))),
            "lines_analyzed": result.performance_metrics.get('total_lines_processed', 0) if result.performance_metrics else 0
        },

        "code_changes": {
            "functions_modified": [
                {
                    "name": getattr(change, 'element_name', ''),
                    "file": getattr(change, 'file_path', ''),
                    "line_start": getattr(change, 'line_number', 0),
                    "line_end": getattr(change, 'line_number', 0) + 5,  # Estimate end line
                    "old_signature": _safe_json_string(getattr(change, 'old_content', '')),
                    "new_signature": _safe_json_string(getattr(change, 'new_content', '')),
                    "parameters_added": _extract_parameters_safely(
                        _safe_json_string(getattr(change, 'new_content', '')),
                        _safe_json_string(getattr(change, 'old_content', ''))
                    ),
                    "parameters_removed": _extract_parameters_safely(
                        _safe_json_string(getattr(change, 'old_content', '')),
                        _safe_json_string(getattr(change, 'new_content', ''))
                    ),
                    "return_type_changed": False,
                    "change_type": getattr(change.change_type, 'name', 'unknown').lower() if hasattr(change, 'change_type') and hasattr(change.change_type, 'name') else 'unknown'
                }
                for change in (result.semantic_changes or [])
                if getattr(change, 'change_type', '') and hasattr(change.change_type, 'name') and
                   change.change_type.name in ('FUNCTION_SIGNATURE', 'FUNCTION_LOGIC')
            ][:20],  # Limit for safety

            "classes_modified": [
                {
                    "name": getattr(change, 'element_name', ''),
                    "file": getattr(change, 'file_path', ''),
                    "line_start": getattr(change, 'line_number', 0),
                    "line_end": getattr(change, 'line_number', 0) + 20,  # Estimate end line
                    "methods_added": [],
                    "methods_removed": [],
                    "methods_modified": [],
                    "attributes_added": [],
                    "change_type": getattr(change.change_type, 'name', 'unknown').lower() if hasattr(change, 'change_type') and hasattr(change.change_type, 'name') else 'unknown'
                }
                for change in (result.semantic_changes or [])
                if getattr(change, 'change_type', '') and hasattr(change.change_type, 'name') and
                   change.change_type.name in ('CLASS_DEFINITION', 'CLASS_MODIFICATION')
            ][:10],  # Limit for safety

            "imports_modified": {
                "imports_added": [
                    {
                        "module": getattr(change, 'element_name', ''),
                        "imported_names": _get_imported_symbols_from_graph(result.import_graph, getattr(change, 'element_name', ''), getattr(change, 'file_path', '')),
                        "import_type": "from_import" if "." in getattr(change, 'element_name', '') else "absolute_import",
                        "file": getattr(change, 'file_path', ''),
                        "line_number": getattr(change, 'line_number', 0)
                    }
                    for change in (result.semantic_changes or [])
                    if getattr(change, 'change_type', '') and hasattr(change.change_type, 'name') and
                       'IMPORT' in change.change_type.name and 'ADDED' in change.change_type.name
                ][:10],

                "imports_removed": [
                    {
                        "module": getattr(change, 'element_name', ''),
                        "imported_names": _get_imported_symbols_from_graph(result.import_graph, getattr(change, 'element_name', ''), getattr(change, 'file_path', '')),
                        "file": getattr(change, 'file_path', ''),
                        "line_number": getattr(change, 'line_number', 0)
                    }
                    for change in (result.semantic_changes or [])
                    if getattr(change, 'change_type', '') and hasattr(change.change_type, 'name') and
                       'IMPORT' in change.change_type.name and 'REMOVED' in change.change_type.name
                ][:10]
            },

            "variables_modified": [
                {
                    "name": getattr(change, 'element_name', ''),
                    "file": getattr(change, 'file_path', ''),
                    "old_value": _safe_json_string(getattr(change, 'old_content', ''), 100),
                    "new_value": _safe_json_string(getattr(change, 'new_content', ''), 100),
                    "line_number": getattr(change, 'line_number', 0),
                    "change_type": "value_change"
                }
                for change in (result.semantic_changes or [])
                if getattr(change, 'change_type', '') and hasattr(change.change_type, 'name') and
                   'VARIABLE' in change.change_type.name
            ][:10],

            "files_affected": {
                "modified": list(set(getattr(c, 'file_path', '') for c in (result.semantic_changes or []) if getattr(c, 'file_path', ''))),
                "added": [],
                "removed": []
            }
        },

        "usage_analysis": {
            "function_calls": [
                {
                    "caller_function": getattr(usage, 'function_name', 'unknown'),
                    "caller_file": getattr(usage, 'file_path', ''),
                    "callee_function": getattr(usage, 'function_name', 'unknown'),
                    "callee_file": getattr(usage, 'file_path', ''),
                    "line_number": getattr(usage, 'line_number', 0),
                    "call_type": getattr(usage, 'usage_type', 'direct_call'),
                    "parameters_used": [],
                    "in_context": _safe_json_string(getattr(usage, 'context', ''), 200)
                }
                for usage in (result.usage_analysis.results.values() if result.usage_analysis and hasattr(result.usage_analysis, 'results') else [])
                for usage_item in (getattr(usage, 'usages', []) if hasattr(usage, 'usages') else [])
            ][:30],  # Limit for safety

            "class_instantiations": _get_class_instantiations_from_usage(result.usage_analysis),
            "attribute_access": _get_attribute_access_from_usage(result.usage_analysis),
            "inheritance_relationships": _get_inheritance_relationships_from_mappings(result.element_mappings),

            "usage_summary": {
                func_name: {
                    "total_calls": getattr(usage_result, 'total_usages', 0),
                    "files_using": list(set(getattr(u, 'file_path', '') for u in getattr(usage_result, 'usages', []))),
                    "most_used_in": getattr(usage_result, 'most_used_in', ''),
                    "usage_patterns": ["direct_call"]
                }
                for func_name, usage_result in (result.usage_analysis.results.items() if result.usage_analysis and hasattr(result.usage_analysis, 'results') else [])
            }
        },

        "dependency_analysis": {
            "import_dependencies": [
                {
                    "from_file": getattr(edge, 'from_file', ''),
                    "to_module": getattr(edge, 'to_file', '').replace('.py', '').replace('/', '.'),
                    "import_type": getattr(edge, 'import_type', 'absolute_import'),
                    "imported_symbols": getattr(edge, 'imported_symbols', [])
                }
                for edge in (result.import_graph.edges if result.import_graph else [])
            ][:20],  # Limit for safety

            "call_dependencies": _get_call_chains_from_graph(result),

            "module_dependencies": {
                module_path: {
                    "depends_on": [getattr(edge, 'to_file', '') for edge in getattr(node, 'imports', [])],
                    "depended_by": [getattr(edge, 'from_file', '') for edge in getattr(node, 'imported_by', [])],
                    "dependency_depth": 1
                }
                for module_path, node in (result.import_graph.nodes.items() if result.import_graph else [])
            },

            "circular_dependencies": getattr(result.import_graph, 'cycles', []) if result.import_graph else [],

            "impact_scope": {
                "directly_affected_files": list(set(getattr(c, 'file_path', '') for c in (result.semantic_changes or []) if getattr(c, 'file_path', ''))),
                "indirectly_affected_files": [],  # Can be calculated from dependencies
                "affected_symbols": list(set(getattr(c, 'element_name', '') for c in (result.semantic_changes or []) if getattr(c, 'element_name', ''))),
                "propagation_paths": []  # Can be calculated from dependency analysis
            }
        },

        "code_structure": {
            "element_mappings": [
                {
                    "element_type": getattr(elem, 'element_type', 'unknown'),
                    "name": getattr(elem, 'name', ''),
                    "file": getattr(elem, 'file_path', ''),
                    "line_start": getattr(elem, 'start_line', 0),
                    "line_end": getattr(elem, 'end_line', 0),
                    "parent_element": _get_parent_class_relationships(result.element_mappings).get(getattr(elem, 'name', ''), None),
                    "visibility": "public"
                }
                for mapping in (result.element_mappings or [])
                for elem in (getattr(mapping, 'mapped_elements', []) if hasattr(mapping, 'mapped_elements') else [])
            ][:50],  # Limit for safety

            "file_structure": [
                {
                    "file": file_path,
                    "functions_count": sum(1 for m in (result.element_mappings or [])
                                         for e in getattr(m, 'mapped_elements', [])
                                         if getattr(e, 'file_path', '') == file_path
                                         and getattr(e, 'element_type', '') == 'function'),
                    "classes_count": sum(1 for m in (result.element_mappings or [])
                                       for e in getattr(m, 'mapped_elements', [])
                                       if getattr(e, 'file_path', '') == file_path
                                       and getattr(e, 'element_type', '') == 'class'),
                    "imports_count": len([edge for edge in (result.import_graph.edges if result.import_graph else [])
                                        if getattr(edge, 'from_file', '') == file_path]),
                    "total_lines": 0
                }
                for file_path in set(getattr(e, 'file_path', '') for m in (result.element_mappings or [])
                                   for e in getattr(m, 'mapped_elements', []) if getattr(e, 'file_path', ''))
            ][:10]
        },

        "change_classifications": [
            {
                "category": str(getattr(classification, 'category', '')).split('.')[-1],
                "affected_elements": getattr(classification, 'affected_elements', []),
                "change_scope": "local"
            }
            for classification in (result.change_classifications or [])
        ],

        "quantitative_metrics": {
            "change_counts": {
                "functions_modified": len([c for c in (result.semantic_changes or []) if 'function' in str(getattr(c, 'change_type', ''))]),
                "classes_modified": len([c for c in (result.semantic_changes or []) if 'class' in str(getattr(c, 'change_type', ''))]),
                "imports_added": len([c for c in (result.semantic_changes or []) if 'import' in str(getattr(c, 'change_type', '')) and 'added' in str(getattr(c, 'change_type', ''))]),
                "imports_removed": len([c for c in (result.semantic_changes or []) if 'import' in str(getattr(c, 'change_type', '')) and 'removed' in str(getattr(c, 'change_type', ''))]),
                "variables_modified": len([c for c in (result.semantic_changes or []) if 'variable' in str(getattr(c, 'change_type', ''))]),
                "files_affected": len(set(getattr(c, 'file_path', '') for c in (result.semantic_changes or []) if getattr(c, 'file_path', '')))
            },

            "usage_counts": {
                "total_function_calls": sum(len(getattr(usage, 'usages', [])) for usage in (result.usage_analysis.results.values() if result.usage_analysis and hasattr(result.usage_analysis, 'results') else [])),
                "total_class_instantiations": len(_get_class_instantiations_from_usage(result.usage_analysis)),
                "total_attribute_accesses": len(_get_attribute_access_from_usage(result.usage_analysis)),
                "unique_symbols_used": len(result.usage_analysis.results) if result.usage_analysis and hasattr(result.usage_analysis, 'results') else 0
            },

            "dependency_counts": {
                "total_import_relationships": len(result.import_graph.edges) if result.import_graph else 0,
                "total_call_dependencies": len(_get_call_chains_from_graph(result)),
                "max_dependency_depth": 1,    # Can be calculated from dependency analysis
                "circular_dependencies": len(getattr(result.import_graph, 'cycles', [])) if result.import_graph else 0
            },

            "code_metrics": {
                "total_files": len(set(getattr(e, 'file_path', '') for m in (result.element_mappings or [])
                                     for e in getattr(m, 'mapped_elements', []) if getattr(e, 'file_path', ''))),
                "total_functions": sum(1 for m in (result.element_mappings or [])
                                     for e in getattr(m, 'mapped_elements', [])
                                     if getattr(e, 'element_type', '') == 'function'),
                "total_classes": sum(1 for m in (result.element_mappings or [])
                                   for e in getattr(m, 'mapped_elements', [])
                                   if getattr(e, 'element_type', '') == 'class'),
                "total_lines_of_code": result.performance_metrics.get('total_lines_processed', 0) if result.performance_metrics else 0
            }
        }
    }

    # Clean JSON output with proper error handling
    try:
        json_output = json.dumps(facts_data, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        # Fallback to basic JSON if there are serialization issues
        facts_data = {
            "error": "JSON serialization issue",
            "basic_info": {
                "total_changes": len(result.semantic_changes) if result.semantic_changes else 0,
                "files_affected": len(set(getattr(c, 'file_path', '') for c in (result.semantic_changes or []) if getattr(c, 'file_path', '')))
            }
        }
        json_output = json.dumps(facts_data, indent=2)

    if output:
        Path(output).write_text(json_output)
        if format != 'json':  # Don't show console output when format is JSON
            console.print(f"📄 Pure factual analysis saved to {output}")
    else:
        console.print(json_output)

def _extract_file_paths_from_diff(diff_content: str) -> list:
    """Extract file paths from a diff content."""
    import re
    file_paths = []

    # Match +++ b/path/to/file patterns
    for line in diff_content.split('\n'):
        if line.startswith('+++ b/'):
            path = line[6:]  # Remove '+++ b/' prefix
            if path and not path.startswith('/dev/null'):
                file_paths.append(path)

    return file_paths

def _output_results(results, format, output, diff_files):
    """Output results in the specified format."""
    if format == 'json':
        output_data = json.dumps(results, indent=2, default=str)
        if output:
            Path(output).write_text(output_data)
            console.print(f"📄 Results saved to {output}")
        else:
            console.print(output_data)
    elif format == 'html':
        # Generate HTML report
        html_content = _generate_html_report(results)
        if output:
            Path(output).write_text(html_content)
            console.print(f"📄 HTML report saved to {output}")
        else:
            console.print("[yellow]⚠️  HTML output requires --output parameter[/yellow]")
    else:
        # Text format output
        _display_text_results(results, diff_files)

def _output_advanced_results(result: ComprehensiveAnalysisResult, format, output):
    """Output advanced analysis results in the specified format."""
    if format == 'json':
        # Convert to JSON-serializable format
        output_data = {
            'analysis_mode': 'advanced',
            'timestamp': time.time(),
            'summary': result.analysis_summary,
            'performance': result.performance_metrics,
            'semantic_changes_count': len(result.semantic_changes),
            'change_classifications_count': len(result.change_classifications)
            # REMOVED: impact_propagation section - Pure heuristic analysis removed
        }

        json_output = json.dumps(output_data, indent=2, default=str)
        if output:
            Path(output).write_text(json_output)
            console.print(f"📄 Advanced analysis results saved to {output}")
        else:
            console.print(json_output)

    elif format == 'html':
        html_content = _generate_advanced_html_report(result)
        if output:
            Path(output).write_text(html_content)
            console.print(f"📄 Advanced HTML report saved to {output}")
        else:
            console.print("[yellow]⚠️  HTML output requires --output parameter[/yellow]")

    else:
        # Text format for advanced results
        _display_advanced_text_results(result)

def _display_text_results(results, diff_files):
    """Display results in text format using Rich."""
    # File analysis summary
    console.print(f"\n📁 [bold]File Analysis Summary[/bold]")
    table = Table()
    table.add_column("File", style="cyan")
    table.add_column("Language", style="magenta")
    table.add_column("Changes", style="yellow")

    for file_info in results['files']:
        table.add_row(file_info['path'], file_info['language'], str(file_info['changes']))

    console.print(table)

    # Call graph summary
    if results['call_graphs']:
        console.print(f"\n🔗 [bold]Call Graph Analysis[/bold]")
        for cg_info in results['call_graphs']:
            console.print(f"\n[bold]{cg_info['file']}[/bold]:")
            console.print(f"  • {cg_info['nodes']} nodes, {cg_info['edges']} relationships")
            console.print(f"  • Languages: {', '.join(cg_info['languages'])}")

            if cg_info['functions']:
                console.print("  • Functions found:")
                for func in cg_info['functions'][:5]:  # Show first 5
                    console.print(f"    - {func['name']} (lines {func['start_line']}-{func['end_line']})")
                if len(cg_info['functions']) > 5:
                    console.print(f"    ... and {len(cg_info['functions']) - 5} more")

            if cg_info['relationships']:
                console.print("  • Call relationships:")
                for rel in cg_info['relationships'][:3]:  # Show first 3
                    console.print(f"    - {rel['source']} → {rel['target']} ({rel['type']})")
                if len(cg_info['relationships']) > 3:
                    console.print(f"    ... and {len(cg_info['relationships']) - 3} more")

    # Impact analysis
    if results.get('impact_analysis'):
        console.print(f"\n🎯 [bold]Impact Analysis[/bold]")
        impact = results['impact_analysis']
        console.print(f"  • Changed functions: {len(impact['changed_functions'])}")
        if 'note' in impact:
            console.print(f"  • Note: {impact['note']}")

    console.print(f"\n✅ Analysis complete!")

def _display_advanced_text_results(result: ComprehensiveAnalysisResult):
    """Display advanced analysis results in text format using Rich - Pure Facts Only."""
    console.print(f"\n🔬 [bold]Code Change Analysis - Pure Facts[/bold]")
    console.print("=" * 60)

    # Raw Metrics
    console.print(f"\n📊 [bold]Observed Facts[/bold]")

    semantic_count = len(result.semantic_changes) if result.semantic_changes else 0
    classification_count = len(result.change_classifications) if result.change_classifications else 0
    mapping_count = len(result.element_mappings) if result.element_mappings else 0

    console.print(f"• Semantic Changes Detected: {semantic_count}")
    console.print(f"• Code Elements Mapped: {mapping_count}")
    console.print(f"• Change Classifications: {classification_count}")

# REMOVED: Impact propagation display - Pure heuristic analysis removed

    # Performance Metrics
    perf = result.performance_metrics
    if perf:
        console.print(f"\n⚡ [bold]Analysis Performance[/bold]")
        console.print(".2f")
        console.print(f"• Analysis Phases: {perf.get('phases_completed', 0)}")

    # Code Structure Facts
    if result.import_graph:
        console.print(f"\n🔗 [bold]Code Structure Observed[/bold]")
        console.print(f"• Import Graph Nodes: {len(result.import_graph.nodes)}")
        console.print(f"• Import Relationships: {len(result.import_graph.edges)}")

    if result.usage_analysis:
        console.print(f"• Function Usage Locations: {len(result.usage_analysis.results) if hasattr(result.usage_analysis, 'results') else 0}")

    console.print(f"\n✅ [bold]Factual Analysis Complete - Ready for LLM Processing[/bold]")
    console.print(f"[dim]Use --format json for structured data suitable for LLM consumption[/dim]")

def _generate_html_report(results):
    """Generate HTML report for basic analysis results."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Conscious - Code Change Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; background: #e9ecef; padding: 10px; margin: 5px; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Conscious - Code Change Analysis Report</h1>
        <p>Analysis Mode: {results.get('analysis_mode', 'basic')}</p>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>📊 Summary</h2>
        <div class="metric">Files: {len(results.get('files', []))}</div>
        <div class="metric">Call Graphs: {len(results.get('call_graphs', []))}</div>
        <div class="metric">Analysis Time: {results.get('timestamp', 0) - time.time():.2f}s</div>
    </div>

    <div class="section">
        <h2>📁 File Analysis</h2>
        <table>
            <tr><th>File</th><th>Language</th><th>Changes</th></tr>
            {"".join(f"<tr><td>{f['path']}</td><td>{f['language']}</td><td>{f['changes']}</td></tr>" for f in results.get('files', []))}
        </table>
    </div>

    {f'''
    <div class="section">
        <h2>🔗 Call Graph Analysis</h2>
        {"".join(f"<div class='metric'>{cg['file']}: {cg['nodes']} nodes, {cg['edges']} edges</div>" for cg in results.get('call_graphs', []))}
    </div>
    ''' if results.get('call_graphs') else ''}
</body>
</html>"""
    return html

def _generate_advanced_html_report(result: ComprehensiveAnalysisResult):
    """Generate comprehensive HTML report for advanced analysis."""
    summary = result.analysis_summary or {}
    perf = result.performance_metrics or {}

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Conscious - Advanced Impact Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }}
        .section {{ margin: 30px 0; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric.large {{ font-size: 2em; font-weight: bold; }}
        .metric-label {{ font-size: 0.9em; opacity: 0.9; margin-bottom: 5px; }}
        .metric-value {{ font-size: 1.5em; }}
        .severity-critical {{ background: linear-gradient(135deg, #ff4757 0%, #ff3838 100%); }}
        .severity-major {{ background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%); }}
        .severity-minor {{ background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%); }}
        .severity-patch {{ background: linear-gradient(135deg, #26a69a 0%, #00897b 100%); }}
        .recommendations {{ background: #f8f9fa; padding: 20px; border-left: 4px solid #007bff; margin: 15px 0; }}
        .risk-high {{ background: #ffebee; border-left-color: #f44336; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #007bff; color: white; }}
        .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Conscious - Advanced Impact Analysis Report</h1>
            <p>Comprehensive Code Change Impact Assessment</p>
            <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Semantic Changes</div>
                    <div class="metric-value">{summary.get('total_semantic_changes', 0)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Classifications</div>
                    <div class="metric-value">{summary.get('total_change_classifications', 0)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Breaking Changes</div>
                    <div class="metric-value">{summary.get('breaking_changes', 0)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Analysis Time</div>
                    <div class="metric-value">{perf.get('total_analysis_time', 0):.2f}s</div>
                </div>
            </div>
        </div>

# REMOVED: Impact severity breakdown - Pure heuristic analysis removed

# REMOVED: Risk assessment section - Pure heuristic analysis removed

# REMOVED: Deployment recommendations section - Pure heuristic analysis removed

# REMOVED: Coordination needed section - Pure heuristic analysis removed

        <div class="section">
            <h2>📈 Analysis Details</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Semantic Changes Detected</td><td>{len(result.semantic_changes) if result.semantic_changes else 0}</td></tr>
                <tr><td>Change Classifications</td><td>{len(result.change_classifications) if result.change_classifications else 0}</td></tr>
                <tr><td>Import Graph Nodes</td><td>{len(result.import_graph.nodes) if result.import_graph else 0}</td></tr>
                <tr><td>Import Graph Edges</td><td>{len(result.import_graph.edges) if result.import_graph else 0}</td></tr>
                <tr><td>Usage Analysis Results</td><td>{len(result.usage_analysis.results) if result.usage_analysis else 0}</td></tr>
                <tr><td>Analysis Phases Completed</td><td>{len(result.progress.completed_phases) if result.progress else 0}</td></tr>
            </table>
        </div>

        <div class="footer">
            <p>Report generated by Conscious - Advanced Impact Analysis System</p>
            <p>For questions or support, please refer to the documentation.</p>
        </div>
    </div>
</body>
</html>"""
    return html

if __name__ == '__main__':
    cli()

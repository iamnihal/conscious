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
        console.print("[yellow]⚠️  No Python files found in diff or they don't exist at specified paths[/yellow]")
        console.print("💡 Try using --root-dir to specify the project root directory")
        return {}

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

    # Perform comprehensive analysis with progress tracking
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
    if result.progress:
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
        'impact_propagation': {
            'overall_severity': result.impact_propagation.overall_severity.value if result.impact_propagation else None,
            'overall_confidence': result.impact_propagation.overall_confidence.value if result.impact_propagation else None,
            'critical_changes': result.impact_propagation.critical_changes if result.impact_propagation else 0,
            'major_changes': result.impact_propagation.major_changes if result.impact_propagation else 0,
            'minor_changes': result.impact_propagation.minor_changes if result.impact_propagation else 0,
            'patch_changes': result.impact_propagation.patch_changes if result.impact_propagation else 0,
            'risk_assessment': result.impact_propagation.risk_assessment if result.impact_propagation else None,
            'deployment_recommendations': result.impact_propagation.deployment_recommendations if result.impact_propagation else [],
            'coordination_needed': result.impact_propagation.coordination_needed if result.impact_propagation else []
        } if result.impact_propagation else {},
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
    _output_advanced_results(result, format, output)
    return result

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
            'change_classifications_count': len(result.change_classifications),
            'impact_propagation': {
                'overall_severity': result.impact_propagation.overall_severity.value,
                'overall_confidence': result.impact_propagation.overall_confidence.value,
                'critical_changes': result.impact_propagation.critical_changes,
                'major_changes': result.impact_propagation.major_changes,
                'minor_changes': result.impact_propagation.minor_changes,
                'patch_changes': result.impact_propagation.patch_changes,
                'risk_assessment': result.impact_propagation.risk_assessment,
                'deployment_recommendations': result.impact_propagation.deployment_recommendations,
                'coordination_needed': result.impact_propagation.coordination_needed
            } if result.impact_propagation else None
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
    """Display advanced analysis results in text format using Rich."""
    console.print(f"\n🚀 [bold]Advanced Impact Analysis Results[/bold]")
    console.print("=" * 60)

    # Executive Summary
    console.print(f"\n📊 [bold]Executive Summary[/bold]")
    summary = result.analysis_summary
    if summary:
        console.print(f"• Total Semantic Changes: {summary.get('total_semantic_changes', 0)}")
        console.print(f"• Change Classifications: {summary.get('total_change_classifications', 0)}")
        console.print(f"• Breaking Changes: {summary.get('breaking_changes', 0)}")
        console.print(f"• Non-Breaking Changes: {summary.get('non_breaking_changes', 0)}")

        if 'overall_severity' in summary:
            console.print(f"• Overall Severity: [bold]{summary['overall_severity']}[/bold]")
            console.print(f"• Overall Confidence: [bold]{summary['overall_confidence']}[/bold]")

    # Performance Metrics
    perf = result.performance_metrics
    if perf:
        console.print(f"\n⚡ [bold]Performance Metrics[/bold]")
        console.print(".2f")
        console.print(f"• Phases Completed: {perf.get('phases_completed', 0)}")

    # Impact Analysis
    if result.impact_propagation:
        console.print(f"\n🎯 [bold]Impact Analysis[/bold]")
        prop = result.impact_propagation
        console.print(f"• Critical Changes: [red]{prop.critical_changes}[/red]")
        console.print(f"• Major Changes: [yellow]{prop.major_changes}[/yellow]")
        console.print(f"• Minor Changes: [blue]{prop.minor_changes}[/blue]")
        console.print(f"• Patch Changes: [green]{prop.patch_changes}[/green]")

        if prop.risk_assessment:
            console.print(f"\n🔍 [bold]Risk Assessment[/bold]")
            console.print(f"{prop.risk_assessment}")

        # Recommendations
        if prop.deployment_recommendations:
            console.print(f"\n🚀 [bold]Deployment Recommendations[/bold]")
            for rec in prop.deployment_recommendations[:5]:  # Show first 5
                console.print(f"• {rec}")
            if len(prop.deployment_recommendations) > 5:
                console.print(f"• ... and {len(prop.deployment_recommendations) - 5} more")

        if prop.coordination_needed:
            console.print(f"\n👥 [bold]Coordination Needed[/bold]")
            for coord in prop.coordination_needed[:3]:  # Show first 3
                console.print(f"• {coord}")
            if len(prop.coordination_needed) > 3:
                console.print(f"• ... and {len(prop.coordination_needed) - 3} more")

    # Graph Analysis
    if result.import_graph:
        console.print(f"\n🔗 [bold]Import Graph Analysis[/bold]")
        console.print(f"• Graph Nodes: {len(result.import_graph.nodes)}")
        console.print(f"• Graph Edges: {len(result.import_graph.edges)}")

    console.print(f"\n✅ [bold]Advanced Analysis Complete![/bold]")

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

        {f'''
        <div class="section">
            <h2>🎯 Impact Severity Breakdown</h2>
            <div class="metric-grid">
                <div class="metric severity-critical">
                    <div class="metric-label">Critical</div>
                    <div class="metric-value large">{result.impact_propagation.critical_changes if result.impact_propagation else 0}</div>
                </div>
                <div class="metric severity-major">
                    <div class="metric-label">Major</div>
                    <div class="metric-value large">{result.impact_propagation.major_changes if result.impact_propagation else 0}</div>
                </div>
                <div class="metric severity-minor">
                    <div class="metric-label">Minor</div>
                    <div class="metric-value large">{result.impact_propagation.minor_changes if result.impact_propagation else 0}</div>
                </div>
                <div class="metric severity-patch">
                    <div class="metric-label">Patch</div>
                    <div class="metric-value large">{result.impact_propagation.patch_changes if result.impact_propagation else 0}</div>
                </div>
            </div>
        </div>
        ''' if result.impact_propagation else ''}

        {f'''
        <div class="section">
            <h2>🔍 Risk Assessment</h2>
            <div class="recommendations">
                <h3>Overall Assessment</h3>
                <p>{result.impact_propagation.risk_assessment if result.impact_propagation else 'No risk assessment available'}</p>
            </div>
        </div>
        ''' if result.impact_propagation and result.impact_propagation.risk_assessment else ''}

        {f'''
        <div class="section">
            <h2>🚀 Deployment Recommendations</h2>
            {"".join(f"<div class='recommendations'><strong>•</strong> {rec}</div>" for rec in result.impact_propagation.deployment_recommendations[:10])}
            {f"<p>... and {len(result.impact_propagation.deployment_recommendations) - 10} more recommendations</p>" if result.impact_propagation and len(result.impact_propagation.deployment_recommendations) > 10 else ""}
        </div>
        ''' if result.impact_propagation and result.impact_propagation.deployment_recommendations else ''}

        {f'''
        <div class="section">
            <h2>👥 Coordination Required</h2>
            {"".join(f"<div class='recommendations'><strong>•</strong> {coord}</div>" for coord in result.impact_propagation.coordination_needed[:5])}
            {f"<p>... and {len(result.impact_propagation.coordination_needed) - 5} more coordination items</p>" if result.impact_propagation and len(result.impact_propagation.coordination_needed) > 5 else ""}
        </div>
        ''' if result.impact_propagation and result.impact_propagation.coordination_needed else ''}

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

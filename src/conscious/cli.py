"""Command-line interface for the conscious tool."""

import click
from rich.console import Console
from rich.table import Table
from rich.traceback import install
from pathlib import Path
import json

from .core import DiffParser
from .analyzers import CallGraphAnalyzer

# Set up rich error handling
install()
console = Console()

@click.group()
def cli():
    """Conscious - A syntax-aware code change analyzer."""
    pass

@cli.command()
@click.argument('diff_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for analysis results')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text', help='Output format')
@click.option('--call-graph/--no-call-graph', default=True, help='Generate call graph analysis')
@click.option('--impact-analysis', '-i', is_flag=True, help='Show impact analysis for changes')
@click.option('--cache/--no-cache', default=True, help='Enable caching for performance')
@click.option('--cache-size', type=int, default=100, help='Maximum cache size (number of items)')
def analyze_changes(diff_file, output, format, call_graph, impact_analysis, cache, cache_size):
    """Analyze changes in a diff file with call graph generation."""
    try:
        # Parse the diff
        console.print("🔍 Parsing diff...")
        parser = DiffParser(str(diff_file), cache_enabled=cache, cache_size=cache_size)
        diff_files = parser.parse()

        console.print(f"✅ Parsed {len(diff_files)} files")

        results = {
            'files': [],
            'call_graphs': [],
            'impact_analysis': {}
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
            console.print("📊 Generating call graphs...")
            call_graphs = parser.generate_call_graphs(diff_files)

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

        # Perform impact analysis if requested
        if impact_analysis and call_graph and call_graphs:
            console.print("🎯 Analyzing impact...")
            analyzer = CallGraphAnalyzer()

            # Identify changed functions (simplified - could be enhanced)
            changed_functions = set()
            for file in diff_files:
                for change in file.changes:
                    # This is a simplified approach - in practice you'd need more sophisticated
                    # logic to map changes to specific functions
                    changed_functions.add(f"{file.path}:function")  # Placeholder

            if changed_functions:
                # Use the first call graph for impact analysis
                cg = call_graphs[0]
                # Convert back to CallGraph object for analysis
                # This is simplified - in a real implementation you'd store the actual graph

                impact_info = {
                    'changed_functions': list(changed_functions),
                    'note': 'Impact analysis requires enhanced function-to-change mapping'
                }
                results['impact_analysis'] = impact_info

        # Output results
        if format == 'json':
            output_data = json.dumps(results, indent=2)
            if output:
                Path(output).write_text(output_data)
                console.print(f"📄 Results saved to {output}")
            else:
                console.print(output_data)
        else:
            # Text format output
            _display_text_results(results, diff_files)

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise click.Abort()

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

if __name__ == '__main__':
    cli()

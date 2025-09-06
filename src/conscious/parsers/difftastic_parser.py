"""Difftastic integration for syntax-aware diff parsing."""

import os
import json
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List

@dataclass
class DifftasticChange:
    """Represents a syntax-aware change from Difftastic."""
    old_start: int
    old_end: int
    new_start: int
    new_end: int
    syntax_type: str  # e.g., 'function', 'class', 'import'
    content: str

class DifftasticParser:
    """Wrapper for Difftastic diff tool."""
    
    def __init__(self):
        # Find difft executable
        try:
            result = subprocess.run(['which', 'difft'],
                                 capture_output=True,
                                 text=True,
                                 check=True)
            self.difft_path = result.stdout.strip()
            
            # Verify it works
            subprocess.run([self.difft_path, '--version'],
                         capture_output=True,
                         check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(
                "Difftastic (difft) not found. Please install it first: "
                "https://github.com/Wilfred/difftastic#installation"
            ) from exc
    
    def parse_diff(self, old_content: str, new_content: str, 
                  language: str) -> List[DifftasticChange]:
        """Parse differences between old and new content using Difftastic."""
        old_file = None
        new_file = None
        try:
            # Create temporary files for old and new content
            old_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False)
            new_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False)
            
            # Write content to temp files
            old_file.write(old_content)
            old_file.flush()
            new_file.write(new_content)
            new_file.flush()
            
            # Close files to ensure content is written
            old_file.close()
            new_file.close()
            
            # Run difft
            try:
                cmd = [self.difft_path, '--display=json', old_file.name, new_file.name]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    env={'DFT_UNSTABLE': 'yes'}  # Enable JSON output
                )
                
                # Parse JSON output
                return self._parse_difft_output(result.stdout)
            
            except subprocess.CalledProcessError as e:
                print(f"Difftastic error: {e.stderr}")
                return []
            except KeyboardInterrupt:
                print("\nDifftastic interrupted")
                raise
                
        except (OSError, ValueError) as e:
            print(f"Error running Difftastic: {str(e)}")
            return []
            
        finally:
            # Clean up temporary files
            if old_file:
                try:
                    os.unlink(old_file.name)
                except OSError:
                    pass
            if new_file:
                try:
                    os.unlink(new_file.name)
                except OSError:
                    pass
    
    def _parse_difft_output(self, output: str) -> List[DifftasticChange]:
        """Parse Difftastic JSON output into change objects."""
        try:
            if not output.strip():
                return []
                
            data = json.loads(output)
            changes = []
            
            # Try old format first
            if 'hunks' in data:
                for hunk in data.get('hunks', []):
                    for syntax_change in hunk.get('syntax_changes', []):
                        old_pos = syntax_change.get('old_pos', {})
                        new_pos = syntax_change.get('new_pos', {})
                        
                        # Extract content from changes
                        content = '\n'.join(
                            change.get('content', '')
                            for change in syntax_change.get('changes', [])
                            if change.get('content')
                        )
                        
                        if content:  # Only add changes with content
                            change = DifftasticChange(
                                old_start=old_pos.get('start', 0),
                                old_end=old_pos.get('end', 0),
                                new_start=new_pos.get('start', 0),
                                new_end=new_pos.get('end', 0),
                                syntax_type=syntax_change.get('type', 'unknown'),
                                content=content
                            )
                            changes.append(change)
            
            # Try new format
            elif 'chunks' in data:
                for chunk_group in data.get('chunks', []):
                    for chunk in chunk_group:
                        # Look for RHS (new) changes
                        if 'rhs' in chunk:
                            rhs = chunk['rhs']
                            line_number = rhs.get('line_number', 0)
                            content = ''.join(
                                c.get('content', '')
                                for c in rhs.get('changes', [])
                            )
                            
                            if content.strip():  # Only add non-empty changes
                                change = DifftasticChange(
                                    old_start=line_number,
                                    old_end=line_number + 1,
                                    new_start=line_number,
                                    new_end=line_number + 1,
                                    syntax_type='change',  # We'll enhance this later
                                    content=content
                                )
                                changes.append(change)
            
            return changes
            
        except json.JSONDecodeError:
            print(f"Failed to parse Difftastic output as JSON: {output[:100]}...")
            return []
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error parsing Difftastic output: {str(e)}")
            return []
    
    def _detect_syntax_type(self, node_type: str) -> str:
        """Convert Difftastic syntax node types to our types."""
        # Mapping of Difftastic types to our internal types
        type_map = {
            # Common types across languages
            'function': 'function',
            'method': 'function',
            'class': 'class',
            'module': 'module',
            'import': 'import',
            
            # Python specific
            'def': 'function',
            'class_def': 'class',
            'import_from': 'import',
            'decorator': 'decorator',
            
            # JavaScript/TypeScript specific
            'function_declaration': 'function',
            'class_declaration': 'class',
            'interface': 'interface',
            'import_declaration': 'import',
            
            # Java specific
            'method_declaration': 'function',
            'interface_declaration': 'interface',
            'package_declaration': 'module'
        }
        
        return type_map.get(node_type.lower(), 'unknown')

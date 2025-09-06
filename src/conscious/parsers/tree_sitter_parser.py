"""Tree-sitter based code parsing."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from tree_sitter import Tree  # Only need Tree for type hints
from tree_sitter_languages import get_parser, get_language  # For language and parser support

@dataclass
class Function:
    """Represents a function/method definition."""
    name: str
    start_line: int
    end_line: int
    parameters: List[str]
    return_type: Optional[str]
    is_method: bool = False
    class_name: Optional[str] = None

@dataclass
class Class:
    """Represents a class definition."""
    name: str
    start_line: int
    end_line: int
    bases: List[str]
    methods: List[Function]

@dataclass
class Import:
    """Represents an import statement."""
    module: str
    names: List[str]
    start_line: int
    is_from: bool = False
    alias: Optional[str] = None

@dataclass
class Call:
    """Represents a function/method call."""
    name: str
    start_line: int
    arguments: List[str]
    is_method: bool = False
    object_name: Optional[str] = None

class TreeSitterParser:
    """Parser using Tree-sitter for syntax-aware code analysis."""
    
    def __init__(self):
        """Initialize parser with supported languages."""
        # Initialize parsers and languages
        self.parsers: Dict[str, Any] = {}
        self.languages: Dict[str, Any] = {}

        # Initialize supported languages
        for lang in ['python', 'javascript', 'typescript', 'java']:
            try:
                # Get pre-configured parser and language
                self.languages[lang] = get_language(lang)
                self.parsers[lang] = get_parser(lang)
            except (ValueError, TypeError, RuntimeError) as e:
                print(f"Warning: Failed to initialize {lang} support: {e}")
                # Only remove a language if it was partially initialized
                if lang in self.languages:
                    del self.languages[lang]
                if lang in self.parsers:
                    del self.parsers[lang]
        
    def parse_file(self, content: str, language: str) -> Optional[Tree]:
        """Parse file content using appropriate language grammar."""
        if language not in self.parsers:
            raise ValueError(f"Language {language} not supported")
            
        return self.parsers[language].parse(bytes(content, 'utf8'))
        
    def get_functions(self, tree: Tree, language: str) -> List[Function]:
        """Extract function definitions from AST."""
        functions = []

        # Simple traversal to find function definitions
        def traverse(node):
            if node.type == 'function_definition':
                # Get function name
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = name_node.text.decode('utf8')

                    # Get parameters
                    params_node = node.child_by_field_name('parameters')
                    params = []
                    if params_node:
                        for param in params_node.named_children:
                            if param.type == 'identifier':
                                params.append(param.text.decode('utf8'))

                    # Check if it's a method
                    is_method = False
                    class_name = None
                    parent = node.parent
                    while parent:
                        if parent.type == 'class_definition':
                            is_method = True
                            class_name_node = parent.child_by_field_name('name')
                            if class_name_node:
                                class_name = class_name_node.text.decode('utf8')
                            break
                        parent = parent.parent

                    functions.append(Function(
                        name=name,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        parameters=params,
                        return_type=None,  # Simplified
                        is_method=is_method,
                        class_name=class_name
                    ))

            # Continue traversal
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return functions
        
    def get_imports(self, tree: Tree, language: str) -> List[Import]:
        """Extract import statements from AST."""
        imports = []

        def traverse(node):
            if node.type == 'import_statement':
                # Get module name
                name_node = node.child_by_field_name('name')
                if name_node:
                    module = name_node.text.decode('utf8')
                    imports.append(Import(
                        module=module,
                        names=[],
                        start_line=node.start_point[0] + 1
                    ))

            elif node.type == 'import_from_statement':
                # Get module name
                module_node = node.child_by_field_name('module_name')
                if module_node:
                    module = module_node.text.decode('utf8')

                    # Get imported names
                    names = []
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        if name_node.type == 'dotted_name':
                            for child in name_node.named_children:
                                if child.type == 'identifier':
                                    names.append(child.text.decode('utf8'))
                        elif name_node.type == 'identifier':
                            names.append(name_node.text.decode('utf8'))

                    imports.append(Import(
                        module=module,
                        names=names,
                        start_line=node.start_point[0] + 1,
                        is_from=True
                    ))

            # Continue traversal
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return imports
        
    def get_classes(self, tree: Tree, language: str) -> List[Class]:
        """Extract class definitions from AST."""
        classes = []

        def traverse(node):
            if node.type == 'class_definition':
                # Get class name
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = name_node.text.decode('utf8')

                    # Get base classes (simplified)
                    bases = []

                    # Get methods (simplified - just count them)
                    methods = []
                    for child in node.named_children:
                        if child.type == 'function_definition':
                            methods.append(Function(
                                name="method",
                                start_line=child.start_point[0] + 1,
                                end_line=child.end_point[0] + 1,
                                parameters=[],
                                return_type=None
                            ))

                    classes.append(Class(
                        name=name,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        bases=bases,
                        methods=methods
                    ))

            # Continue traversal
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return classes
        
    def get_calls(self, tree: Tree, language: str) -> List[Call]:
        """Extract function/method calls from AST."""
        calls = []

        def traverse(node):
            if node.type == 'call':
                # Get function name
                function_node = node.child_by_field_name('function')
                if function_node:
                    if function_node.type == 'identifier':
                        name = function_node.text.decode('utf8')
                        calls.append(Call(
                            name=name,
                            start_line=node.start_point[0] + 1,
                            arguments=[],
                            is_method=False
                        ))
                    elif function_node.type == 'attribute':
                        # Method call
                        object_node = function_node.child_by_field_name('object')
                        attr_node = function_node.child_by_field_name('attribute')

                        if object_node and attr_node:
                            object_name = object_node.text.decode('utf8')
                            method_name = attr_node.text.decode('utf8')
                            calls.append(Call(
                                name=method_name,
                                start_line=node.start_point[0] + 1,
                                arguments=[],
                                is_method=True,
                                object_name=object_name
                            ))

            # Continue traversal
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return calls
        
    def get_dependencies(self, tree: Tree, language: str) -> List[tuple[str, str, str]]:
        """Extract dependencies (imports, function calls, etc.) from AST.

        Returns:
            List of tuples (source, target, type) where type is one of:
            - 'import': Module import
            - 'call': Function/method call
            - 'inherit': Class inheritance
            - 'implement': Interface implementation
        """
        deps = []

        # Add import dependencies
        for imp in self.get_imports(tree, language):
            if imp.is_from:
                for name in imp.names:
                    deps.append((imp.module, name, 'import'))
            else:
                deps.append((imp.module, '', 'import'))

        # Add inheritance dependencies
        for cls in self.get_classes(tree, language):
            for base in cls.bases:
                deps.append((cls.name, base, 'inherit'))

        # Add call dependencies
        seen = set()  # Avoid duplicates
        for call in self.get_calls(tree, language):
            key = (call.name, call.object_name if call.is_method else None)
            if key not in seen:
                if call.is_method:
                    deps.append((call.object_name, call.name, 'call'))
                else:
                    deps.append(('', call.name, 'call'))
                seen.add(key)

        return deps

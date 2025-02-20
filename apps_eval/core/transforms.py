"""
Code transformation utilities for the APPS evaluation service.
"""
import re
from typing import Optional


def extract_function_from_class(code: str, function_name: str) -> str:
    """
    Extract a function from a LeetCode-style class solution.
    
    Args:
        code: The source code containing the class
        function_name: Name of the function to extract
        
    Returns:
        The extracted function code with class context removed
    """
    # Find the method definition
    pattern = rf"def\s+{function_name}\s*\([^)]*\)[^:]*:"
    match = re.search(pattern, code)
    if not match:
        return code  # Return original code if no match found
    
    # Get the indentation level of the method
    method_start = match.start()
    lines = code.split('\n')
    current_line = 0
    for i, line in enumerate(lines):
        current_line += len(line) + 1  # +1 for newline
        if current_line > method_start:
            method_line = i
            break
    else:
        return code
    
    # Get the method body
    method_lines = []
    base_indent = len(lines[method_line]) - len(lines[method_line].lstrip())
    method_lines.append(lines[method_line].lstrip())
    
    for line in lines[method_line + 1:]:
        if line.strip() == "" or len(line) - len(line.lstrip()) > base_indent:
            method_lines.append(line[base_indent:] if line.strip() else "")
        else:
            break
    
    # Remove self parameter if present
    first_line = method_lines[0]
    if "self" in first_line:
        first_line = re.sub(r'\(\s*self\s*,\s*', '(', first_line)
        first_line = re.sub(r'\(\s*self\s*\)', '()', first_line)
        method_lines[0] = first_line
    
    # Join the lines and remove self references
    code = '\n'.join(method_lines)
    
    # Replace self.method_name with method_name
    code = re.sub(r'self\.([a-zA-Z_][a-zA-Z0-9_]*)', r'\1', code)
    
    return code


def normalize_test_case_io(inputs: list, outputs: list) -> tuple[list, list]:
    """
    Normalize test case inputs and outputs for consistent handling.
    
    Args:
        inputs: Raw input values
        outputs: Raw output values
        
    Returns:
        Tuple of (normalized inputs, normalized outputs)
    """
    # Normalize inputs
    if inputs and isinstance(inputs[0], dict):
        inputs = [{int(k): v for k, v in inputs[0].items()}]
    
    # Normalize outputs
    if isinstance(outputs, bool):
        outputs = [outputs]  # Wrap booleans in a list
    elif isinstance(outputs, dict):
        outputs = {int(k): v for k, v in outputs.items()}
    elif isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        outputs = [{int(k): v for k, v in d.items()} for d in outputs]
    
    return inputs, outputs


def wrap_standard_input_code(code: str) -> str:
    """
    Wrap standard input code in a function for consistent execution.
    
    Args:
        code: The original code that uses standard input
        
    Returns:
        Code wrapped in a function that can be executed
    """
    code_lines = code.split('\n')
    indented_code = ['    ' + line if line.strip() else line for line in code_lines]
    return 'def code():\n' + '\n'.join(indented_code) 
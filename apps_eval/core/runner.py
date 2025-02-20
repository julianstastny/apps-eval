"""
Core code execution and evaluation functionality.
"""
import sys
import time
import signal
import logging
import traceback
from io import StringIO
from types import ModuleType
from typing import Any, List, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from apps_eval.core.models import (
    CodeSubmission,
    CodeType,
    EvaluationResult,
    ExecutionResult,
    TestCase,
    TestResult,
)
from apps_eval.core.transforms import (
    extract_function_from_class,
    normalize_test_case_io,
    wrap_standard_input_code,
)
from apps_eval.utils.safety import secure_execution_environment, SecurityError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class ExecutionContext:
    """Context for code execution including stdout capture and timing."""
    stdout: StringIO
    start_time: float
    end_time: Optional[float] = None
    
    @property
    def execution_time(self) -> float:
        """Get the execution time in seconds."""
        if self.end_time is None:
            self.end_time = time.time()
        return self.end_time - self.start_time
    
    @property
    def captured_output(self) -> str:
        """Get the captured stdout as a string."""
        return self.stdout.getvalue()


@contextmanager
def execution_context() -> ExecutionContext:
    """Create a context for code execution with stdout capture and timing."""
    original_stdout = sys.stdout
    stdout = StringIO()
    sys.stdout = stdout
    
    try:
        yield ExecutionContext(stdout=stdout, start_time=time.time())
    finally:
        sys.stdout = original_stdout


def normalize_string(s: str) -> str:
    """Normalize a string by removing whitespace and converting to lowercase."""
    return ' '.join(s.lower().split())


def normalize_output(output: Any) -> Any:
    """Normalize output for consistent comparison."""
    # Convert tuples to lists (ground truth sequences are not tuples)
    if isinstance(output, tuple):
        return list(output)
    
    # Convert numpy arrays to lists
    if isinstance(output, np.ndarray):
        return output.tolist()
    
    # Convert dictionaries with string keys to int keys
    if isinstance(output, dict):
        return {int(k): v for k, v in output.items()}
    
    # Convert list of dictionaries
    if isinstance(output, list) and output and isinstance(output[0], dict):
        return [{int(k): v for k, v in d.items()} for d in output]
    
    return output


def compare_outputs(actual: Any, expected: Any, rtol: float = 1e-3) -> bool:
    """
    Compare actual and expected outputs with support for various types.
    
    Args:
        actual: The actual output from the code
        expected: The expected output
        rtol: Relative tolerance for floating point comparisons (default: 1e-3)
    
    Returns:
        bool: True if outputs match within tolerance
    """
    # Normalize both outputs
    actual = normalize_output(actual)
    expected = normalize_output(expected)
    
    # Direct comparison first
    tmp_result = actual == expected
    
    # If expected is a list, try comparing with its first element
    if isinstance(expected, list) and expected:
        tmp_result = tmp_result or (actual == expected[0])
        
        # For nested lists, try numeric comparison on flattened arrays
        try:
            actual_array = np.array(actual, dtype=float)
            expected_array = np.array(expected[0] if isinstance(expected[0], list) else expected, dtype=float)
            if actual_array.shape == expected_array.shape:
                tmp_result = tmp_result or np.allclose(actual_array, expected_array, rtol=rtol)
        except (ValueError, TypeError):
            pass
    
    # If that fails, try string comparison for string outputs
    if not tmp_result and isinstance(actual, str) and isinstance(expected, str):
        actual_norm = normalize_string(actual)
        expected_norm = normalize_string(expected)
        tmp_result = actual_norm == expected_norm
        
        # If that fails, try line by line comparison
        if not tmp_result:
            actual_lines = [normalize_string(line) for line in actual.split('\n')]
            expected_lines = [normalize_string(line) for line in expected.split('\n')]
            actual_lines = [line for line in actual_lines if line]
            expected_lines = [line for line in expected_lines if line]
            tmp_result = actual_lines == expected_lines
    
    # If still no match, try numeric comparison
    if not tmp_result:
        try:
            # Try converting to float arrays
            actual_float = [float(e) for e in (actual if isinstance(actual, list) else [actual])]
            expected_float = [float(e) for e in (expected if isinstance(expected, list) else [expected])]
            if len(actual_float) == len(expected_float):
                tmp_result = np.allclose(actual_float, expected_float, rtol=rtol)
        except (ValueError, TypeError):
            pass
        
        # Try comparing as sets of strings
        if not tmp_result:
            try:
                actual_set = set(normalize_string(str(x)) for x in (actual if isinstance(actual, list) else [actual]))
                expected_set = set(normalize_string(str(x)) for x in (expected if isinstance(expected, list) else [expected]))
                tmp_result = actual_set == expected_set
            except (ValueError, TypeError):
                pass
    
    return tmp_result


class CodeRunner:
    """Executes and evaluates Python code submissions."""
    
    def __init__(
        self,
        max_memory_bytes: Optional[int] = None,
        default_timeout: float = 4.0,
    ):
        self.max_memory_bytes = max_memory_bytes
        self.default_timeout = default_timeout
    
    def compile_code(
        self,
        submission: CodeSubmission,
    ) -> Tuple[Optional[ModuleType], Optional[str]]:
        """
        Compile the submission code into a module namespace.
        
        Args:
            submission: The code submission to compile
            
        Returns:
            Tuple of (module namespace or None, error message or None)
        """
        try:
            # Create a new module namespace
            module = ModuleType("submission")
            
            # Add all common imports from testing_util.py
            exec(
                "import sys\n"
                "import time\n"
                "import itertools\n"
                "from itertools import accumulate, product, permutations, combinations\n"
                "import collections\n"
                "from collections import Counter, OrderedDict, deque, defaultdict, ChainMap\n"
                "from functools import lru_cache\n"
                "import math\n"
                "from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\n"
                "import fractions\n"
                "from typing import List, Tuple, Optional, Any\n"
                "import numpy as np\n"
                "import random\n"
                "import heapq\n"
                "from heapq import *\n",
                module.__dict__
            )
            
            # Extract function from class if needed
            code = submission.code
            if submission.code_type == CodeType.CALL_BASED and "class Solution" in code:
                if not submission.function_name:
                    raise ValueError("Function name required for call-based submission")
                code = extract_function_from_class(code, submission.function_name)
            
            # Wrap standard input code in a function if needed
            if submission.code_type == CodeType.STANDARD_INPUT:
                code = wrap_standard_input_code(code)
            
            # First compile to check for syntax errors
            compile(code, '<string>', 'exec')
            
            # Then execute to define the function
            exec(code, module.__dict__)
            
            # Verify the function was defined
            if submission.code_type == CodeType.STANDARD_INPUT:
                if 'code' not in module.__dict__ or not callable(module.__dict__['code']):
                    raise ValueError("Failed to define code() function")
            else:
                if submission.function_name not in module.__dict__ or not callable(module.__dict__[submission.function_name]):
                    raise ValueError(f"Failed to define {submission.function_name}() function")
            
            return module, None
            
        except Exception as e:
            error_msg = f"Compilation error: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            return None, error_msg
    
    def execute_test_case(
        self,
        module: ModuleType,
        submission: CodeSubmission,
        test_case: TestCase,
    ) -> TestResult:
        """
        Execute a single test case.
        
        Args:
            module: The module namespace containing the code
            submission: The code submission
            test_case: The test case to execute
            
        Returns:
            TestResult containing the execution results
        """
        # Set up timeout handler
        timeout = test_case.timeout_seconds or self.default_timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            with secure_execution_environment(max_memory_bytes=self.max_memory_bytes):
                with execution_context() as ctx:
                    if submission.code_type == CodeType.CALL_BASED:
                        if not submission.function_name:
                            raise ValueError("Function name required for call-based submission")
                        
                        # Normalize inputs and outputs
                        inputs, expected_outputs = normalize_test_case_io(
                            test_case.inputs,
                            test_case.expected_outputs
                        )
                        
                        func = getattr(module, submission.function_name)
                        actual_output = func(*inputs)
                        
                    else:  # STANDARD_INPUT
                        if isinstance(test_case.inputs, list):
                            input_str = "\n".join(str(x) for x in test_case.inputs)
                        else:
                            input_str = str(test_case.inputs)
                        
                        # Redirect stdin
                        sys.stdin = StringIO(input_str)
                        # Execute the code in the module namespace
                        exec("code()", module.__dict__)
                        actual_output = ctx.captured_output
                    
                    signal.alarm(0)  # Disable alarm
                    
                    # Compare outputs
                    passed = compare_outputs(actual_output, test_case.expected_outputs)
                    
                    return TestResult(
                        status=ExecutionResult.PASS if passed else ExecutionResult.FAIL,
                        actual_output=actual_output,
                        execution_time=ctx.execution_time,
                        memory_used=None,  # TODO: Implement memory tracking
                    )
                    
        except TimeoutError:
            return TestResult(
                status=ExecutionResult.TIMEOUT,
                error_message="Execution timed out",
                execution_time=timeout,
                memory_used=None,
            )
            
        except SecurityError as e:
            return TestResult(
                status=ExecutionResult.MEMORY_ERROR if "memory" in str(e).lower() else ExecutionResult.RUNTIME_ERROR,
                error_message=str(e),
                execution_time=time.time() - ctx.start_time if 'ctx' in locals() else 0,
                memory_used=None,
            )
            
        except Exception as e:
            error_msg = f"Runtime error: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            return TestResult(
                status=ExecutionResult.RUNTIME_ERROR,
                error_message=error_msg,
                execution_time=time.time() - ctx.start_time if 'ctx' in locals() else 0,
                memory_used=None,
            )
        finally:
            signal.alarm(0)  # Always disable alarm
    
    def evaluate(self, submission: CodeSubmission) -> EvaluationResult:
        """
        Evaluate a code submission against its test cases.
        
        Args:
            submission: The code submission to evaluate
            
        Returns:
            EvaluationResult containing all test results and statistics
        """
        start_time = time.time()
        
        # Compile the code
        module, compile_error = self.compile_code(submission)
        if compile_error:
            return EvaluationResult(
                test_results=[
                    TestResult(
                        status=ExecutionResult.COMPILATION_ERROR,
                        error_message=compile_error,
                        execution_time=0.0,
                    )
                ],
                passed_count=0,
                total_count=len(submission.test_cases),
                total_time=0.0,
            )
        
        # Execute each test case
        test_results: List[TestResult] = []
        for test_case in submission.test_cases:
            result = self.execute_test_case(module, submission, test_case)
            test_results.append(result)
        
        # Calculate statistics
        passed_count = sum(1 for r in test_results if r.status == ExecutionResult.PASS)
        total_time = time.time() - start_time
        
        return EvaluationResult(
            test_results=test_results,
            passed_count=passed_count,
            total_count=len(test_results),
            total_time=total_time,
            metadata={
                "code_type": submission.code_type,
                "function_name": submission.function_name,
            },
        )


def timeout_handler(signum, frame):
    """Signal handler for timeouts."""
    raise TimeoutError("Code execution timed out") 
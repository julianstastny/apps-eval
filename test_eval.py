"""
Test script for evaluating code submissions.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from apps_eval.core.models import CodeSubmission, CodeType, TestCase
from apps_eval.core.runner import CodeRunner

def load_test_data(problem_id: int = 0) -> tuple[str, List[str], Dict[str, Any]]:
    """Load test data from the test_data directory."""
    problem_dir = Path("test_data") / f"problem_{problem_id}"
    
    # Load solutions
    with open(problem_dir / "solutions.json") as f:
        solutions = json.load(f)
    
    # Load input/output
    with open(problem_dir / "input_output.json") as f:
        input_output = json.load(f)
    
    # Load question
    with open(problem_dir / "question.txt") as f:
        question = f.read()
    
    return question, solutions, input_output

def create_test_cases(inputs: List[str], outputs: List[str]) -> List[TestCase]:
    """Create test cases from inputs and outputs."""
    return [
        TestCase(
            inputs=input_str,
            expected_outputs=output_str,
            timeout_seconds=4.0
        )
        for input_str, output_str in zip(inputs, outputs)
    ]

def main():
    # Load test data
    question, solutions, input_output = load_test_data(0)
    print(f"Testing accordion problem with {len(solutions)} solutions")
    print(f"Number of test cases: {len(input_output['inputs'])}")
    
    # Create test cases
    test_cases = create_test_cases(input_output["inputs"], input_output["outputs"])
    
    # Initialize runner
    runner = CodeRunner(max_memory_bytes=512 * 1024 * 1024)  # 512MB limit
    
    # Test each solution
    for i, solution in enumerate(solutions):
        print(f"\nTesting solution {i+1}:")
        print("-" * 40)
        
        # Create submission
        submission = CodeSubmission(
            code=solution,
            code_type=CodeType.STANDARD_INPUT,
            test_cases=test_cases
        )
        
        # Evaluate
        result = runner.evaluate(submission)
        
        # Print results
        print(f"Passed {result.passed_count}/{result.total_count} tests")
        print(f"Total time: {result.total_time:.2f}s")
        
        # Print details of failed tests
        for j, test_result in enumerate(result.test_results):
            if test_result.status != "pass":
                print(f"\nFailed test {j}:")
                print(f"Input: {test_cases[j].inputs}")
                print(f"Expected: {test_cases[j].expected_outputs}")
                print(f"Got: {test_result.actual_output}")
                print(f"Status: {test_result.status}")
                if test_result.error_message:
                    print(f"Error: {test_result.error_message}")

if __name__ == "__main__":
    main() 
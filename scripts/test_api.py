#!/usr/bin/env python3
"""
Script to test the API against all problems in the test_data directory.
"""
import json
from pathlib import Path
import requests
from rich import print
from rich.console import Console
from rich.table import Table
from typing import List, Dict, Any

console = Console()

def load_problem(problem_dir: Path) -> dict:
    """Load all problem data from a directory."""
    with open(problem_dir / "question.txt") as f:
        question = f.read()
    
    with open(problem_dir / "solutions.json") as f:
        solutions = json.load(f)
    
    with open(problem_dir / "input_output.json") as f:
        input_output = json.load(f)
    
    return {
        "question": question,
        "solutions": solutions,
        "input_output": input_output,
    }

def test_solution(problem_id: int, solution_code: str, test_cases: List[Dict[str, Any]], code_type: str = "call_based", function_name: str = None) -> bool:
    """Test a solution against the API."""
    submission = {
        "code": solution_code,
        "code_type": code_type,
        "function_name": function_name,
        "test_cases": [
            {
                "inputs": inp,
                "expected_outputs": [out] if not (isinstance(out, list) or isinstance(out, str)) else out,
                "timeout_seconds": 4.0
            }
            for inp, out in test_cases
        ]
    }

    # Only print submission details on error
    try:
        response = requests.post("http://0.0.0.0:8000/evaluate", json=submission)
        if response.status_code == 422:
            print(f"\nError making request: {response.status_code} {response.reason} for url: {response.url}")
            print("Validation error:", response.json())
            print("Request payload:", json.dumps(submission, indent=2))
            return False
        response.raise_for_status()
        result = response.json()
        return all(r["status"] == "pass" for r in result["test_results"])
    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {e}")
        return False

def main():
    # Find all problem directories
    test_data_dir = Path("test_data")
    problem_dirs = sorted(test_data_dir.glob("problem_*"))
    
    # Create results table
    table = Table(title="API Test Results")
    table.add_column("Problem ID", justify="right")
    table.add_column("Type")
    table.add_column("Solutions")
    table.add_column("Tests")
    table.add_column("Results", justify="center")
    
    for problem_dir in problem_dirs:
        problem_id = problem_dir.name.split("_")[1]
        console.print(f"\n[bold blue]Testing problem {problem_id}...[/bold blue]")
        
        # Load problem data
        problem_data = load_problem(problem_dir)
        
        # Determine problem type
        code_type = "call_based" if "fn_name" in problem_data["input_output"] else "standard_input"
        
        # Test each solution
        total_solutions = len(problem_data["solutions"])
        successful_solutions = 0
        
        # Pre-compute test cases to avoid repeated processing
        test_cases = [
            (problem_data["input_output"]["inputs"][i], problem_data["input_output"]["outputs"][i])
            for i in range(len(problem_data["input_output"]["inputs"]))
        ]
        
        for i, solution in enumerate(problem_data["solutions"], 1):
            # Minimal progress indicator
            console.print(f"Testing {i}/{total_solutions}", end="\r")
            
            result = test_solution(
                problem_id=int(problem_id),
                solution_code=solution,
                test_cases=test_cases,
                code_type=code_type,
                function_name=problem_data["input_output"].get("fn_name")
            )
            
            if result:
                successful_solutions += 1
            else:
                # Only print failures
                console.print(f"\n[red]Solution {i} failed[/red]")
        
        # Print final result for problem
        console.print(f"\nProblem {problem_id}: {successful_solutions}/{total_solutions} solutions passed")
        
        # Add row to results table
        table.add_row(
            problem_id,
            code_type,
            f"{successful_solutions}/{total_solutions}",
            str(len(test_cases)),
            "✅" if successful_solutions > 0 else "❌"
        )
    
    # Print final results
    console.print("\n")
    console.print(table)

if __name__ == "__main__":
    main() 
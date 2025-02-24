"""
Core data models for code evaluation.
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator
import hashlib
import json


class CodeType(str, Enum):
    """Type of code being evaluated."""
    CALL_BASED = "call_based"  # Function calls with specific inputs
    STANDARD_INPUT = "standard_input"  # Code that reads from stdin


class JobStatus(str, Enum):
    """Status of a batch job."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class ExecutionResult(str, Enum):
    """Possible execution results."""
    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    COMPILATION_ERROR = "compilation_error"
    RUNTIME_ERROR = "runtime_error"
    MEMORY_ERROR = "memory_error"


class TestCase(BaseModel):
    """A single test case with inputs and expected outputs."""
    model_config = ConfigDict(frozen=True)

    inputs: Union[List[Any], str] = Field(
        description="Input values for the test case. Either a list of arguments or a string for stdin."
    )
    expected_outputs: Union[List[Any], str] = Field(
        description="Expected output values. Either a list of return values or expected stdout."
    )
    timeout_seconds: float = Field(default=4.0, ge=0.0, description="Timeout in seconds for this test case.")


class CodeSubmission(BaseModel):
    """A code submission to be evaluated."""
    model_config = ConfigDict(frozen=True)

    code: str = Field(description="The Python code to evaluate.")
    code_type: Optional[CodeType] = Field(
        default=None,
        description="The type of code (call-based or standard input). If not provided, will be inferred from function_name."
    )
    function_name: Optional[str] = Field(
        default=None, 
        description="For call-based submissions, the name of the function to call."
    )
    test_cases: List[TestCase] = Field(description="List of test cases to run.")

    @model_validator(mode='before')
    @classmethod
    def infer_code_type(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Infer code_type from function_name if not provided."""
        if values.get('code_type') is None:
            values['code_type'] = CodeType.CALL_BASED if values.get('function_name') is not None else CodeType.STANDARD_INPUT
        return values
    
    def get_hash(self) -> str:
        """Generate a deterministic hash for this submission."""
        # Create a dictionary of all fields that affect execution
        hash_dict = {
            "code": self.code,
            "code_type": self.code_type,
            "function_name": self.function_name,
            "test_cases": [
                {
                    "inputs": tc.inputs,
                    "expected_outputs": tc.expected_outputs,
                    "timeout_seconds": tc.timeout_seconds
                }
                for tc in self.test_cases
            ]
        }
        # Convert to stable JSON string and hash
        json_str = json.dumps(hash_dict, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()


class BatchSubmission(BaseModel):
    """A batch of code submissions to be evaluated."""
    model_config = ConfigDict(frozen=True)
    
    submissions: List[CodeSubmission] = Field(description="List of code submissions to evaluate.")
    
    def get_hash(self) -> str:
        """Generate a deterministic hash for this batch."""
        # Hash each submission and combine
        submission_hashes = [s.get_hash() for s in self.submissions]
        combined = "|".join(submission_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()


class TestResult(BaseModel):
    """Result of a single test case execution."""
    model_config = ConfigDict(frozen=True)

    status: ExecutionResult = Field(description="The execution result status.")
    actual_output: Optional[Any] = Field(default=None, description="The actual output produced.")
    error_message: Optional[str] = Field(default=None, description="Error message if execution failed.")
    execution_time: float = Field(description="Time taken to execute the test case in seconds.")
    memory_used: Optional[int] = Field(
        default=None, 
        description="Peak memory usage in bytes during execution."
    )


class EvaluationResult(BaseModel):
    """Complete evaluation results for a code submission."""
    model_config = ConfigDict(frozen=True)

    test_results: List[TestResult] = Field(description="Results for each test case.")
    passed_count: int = Field(description="Number of test cases passed.")
    total_count: int = Field(description="Total number of test cases.")
    total_time: float = Field(description="Total execution time in seconds.")
    peak_memory: Optional[int] = Field(
        default=None,
        description="Peak memory usage in bytes across all test cases."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the evaluation."
    )


class BatchResult(BaseModel):
    """Results for a batch of code submissions."""
    model_config = ConfigDict(frozen=True)
    
    job_id: str = Field(description="The unique job ID (hash) for this batch.")
    status: JobStatus = Field(description="Current status of the batch job.")
    results: List[Optional[EvaluationResult]] = Field(description="Results for each submission, None if not yet evaluated.")
    error: Optional[str] = Field(default=None, description="Error message if the batch failed.")
    total_time: Optional[float] = Field(default=None, description="Total time taken for all evaluations.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the batch evaluation."
    ) 
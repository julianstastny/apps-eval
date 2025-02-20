**WARNING: This repository is not throughly tested! And it's 90% AI-generated. I don't recommend using it yet.**

# APPS Evaluation Pipeline

A streamlined pipeline for testing Python code on the APPS dataset on RunPod. This project provides a clean, efficient, and safe way to evaluate code submissions against the APPS dataset test cases.

## Features

- Modern, async-first architecture using FastAPI
- Secure code execution environment with resource limits
- RESTful API for remote evaluation
- Comprehensive test case validation
- Detailed execution metrics and logging

## Requirements
- Python 3.9 
- uv (recommended) or pip for dependency management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/apps-eval.git
cd apps-eval
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
# Using uv (recommended for faster installation)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

The service can be configured using the following environment variables:
- `MAX_MEMORY_MB`: Maximum memory limit for code execution (default: 512)
- `DEFAULT_TIMEOUT`: Default timeout in seconds (default: 4.0)
- `LOG_LEVEL`: Logging level (default: INFO)

## Usage

### Local Development

1. Start the API server:
```bash
uvicorn apps_eval.api.server:app --reload --host 0.0.0.0 --port 8000
```

2. Run tests:
```bash
python -m pytest
```

The API documentation will be available at http://localhost:8000/docs.

### RunPod Deployment

1. Create a new pod on RunPod:
   - Select a CPU-optimized template (e.g., `runpod/base:0.4.0-cpu`)
   - Recommended specs: 
     - At least 4GB RAM
     - CPU-only (no GPU needed)
   - Open HTTP port 8000

2. Connect to your pod via SSH or the web terminal

3. Clone and install:
```bash
git clone https://github.com/yourusername/apps-eval.git
cd apps-eval
python -m venv venv
source venv/bin/activate
uv pip install -r requirements.txt
```

4. Start the server:
```bash
uvicorn apps_eval.api.server:app --host 0.0.0.0 --port 8000
```

The API will be available at `https://{POD_ID}-8000.proxy.runpod.net`.

## API Usage

The API supports two types of code evaluation, which are automatically inferred:

1. Call-based evaluation (when `function_name` is provided)
2. Standard input evaluation (when no `function_name` is provided)

### Call-Based Example

```python
import requests

code = """
def reverseWords(s: str) -> str:
    words = [word for word in s.split(' ') if word]
    return ' '.join(words[::-1])
"""

submission = {
    "code": code,
    "function_name": "reverseWords",  # This makes it a call-based submission
    "test_cases": [
        {
            "inputs": ["the sky is blue"],
            "expected_outputs": "blue is sky the"
        }
    ]
}

response = requests.post("http://localhost:8000/evaluate", json=submission)
result = response.json()
print(f"Passed {result['passed_count']}/{result['total_count']} tests")
```

### Standard Input Example

```python
import requests

code = """
n = int(input())
numbers = list(map(int, input().split()))
print(sum(numbers))
"""

submission = {
    "code": code,  # No function_name -> standard input submission
    "test_cases": [
        {
            "inputs": ["3", "1 2 3"],
            "expected_outputs": ["6"]
        }
    ]
}

response = requests.post("http://localhost:8000/evaluate", json=submission)
result = response.json()
print(f"Passed {result['passed_count']}/{result['total_count']} tests")
```

### Response Format

```python
{
    "test_results": [
        {
            "status": "pass" | "fail" | "timeout" | "compilation_error" | "runtime_error",
            "actual_output": Any,  # The output produced by the code
            "error_message": Optional[str],  # Present if there was an error
            "execution_time": float  # Time taken in seconds
        }
    ],
    "passed_count": int,  # Number of tests passed
    "total_count": int,   # Total number of tests
    "total_time": float,  # Total execution time in seconds
    "metadata": {
        "code_type": str,
        "function_name": Optional[str]
    }
}
```

## Security Features

The project implements several security measures for safe code execution:

- Resource limits
  - Memory limits (configurable via `MAX_MEMORY_MB`, Linux only)
  - CPU time limits via configurable timeouts
  - Process resource restrictions

- Sandboxed execution environment
  - Disables dangerous built-in functions
  - Blocks access to system and network operations
  - Prevents file system access
  - Based on OpenAI's HumanEval security approach
  - Note: This is not a full security sandbox - use with trusted code only

- Input validation and sanitization
  - Type checking via Pydantic models
  - Input format validation
  - Safe type conversion for test cases

- Configurable timeouts
  - Global default timeout setting
  - Per-test case timeout configuration
  - Proper cleanup on timeout

- Error isolation
  - Separate error types (compilation, runtime, timeout, etc.)
  - Detailed error messages and stack traces
  - Resource cleanup on errors

### Security Limitations

- Memory limits are not available on macOS
- Memory usage tracking is not yet implemented
- The sandbox is not suitable for running untrusted code
- Some system calls may still be available# apps-eval

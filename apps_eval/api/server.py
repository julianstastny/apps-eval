"""
FastAPI server for code evaluation.
"""
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from apps_eval.core.models import CodeSubmission, EvaluationResult
from apps_eval.core.runner import CodeRunner
from apps_eval.core.config import settings

# Configure logging
logger.add(
    settings.log_file,
    rotation="100 MB",
    retention="10 days",
    level=settings.log_level,
    backtrace=True,
    diagnose=True,
)

app = FastAPI(
    title="APPS Evaluation API",
    description="API for evaluating Python code submissions against APPS dataset test cases",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize code runner with settings
runner = CodeRunner(
    max_memory_bytes=settings.max_memory_bytes,
    default_timeout=settings.default_timeout,
)


@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate_code(submission: CodeSubmission) -> EvaluationResult:
    """
    Evaluate a code submission against its test cases.
    
    Args:
        submission: The code submission to evaluate
        
    Returns:
        EvaluationResult containing the evaluation results
        
    Raises:
        HTTPException: If there's an error during evaluation
    """
    try:
        logger.info(f"Received submission of type {submission.code_type}")
        result = runner.evaluate(submission)
        logger.info(
            f"Evaluation completed: {result.passed_count}/{result.total_count} tests passed "
            f"in {result.total_time:.2f}s"
        )
        return result
        
    except Exception as e:
        logger.exception("Error during evaluation")
        raise HTTPException(
            status_code=500,
            detail=f"Error during evaluation: {str(e)}",
        )


@app.get("/health")
async def health_check() -> dict:
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "config": {
            "max_memory_mb": settings.max_memory_mb,
            "default_timeout": settings.default_timeout,
            "log_level": settings.log_level,
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "apps_eval.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    ) 
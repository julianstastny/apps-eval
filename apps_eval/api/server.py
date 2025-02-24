"""
FastAPI server for code evaluation.
"""
from typing import Optional
import psutil
from asyncio import Semaphore
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from apps_eval.core.models import (
    BatchSubmission,
    BatchResult,
    CodeSubmission,
    EvaluationResult,
)
from apps_eval.core.runner import CodeRunner
from apps_eval.core.queue import JobQueue
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


class ResourceAwareRateLimiter:
    """Rate limiter that adapts to available system resources."""
    
    def __init__(self, settings):
        self.settings = settings
        self.semaphore = None
        self.update_limit()
        # Log initial capacity
        logger.info(
            f"Initial capacity: {self.semaphore._value} concurrent requests "
            f"(Memory: {self.get_available_memory_mb():.1f}MB available, "
            f"CPU cores: {psutil.cpu_count()})"
        )
    
    def get_available_memory_mb(self) -> float:
        """Get available memory in MB."""
        vm = psutil.virtual_memory()
        return vm.available / (1024 * 1024)
    
    def update_limit(self):
        """Update the concurrent request limit based on current resources."""
        max_concurrent = self.calculate_max_concurrent_requests()
        if not self.semaphore or abs(self.semaphore._value - max_concurrent) / self.semaphore._value > 0.1:
            old_value = self.semaphore._value if self.semaphore else None
            self.semaphore = Semaphore(max_concurrent)
            if old_value is not None:
                logger.info(f"Updated concurrent request limit: {old_value} -> {max_concurrent}")
    
    def calculate_max_concurrent_requests(self) -> int:
        """Calculate the maximum number of concurrent requests based on system resources."""
        available_mem_mb = self.get_available_memory_mb()
        total_mem_mb = psutil.virtual_memory().total / (1024 * 1024)
        cpu_count = psutil.cpu_count()
        
        # More conservative memory limits:
        # - Leave 20% of total memory for system
        # - Each request might use up to max_memory_mb
        # - Add buffer for uvicorn and other overhead
        system_reserve_mb = total_mem_mb * 0.2
        uvicorn_buffer_mb = 512  # Reserve 512MB for uvicorn
        available_for_requests = max(0, available_mem_mb - system_reserve_mb - uvicorn_buffer_mb)
        
        # Consider both memory and CPU constraints
        mem_limit = int(available_for_requests / (self.settings.max_memory_mb * 1.2))  # Add 20% overhead per request
        cpu_limit = cpu_count * 2  # Rule of thumb: 2 I/O-bound tasks per CPU
        
        # Ensure at least one request can run if we have enough memory
        min_requests = 1 if available_mem_mb > (self.settings.max_memory_mb + system_reserve_mb + uvicorn_buffer_mb) else 0
        return max(min_requests, min(mem_limit, cpu_limit))

    async def check_and_update_limit(self):
        """Async wrapper for update_limit that can be called directly."""
        self.update_limit()


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

# Initialize rate limiter and code runner
rate_limiter = ResourceAwareRateLimiter(settings)
runner = CodeRunner(
    max_memory_bytes=settings.max_memory_bytes,
    default_timeout=settings.default_timeout,
)

# Initialize job queue
job_queue = JobQueue(runner)


@app.post("/evaluate")
async def evaluate_code(
    submission: CodeSubmission,
    background_tasks: BackgroundTasks
) -> EvaluationResult:
    """
    Evaluate a code submission against its test cases.
    
    Args:
        submission: The code submission to evaluate
        background_tasks: FastAPI background tasks
        
    Returns:
        EvaluationResult containing the evaluation results
        
    Raises:
        HTTPException: If there's an error during evaluation
    """
    try:
        # Check if we have enough memory before accepting the request
        available_mem_mb = rate_limiter.get_available_memory_mb()
        if available_mem_mb < settings.max_memory_mb * 1.5:  # Need 50% more than max_memory for safety
            raise HTTPException(
                status_code=503,
                detail="Server is low on memory. Please try again later."
            )
        
        async with rate_limiter.semaphore:
            # Update limits after this request completes
            background_tasks.add_task(rate_limiter.check_and_update_limit)
            
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


@app.post("/evaluate/batch")
async def submit_batch(batch: BatchSubmission) -> BatchResult:
    """
    Submit a batch of code submissions for evaluation.
    
    Args:
        batch: The batch of submissions to evaluate
        
    Returns:
        BatchResult containing the job ID and initial status
        
    Raises:
        HTTPException: If there's an error submitting the batch
    """
    try:
        job_id = job_queue.submit_batch(batch)
        return job_queue.get_result(job_id)
    except Exception as e:
        logger.exception("Error submitting batch")
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting batch: {str(e)}",
        )


@app.get("/evaluate/batch/{job_id}")
async def get_batch_result(job_id: str) -> BatchResult:
    """
    Get the results for a batch job.
    
    Args:
        job_id: The job ID to get results for
        
    Returns:
        BatchResult containing the current status and any available results
        
    Raises:
        HTTPException: If the job is not found
    """
    result = job_queue.get_result(job_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    return result


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint with resource metrics."""
    vm = psutil.virtual_memory()
    return {
        "status": "healthy",
        "resources": {
            "total_memory_mb": vm.total / (1024 * 1024),
            "available_memory_mb": rate_limiter.get_available_memory_mb(),
            "memory_percent": vm.percent,
            "max_concurrent_requests": rate_limiter.semaphore._value,
            "current_requests": rate_limiter.semaphore._value - rate_limiter.semaphore._wake_count,
            "cpu_percent": psutil.cpu_percent(),
            "active_jobs": len([j for j in job_queue.jobs.values() if j.status == "in_progress"]),
            "total_jobs": len(job_queue.jobs),
        },
        "config": {
            "max_memory_mb": settings.max_memory_mb,
            "default_timeout": settings.default_timeout,
            "log_level": settings.log_level,
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn with more conservative memory settings
    uvicorn.run(
        "apps_eval.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        limit_concurrency=rate_limiter.semaphore._value,  # Match our rate limiter
        limit_max_requests=rate_limiter.semaphore._value,  # Match our rate limiter
        http={
            "h11_max_incomplete_size": 16384,  # 16KB max incomplete request size
            "max_incomplete_headers": 16384,    # 16KB max incomplete headers
        }
    ) 
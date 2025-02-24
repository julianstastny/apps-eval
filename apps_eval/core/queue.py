"""
Job queue management for batch code evaluations.
"""
from typing import Dict, Optional
import time
import asyncio
from loguru import logger

from apps_eval.core.models import BatchSubmission, BatchResult, JobStatus, EvaluationResult
from apps_eval.core.runner import CodeRunner


class JobQueue:
    """Manages batch job submissions and their results."""
    
    def __init__(self, runner: CodeRunner):
        self.runner = runner
        self.jobs: Dict[str, BatchResult] = {}
        self.processing_lock = asyncio.Lock()
    
    def submit_batch(self, batch: BatchSubmission) -> str:
        """
        Submit a batch of code submissions for evaluation.
        
        Args:
            batch: The batch submission to evaluate
            
        Returns:
            The job ID (hash) for this batch
        """
        job_id = batch.get_hash()
        
        # If job exists and is done or failed, return existing job ID
        if job_id in self.jobs and self.jobs[job_id].status in (JobStatus.DONE, JobStatus.FAILED):
            return job_id
            
        # Create new job if it doesn't exist
        if job_id not in self.jobs:
            self.jobs[job_id] = BatchResult(
                job_id=job_id,
                status=JobStatus.PENDING,
                results=[None] * len(batch.submissions)
            )
            
            # Start processing in background
            asyncio.create_task(self._process_batch(job_id, batch))
            
        return job_id
    
    def get_result(self, job_id: str) -> Optional[BatchResult]:
        """
        Get the results for a batch job.
        
        Args:
            job_id: The job ID to get results for
            
        Returns:
            The batch results if found, None otherwise
        """
        return self.jobs.get(job_id)
    
    def get_result_by_batch(self, batch: BatchSubmission) -> Optional[BatchResult]:
        """
        Get results for a batch by recomputing its hash.
        
        Args:
            batch: The batch submission to get results for
            
        Returns:
            The batch results if found, None otherwise
        """
        return self.get_result(batch.get_hash())
    
    async def _process_batch(self, job_id: str, batch: BatchSubmission) -> None:
        """
        Process a batch job in the background.
        
        Args:
            job_id: The job ID to process
            batch: The batch submission to evaluate
        """
        async with self.processing_lock:
            try:
                job = self.jobs[job_id]
                if job.status != JobStatus.PENDING:
                    return
                    
                job.status = JobStatus.IN_PROGRESS
                start_time = time.time()
                
                # Process each submission
                for i, submission in enumerate(batch.submissions):
                    try:
                        result = self.runner.evaluate(submission)
                        job.results[i] = result
                    except Exception as e:
                        logger.exception(f"Error processing submission {i} in batch {job_id}")
                        # Continue processing other submissions
                        job.results[i] = EvaluationResult(
                            test_results=[],
                            passed_count=0,
                            total_count=len(submission.test_cases),
                            total_time=0.0,
                            metadata={"error": str(e)}
                        )
                
                # Update job status
                job.status = JobStatus.DONE
                job.total_time = time.time() - start_time
                
            except Exception as e:
                logger.exception(f"Error processing batch {job_id}")
                job = self.jobs[job_id]
                job.status = JobStatus.FAILED
                job.error = str(e) 
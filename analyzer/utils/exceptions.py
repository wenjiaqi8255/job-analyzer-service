class PipelineError(Exception):
    """Base exception for all errors raised by the analysis pipeline."""
    pass

class JobProcessingError(PipelineError):
    """
    Exception raised for errors that occur while processing a single job.
    
    Attributes:
        job_id -- The ID of the job that failed
        original_error -- The original exception that was caught
    """
    def __init__(self, job_id, original_error):
        self.job_id = job_id
        self.original_error = original_error
        super().__init__(f"Error processing job {job_id}: {original_error}")

class ModelLoadingError(PipelineError):
    """Exception raised when a machine learning model fails to load."""
    pass

class ConfigurationError(PipelineError):
    """Exception raised for errors related to application configuration."""
    pass 
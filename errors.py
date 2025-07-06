"""
Custom error classes for the Parallel GPT Framework.
"""


class ParallelLLMError(Exception):
    """Base exception for Parallel GPT Framework."""
    
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(ParallelLLMError):
    """Error in framework configuration."""
    
    def __init__(self, message: str):
        super().__init__(message, "CONFIG_ERROR")


class ProcessingError(ParallelLLMError):
    """Error during parallel processing."""
    
    def __init__(self, message: str, failed_processors: int = None):
        super().__init__(message, "PROCESSING_ERROR")
        self.failed_processors = failed_processors
    
    def __str__(self):
        base_msg = super().__str__()
        if self.failed_processors is not None:
            return f"{base_msg} (Failed processors: {self.failed_processors})"
        return base_msg


class DecisionMakerError(ParallelLLMError):
    """Error in decision maker processing."""
    
    def __init__(self, message: str, fallback_used: bool = False):
        super().__init__(message, "DECISION_MAKER_ERROR")
        self.fallback_used = fallback_used
    
    def __str__(self):
        base_msg = super().__str__()
        if self.fallback_used:
            return f"{base_msg} (Fallback response used)"
        return base_msg


class ValidationError(ParallelLLMError):
    """Error in response validation."""
    
    def __init__(self, message: str, invalid_field: str = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.invalid_field = invalid_field
    
    def __str__(self):
        base_msg = super().__str__()
        if self.invalid_field:
            return f"{base_msg} (Invalid field: {self.invalid_field})"
        return base_msg


class APIError(ParallelLLMError):
    """Error related to OpenAI API calls."""
    
    def __init__(self, message: str, status_code: int = None, retry_after: int = None):
        super().__init__(message, "API_ERROR")
        self.status_code = status_code
        self.retry_after = retry_after
    
    def __str__(self):
        base_msg = super().__str__()
        if self.status_code:
            base_msg += f" (Status: {self.status_code})"
        if self.retry_after:
            base_msg += f" (Retry after: {self.retry_after}s)"
        return base_msg


class TimeoutError(ParallelLLMError):
    """Error when operations exceed timeout limits."""
    
    def __init__(self, message: str, timeout_duration: float = None):
        super().__init__(message, "TIMEOUT_ERROR")
        self.timeout_duration = timeout_duration
    
    def __str__(self):
        base_msg = super().__str__()
        if self.timeout_duration:
            return f"{base_msg} (Timeout: {self.timeout_duration}s)"
        return base_msg


class RateLimitError(ParallelLLMError):
    """Error when API rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message, "RATE_LIMIT_ERROR")
        self.retry_after = retry_after
    
    def __str__(self):
        base_msg = super().__str__()
        if self.retry_after:
            return f"{base_msg} (Retry after: {self.retry_after}s)"
        return base_msg


class AuthenticationError(ParallelLLMError):
    """Error in API authentication."""
    
    def __init__(self, message: str = "Invalid API key or authentication failed"):
        super().__init__(message, "AUTH_ERROR")


class ModelError(ParallelLLMError):
    """Error related to model availability or specification."""
    
    def __init__(self, message: str, model_name: str = None):
        super().__init__(message, "MODEL_ERROR")
        self.model_name = model_name
    
    def __str__(self):
        base_msg = super().__str__()
        if self.model_name:
            return f"{base_msg} (Model: {self.model_name})"
        return base_msg


def handle_openai_error(error):
    """
    Convert OpenAI errors to framework-specific errors.
    
    Args:
        error: OpenAI error instance
        
    Returns:
        Appropriate framework error
    """
    import openai
    
    if isinstance(error, openai.AuthenticationError):
        return AuthenticationError(str(error))
    elif isinstance(error, openai.RateLimitError):
        return RateLimitError(str(error))
    elif isinstance(error, openai.APITimeoutError):
        return TimeoutError(str(error))
    elif isinstance(error, openai.APIError):
        return APIError(str(error), getattr(error, 'status_code', None))
    else:
        return ProcessingError(f"OpenAI API error: {error}")


def get_error_info(error: ParallelLLMError) -> dict:
    """
    Get detailed information about a framework error.
    
    Args:
        error: Framework error instance
        
    Returns:
        Dictionary with error details
    """
    info = {
        "error_type": type(error).__name__,
        "error_code": error.error_code,
        "message": error.message,
        "full_message": str(error)
    }
    
    # Add specific error details
    if hasattr(error, 'failed_processors'):
        info["failed_processors"] = error.failed_processors
    if hasattr(error, 'fallback_used'):
        info["fallback_used"] = error.fallback_used
    if hasattr(error, 'invalid_field'):
        info["invalid_field"] = error.invalid_field
    if hasattr(error, 'status_code'):
        info["status_code"] = error.status_code
    if hasattr(error, 'retry_after'):
        info["retry_after"] = error.retry_after
    if hasattr(error, 'timeout_duration'):
        info["timeout_duration"] = error.timeout_duration
    if hasattr(error, 'model_name'):
        info["model_name"] = error.model_name
    
    return info


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable.
    
    Args:
        error: Error instance
        
    Returns:
        True if the error is retryable, False otherwise
    """
    retryable_errors = (
        RateLimitError,
        TimeoutError,
        APIError,
        ProcessingError
    )
    
    non_retryable_errors = (
        AuthenticationError,
        ValidationError,
        ConfigurationError,
        ModelError
    )
    
    if isinstance(error, non_retryable_errors):
        return False
    
    if isinstance(error, retryable_errors):
        return True
    
    # For unknown errors, default to retryable
    return True 
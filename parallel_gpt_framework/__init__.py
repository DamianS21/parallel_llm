"""
Parallel GPT Framework - A framework for parallelizing GPT structured output processing.

This package provides:
- Parallel processing of GPT requests
- Intelligent decision making to select best responses
- OpenAI API compatibility
- Drop-in replacement for OpenAI's beta.chat.completions.parse
"""

from .core import ParallelGPTFramework
from .config import (
    FrameworkConfig, 
    ConfigurationManager,
    create_default_config,
    create_performance_config,
    create_robust_config,
    create_development_config
)
from .errors import (
    ParallelGPTError,
    ConfigurationError,
    ProcessingError,
    DecisionMakerError,
    ValidationError,
    APIError,
    TimeoutError,
    RateLimitError,
    AuthenticationError,
    ModelError,
    handle_openai_error,
    get_error_info,
    is_retryable_error
)
from .prompts import DECISION_MAKER_PROMPT
from .interfaces import (
    ParsedMessage,
    Choice,
    ParallelCompletion,
    ParallelCompletionInterface,
    ParallelChat,
    ParallelBeta
)

# Version information
__version__ = "1.0.0"
__author__ = "Parallel GPT Framework Team"
__email__ = "contact@parallalgpt.com"
__license__ = "MIT"

# Main exports
__all__ = [
    # Core classes
    "ParallelGPTFramework",
    
    # Configuration
    "FrameworkConfig",
    "ConfigurationManager",
    "create_default_config",
    "create_performance_config", 
    "create_robust_config",
    "create_development_config",
    
    # Errors
    "ParallelGPTError",
    "ConfigurationError",
    "ProcessingError",
    "DecisionMakerError",
    "ValidationError",
    "APIError",
    "TimeoutError",
    "RateLimitError",
    "AuthenticationError",
    "ModelError",
    "handle_openai_error",
    "get_error_info",
    "is_retryable_error",
    
    # Prompts
    "DECISION_MAKER_PROMPT",
    
    # Interfaces
    "ParsedMessage",
    "Choice",
    "ParallelCompletion",
    "ParallelCompletionInterface",
    "ParallelChat",
    "ParallelBeta",
    
    # Convenience function
    "create_framework",
]

# Convenience function
def create_framework(api_key: str, **config_kwargs) -> ParallelGPTFramework:
    """
    Create a ParallelGPTFramework instance with optional configuration.
    
    Args:
        api_key: OpenAI API key
        **config_kwargs: Configuration parameters
        
    Returns:
        Configured ParallelGPTFramework instance
    """
    config = FrameworkConfig(**config_kwargs)
    return ParallelGPTFramework(api_key=api_key, config=config)

# Package metadata
__package_info__ = {
    "name": "parallel-gpt-framework",
    "version": __version__,
    "description": "Framework for parallelizing GPT structured output processing with decision making",
    "author": __author__,
    "email": __email__,
    "license": __license__,
    "url": "https://github.com/your-org/parallel-gpt-framework",
    "keywords": ["gpt", "parallel", "openai", "structured-output", "decision-making", "ai"],
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
} 
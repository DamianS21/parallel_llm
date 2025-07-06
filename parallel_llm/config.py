"""
Configuration classes and utilities for the Parallel GPT Framework.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

from .prompts import DECISION_MAKER_PROMPT


class FrameworkConfig(BaseModel):
    """Configuration for the parallel GPT framework."""
    
    num_processors: int = Field(default=3, description="Number of parallel processors to use")
    timeout: float = Field(default=30.0, description="Timeout for API requests in seconds")
    max_retries: int = Field(default=2, description="Maximum number of retries for failed requests")
    decision_maker_model: str = Field(default="gpt-4o", description="Model to use for decision making")
    decision_maker_temperature: float = Field(default=0.1, description="Temperature for decision maker")
    decision_maker_prompt: str = Field(default=DECISION_MAKER_PROMPT, description="Prompt for decision maker")
    enable_logging: bool = Field(default=True, description="Whether to enable logging")
    log_level: str = Field(default="INFO", description="Logging level")
    
    @field_validator('num_processors')
    @classmethod
    def validate_num_processors(cls, v):
        if v < 1:
            raise ValueError("Number of processors must be at least 1")
        return v
    
    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v
    
    @field_validator('max_retries')
    @classmethod
    def validate_max_retries(cls, v):
        if v < 0:
            raise ValueError("Max retries must be non-negative")
        return v
    
    @field_validator('decision_maker_temperature')
    @classmethod
    def validate_decision_maker_temperature(cls, v):
        if v < 0 or v > 2:
            raise ValueError("Decision maker temperature must be between 0 and 2")
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        if v not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError("Log level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
        return v
    
    @model_validator(mode='after')
    def configure_logging(self):
        """Configure logging if enabled."""
        if self.enable_logging:
            logging.basicConfig(
                level=getattr(logging, self.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        return self


class ConfigurationManager:
    """Manages configuration updates and validation."""
    
    def __init__(self, config: FrameworkConfig):
        self.config = config
    
    def update_config(self, **kwargs) -> None:
        """
        Update framework configuration with validation.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        from .errors import ConfigurationError
        
        try:
            # Create a new config instance with updated values
            config_dict = self.config.model_dump()
            config_dict.update(kwargs)
            
            # Validate the new configuration
            self.config = FrameworkConfig(**config_dict)
            
        except Exception as e:
            raise ConfigurationError(f"Configuration update failed: {e}")
    
    def get_config(self) -> FrameworkConfig:
        """Get current framework configuration."""
        return self.config
    
    def get_config_summary(self) -> dict:
        """
        Get a summary of current configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            "num_processors": self.config.num_processors,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
            "decision_maker_model": self.config.decision_maker_model,
            "decision_maker_temperature": self.config.decision_maker_temperature,
            "decision_maker_prompt": self.config.decision_maker_prompt[:100] + "..." if len(self.config.decision_maker_prompt) > 100 else self.config.decision_maker_prompt,
            "logging_enabled": self.config.enable_logging,
            "log_level": self.config.log_level
        }


def create_default_config() -> FrameworkConfig:
    """Create a default configuration instance."""
    return FrameworkConfig()


def create_performance_config() -> FrameworkConfig:
    """Create a configuration optimized for performance."""
    return FrameworkConfig(
        num_processors=5,
        timeout=20.0,
        max_retries=1,
        decision_maker_temperature=0.0
    )


def create_robust_config() -> FrameworkConfig:
    """Create a configuration optimized for robustness."""
    return FrameworkConfig(
        num_processors=3,
        timeout=60.0,
        max_retries=3,
        decision_maker_temperature=0.1
    )


def create_development_config() -> FrameworkConfig:
    """Create a configuration suitable for development."""
    return FrameworkConfig(
        num_processors=2,
        timeout=15.0,
        max_retries=1,
        enable_logging=True,
        log_level="DEBUG"
    ) 
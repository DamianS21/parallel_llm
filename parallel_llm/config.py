"""
Configuration classes and utilities for the Parallel GPT Framework.
"""

from dataclasses import dataclass
from typing import Optional
import logging

from .prompts import DECISION_MAKER_PROMPT


@dataclass
class FrameworkConfig:
    """Configuration for the parallel GPT framework."""
    num_processors: int = 3
    timeout: float = 30.0
    max_retries: int = 2
    decision_maker_model: str = "gpt-4o"
    decision_maker_temperature: float = 0.1
    decision_maker_prompt: str = DECISION_MAKER_PROMPT
    enable_logging: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        from .errors import ConfigurationError
        
        if self.num_processors < 1:
            raise ConfigurationError("Number of processors must be at least 1")
        if self.timeout <= 0:
            raise ConfigurationError("Timeout must be positive")
        if self.max_retries < 0:
            raise ConfigurationError("Max retries must be non-negative")
        if self.decision_maker_temperature < 0 or self.decision_maker_temperature > 2:
            raise ConfigurationError("Decision maker temperature must be between 0 and 2")
        if self.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ConfigurationError("Log level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
        
        # Configure logging if enabled
        if self.enable_logging:
            logging.basicConfig(
                level=getattr(logging, self.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )


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
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    raise ConfigurationError(f"Unknown config parameter: {key}")
            
            # Validate configuration after updates
            self.config.__post_init__()
            
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
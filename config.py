"""
Configuration classes and utilities for the Parallel GPT Framework.
"""

from dataclasses import dataclass
from typing import Optional
import logging

# Decision maker prompt constant
DECISION_MAKER_PROMPT = """Your goal is to analyse the task and previous agent responses and decide on the best possible response to return to the user.

You are an expert decision maker responsible for evaluating multiple AI responses to the same query. Your task is to:

1. **Understand the Original Task**: Carefully analyze what the user is asking for and the context provided.

2. **Evaluate Each Response**: Assess all provided responses based on:
   - **Accuracy**: How factually correct and reliable is the information?
   - **Completeness**: Does it fully address all aspects of the user's request?
   - **Relevance**: How well does it match the specific question asked?
   - **Quality**: Is the response clear, well-structured, and helpful?
   - **Consistency**: Are there any contradictions or logical issues?

3. **Decision Process**: 
   - If one response is clearly superior, select it
   - If multiple responses have different strengths, synthesize the best elements
   - If responses are similar, choose the most comprehensive one
   - Always prioritize accuracy and completeness over style

4. **Output Requirements**:
   - Return the final decision in the EXACT same format as the input responses
   - Do not add meta-commentary or explanations about your choice
   - Ensure the response directly answers the user's original question
   - Maintain the same level of detail and structure as expected

Remember: Your goal is to provide the user with the single best possible response by leveraging the collective intelligence of multiple AI responses."""


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
        from errors import ConfigurationError
        
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
        from errors import ConfigurationError
        
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
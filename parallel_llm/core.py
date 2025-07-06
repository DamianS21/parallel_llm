"""
Core ParallelLLM class with main processing logic.
"""

import asyncio
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel
import openai
from openai import AsyncOpenAI
import json
import time

from .config import FrameworkConfig, ConfigurationManager
from .errors import (
    ParallelLLMError, ConfigurationError, ProcessingError, 
    DecisionMakerError, ValidationError, handle_openai_error
)
from .prompts import DECISION_MAKER_PROMPT
from .interfaces import ParallelBeta


class ParallelLLM:
    """
    Framework for parallelizing GPT structured output processing with decision maker.
    """
    
    def __init__(self, api_key: str, config: Optional[FrameworkConfig] = None):
        """
        Initialize the parallel GPT framework.
        
        Args:
            api_key: OpenAI API key
            config: Framework configuration
        """
        self.config = config or FrameworkConfig()
        self.config_manager = ConfigurationManager(self.config)
        self.openai_client = AsyncOpenAI(api_key=api_key)
        
        # Create beta interface for API compatibility
        self.beta = ParallelBeta(self)

    async def _make_single_request(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Type[BaseModel],
        temperature: float,
        **kwargs
    ) -> Any:
        """
        Make a single async request to OpenAI API with retries.
        
        Args:
            model: Model name
            messages: Messages for the request
            response_format: Pydantic model for structured output
            temperature: Temperature setting
            **kwargs: Additional API parameters
            
        Returns:
            Parsed response
        """
        for attempt in range(self.config.max_retries + 1):
            try:
                completion = await asyncio.wait_for(
                    self.openai_client.beta.chat.completions.parse(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                        temperature=temperature,
                        **kwargs
                    ),
                    timeout=self.config.timeout
                )
                
                return completion.choices[0].message.parsed
                
            except asyncio.TimeoutError:
                if attempt == self.config.max_retries:
                    raise ProcessingError(f"Request timed out after {self.config.max_retries + 1} attempts")
                await asyncio.sleep(2 ** attempt)
                
            except openai.RateLimitError as e:
                if attempt == self.config.max_retries:
                    raise handle_openai_error(e)
                await asyncio.sleep(2 ** attempt)
                
            except openai.APIError as e:
                if attempt == self.config.max_retries:
                    raise handle_openai_error(e)
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise ProcessingError(f"Unexpected error: {e}")
                await asyncio.sleep(2 ** attempt)

    async def _process_parallel_requests(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Type[BaseModel],
        temperature: float,
        **kwargs
    ) -> List[Any]:
        """Process multiple parallel requests to the same prompt."""
        try:
            # Create tasks for parallel processing
            tasks = []
            for i in range(self.config.num_processors):
                task = asyncio.create_task(
                    self._make_single_request(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                        temperature=temperature,
                        **kwargs
                    ),
                    name=f"processor_{i}"
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            successful_results = []
            failed_count = 0
            for result in results:
                if isinstance(result, Exception):
                    failed_count += 1
                else:
                    successful_results.append(result)
            
            if not successful_results:
                raise ProcessingError("All parallel processors failed", failed_processors=failed_count)
                
            return successful_results
            
        except Exception as e:
            if isinstance(e, ProcessingError):
                raise e
            raise ProcessingError(f"Parallel processing error: {e}")
    
    async def _make_decision(
        self,
        responses: List[Any],
        response_format: Type[BaseModel],
        original_messages: List[Dict[str, str]]
    ) -> Any:
        """Use decision maker to select/synthesize final response."""
        try:
            if len(responses) == 1:
                return responses[0]
                
            # Prepare responses for decision maker
            response_texts = []
            for i, response in enumerate(responses):
                if hasattr(response, 'model_dump_json'):
                    response_json = response.model_dump_json(indent=2)
                else:
                    response_json = json.dumps(response, indent=2, default=str)
                response_texts.append(f"Response {i+1}:\n{response_json}")
            
            # Add original context and responses
            original_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in original_messages])
            all_responses = "\n\n".join(response_texts)
            
            decision_messages = [
                {
                    "role": "system", 
                    "content": self.config.decision_maker_prompt
                },
                {
                    "role": "user",
                    "content": f"""Original Query Context:
{original_context}

All Responses to Analyze:
{all_responses}

Please analyze these responses and return the best one or synthesize a better response using the same format as the original responses."""
                }
            ]
            
            # Make decision using the decision maker
            decision_response = await self._make_single_request(
                model=self.config.decision_maker_model,
                messages=decision_messages,
                response_format=response_format,
                temperature=self.config.decision_maker_temperature
            )
            
            return decision_response
            
        except Exception as e:
            if responses:
                return responses[0]
            else:
                raise DecisionMakerError(f"Decision maker failed and no fallback available: {e}")

    async def _parse_internal(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Type[BaseModel],
        temperature: float = 0,
        **kwargs
    ) -> Any:
        """
        Internal parse method that does the actual parallel processing.
        
        Args:
            model: Model name to use
            messages: List of message dictionaries
            response_format: Pydantic model for structured output
            temperature: Temperature for generation
            **kwargs: Additional parameters to pass to OpenAI API
            
        Returns:
            Parsed response in the specified format
        """
        try:
            # Validate input parameters
            if not model:
                raise ValidationError("Model name is required")
            if not messages:
                raise ValidationError("Messages are required")
            if not response_format:
                raise ValidationError("Response format is required")
            if not issubclass(response_format, BaseModel):
                raise ValidationError("Response format must be a Pydantic BaseModel")
            
            # Step 1: Process parallel requests
            parallel_responses = await self._process_parallel_requests(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                **kwargs
            )
            
            # Step 2: Use decision maker to select/synthesize final response
            final_response = await self._make_decision(
                responses=parallel_responses,
                response_format=response_format,
                original_messages=messages
            )
            
            # Step 3: Validate the final response matches expected format
            if not isinstance(final_response, response_format):
                try:
                    if hasattr(final_response, 'model_dump'):
                        final_response = response_format.model_validate(final_response.model_dump())
                    else:
                        final_response = response_format.model_validate(final_response)
                except Exception as e:
                    raise ValidationError(f"Failed to validate final response: {e}")
            
            return final_response
            
        except (ValidationError, ConfigurationError, ProcessingError, DecisionMakerError) as e:
            raise e
        except Exception as e:
            raise ProcessingError(f"Unexpected error: {e}")

    async def parse(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Type[BaseModel],
        temperature: float = 0,
        **kwargs
    ) -> Any:
        """
        Direct method for parallel parsing (alternative to beta interface).
        
        Args:
            model: Model name to use
            messages: List of message dictionaries
            response_format: Pydantic model for structured output
            temperature: Temperature for generation
            **kwargs: Additional parameters to pass to OpenAI API
            
        Returns:
            Parsed response in the specified format (direct result, not wrapped)
        """
        return await self._parse_internal(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            **kwargs
        )

    # Configuration management methods
    def update_config(self, **kwargs) -> None:
        """Update framework configuration."""
        self.config_manager.update_config(**kwargs)
    
    def get_config(self) -> FrameworkConfig:
        """Get current framework configuration."""
        return self.config_manager.get_config()
    
    def get_config_summary(self) -> dict:
        """Get a summary of current configuration."""
        return self.config_manager.get_config_summary() 
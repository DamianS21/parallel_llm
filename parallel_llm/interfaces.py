"""
Interface classes that mirror OpenAI's API structure for the Parallel GPT Framework.
"""

from typing import Any, Dict, List, Type, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from .core import ParallelLLM


class ParsedMessage:
    """Mimics OpenAI's parsed message structure."""
    
    def __init__(self, parsed_content: Any):
        self.parsed = parsed_content


class Choice:
    """Mimics OpenAI's choice structure."""
    
    def __init__(self, parsed_content: Any):
        self.message = ParsedMessage(parsed_content)


class ParallelCompletion:
    """Mimics OpenAI's completion response structure."""
    
    def __init__(self, parsed_content: Any):
        self.choices = [Choice(parsed_content)]


class ParallelCompletionInterface:
    """Class that mimics OpenAI's beta.chat.completions interface."""
    
    def __init__(self, framework: 'ParallelLLM'):
        self.framework = framework
    
    async def parse(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Type[BaseModel],
        **kwargs
    ) -> ParallelCompletion:
        """
        Parse method that mirrors OpenAI beta.chat.completions.parse exactly.
        
        Args:
            model: Model name to use
            messages: List of message dictionaries
            response_format: Pydantic model for structured output
            temperature: Temperature for generation
            **kwargs: Additional parameters to pass to OpenAI API
            
        Returns:
            ParallelCompletion with same structure as OpenAI's response
        """
        parsed_result = await self.framework._parse_internal(
            model=model,
            messages=messages,
            response_format=response_format,
            **kwargs
        )
        
        return ParallelCompletion(parsed_result)


class ParallelChat:
    """Class that mimics OpenAI's beta.chat interface."""
    
    def __init__(self, framework: 'ParallelLLM'):
        self.completions = ParallelCompletionInterface(framework)


class ParallelBeta:
    """Class that mimics OpenAI's beta interface."""
    
    def __init__(self, framework: 'ParallelLLM'):
        self.chat = ParallelChat(framework) 
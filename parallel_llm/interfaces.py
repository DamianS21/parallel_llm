"""
Interface classes that mirror OpenAI's API structure for the Parallel GPT Framework.
"""

from typing import Any, Dict, List, Type, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .core import ParallelLLM


class ParsedMessage(BaseModel):
    """Mimics OpenAI's parsed message structure."""
    
    parsed: Any = Field(description="The parsed content of the message")


class Choice(BaseModel):
    """Mimics OpenAI's choice structure."""
    
    message: ParsedMessage = Field(description="The parsed message")
    
    def __init__(self, parsed_content: Any = None, **data):
        if parsed_content is not None:
            super().__init__(message=ParsedMessage(parsed=parsed_content), **data)
        else:
            super().__init__(**data)


class ParallelCompletion(BaseModel):
    """Mimics OpenAI's completion response structure."""
    
    choices: List[Choice] = Field(description="List of completion choices")
    
    def __init__(self, parsed_content: Any = None, **data):
        if parsed_content is not None:
            super().__init__(choices=[Choice(parsed_content)], **data)
        else:
            super().__init__(**data)


class ParallelCompletionInterface:
    """Class that mimics OpenAI's beta.chat.completions interface."""
    
    def __init__(self, framework: 'ParallelLLM'):
        self.framework = framework
    
    async def parse(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Type[BaseModel],
        pass_reasoning: bool = False,
        **kwargs
    ) -> ParallelCompletion:
        """
        Parse method that mirrors OpenAI beta.chat.completions.parse exactly.
        
        Args:
            model: Model name to use
            messages: List of message dictionaries
            response_format: Pydantic model for structured output
            pass_reasoning: Whether to include reasoning in intermediate responses for decision maker
            **kwargs: Additional parameters to pass to OpenAI API
            
        Returns:
            ParallelCompletion with same structure as OpenAI's response
        """
        parsed_result = await self.framework._parse_internal(
            model=model,
            messages=messages,
            response_format=response_format,
            pass_reasoning=pass_reasoning,
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
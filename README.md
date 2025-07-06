# Parallel GPT Framework

Simple framework for parallelizing GPT structured output processing with automatic decision making. This framework allows you to run multiple parallel requests to OpenAI's GPT models and uses an intelligent decision maker to select or synthesize the best response.

## Features

- **Parallel Processing**: Run multiple identical requests simultaneously to improve response quality
- **Intelligent Decision Making**: AI agent analyzes all responses and selects the best one
- **Drop-in Replacement**: Perfect replacement for OpenAI's `beta.chat.completions.parse`
- **Two Calling Methods**: Use either OpenAI-compatible interface or direct method
- **Fully Configurable**: Customize processors, timeouts, and decision maker behavior
- **Robust Error Handling**: Built-in retries, timeouts, and fallback mechanisms
- **Type Safety**: Full Pydantic model support with type validation
- **Async/Await Support**: Built for high-performance async operations

## Installation

Using `uv` (recommended):
```bash
uv pip install -r requirements.txt
```

Or using `pip`:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
import asyncio
from pydantic import BaseModel, Field
from parallel_gpt_framework import ParallelGPTFramework

class ProductRecommendation(BaseModel):
    product_name: str = Field(description="Name of the recommended product")
    price: float = Field(description="Price of the product")
    reason: str = Field(description="Reason for recommendation")

async def main():
    # Initialize framework
    framework = ParallelGPTFramework(api_key="your-openai-api-key")
    
    # Define your query
    messages = [
        {"role": "system", "content": "You are a product recommendation expert."},
        {"role": "user", "content": "Recommend a laptop for software development under $1500."}
    ]
    
    # Method 1: Drop-in replacement for OpenAI API
    completion = await framework.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=ProductRecommendation,
        temperature=0
    )
    result = completion.choices[0].message.parsed
    
    # Method 2: Direct call (simpler)
    result = await framework.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=ProductRecommendation,
        temperature=0
    )
    
    print(f"Product: {result.product_name}")
    print(f"Price: ${result.price}")
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

## Core Concepts

### 1. Parallel Processing
The framework spawns multiple identical requests to the same OpenAI endpoint, increasing the chances of getting a high-quality response. Default: 3 parallel processors.

### 2. Decision Maker
After collecting all parallel responses, a decision maker agent analyzes them and:
- Selects the best single response based on accuracy, completeness, and relevance
- Synthesizes elements from multiple responses when beneficial
- Always returns the response in your specified format

### 3. Two Calling Methods
Choose the method that fits your needs:
- **Drop-in replacement**: `framework.beta.chat.completions.parse(...)` - exact OpenAI API compatibility
- **Direct method**: `framework.parse(...)` - simpler, returns result directly

## API Reference

### ParallelGPTFramework

#### Constructor
```python
ParallelGPTFramework(api_key: str, config: Optional[FrameworkConfig] = None)
```

#### Main Methods

**Drop-in Replacement Method:**
```python
completion = await framework.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=messages,
    response_format=YourModel,
    temperature=0
)
result = completion.choices[0].message.parsed
```

**Direct Method:**
```python
result = await framework.parse(
    model="gpt-4o-mini",
    messages=messages,
    response_format=YourModel,
    temperature=0
)
```

#### Configuration Methods

**`update_config(**kwargs)`** - Update configuration options
**`get_config()`** - Get current framework configuration  
**`get_config_summary()`** - Get configuration summary

### FrameworkConfig

```python
@dataclass
class FrameworkConfig:
    num_processors: int = 3                      # Number of parallel processors
    timeout: float = 30.0                       # Request timeout in seconds
    max_retries: int = 2                        # Maximum retry attempts
    decision_maker_model: str = "gpt-4o-mini"   # Model for decision making
    decision_maker_temperature: float = 0.1      # Temperature for decision maker
    decision_maker_prompt: str = DECISION_MAKER_PROMPT # Decision maker prompt
    enable_logging: bool = True                  # Enable/disable logging
    log_level: str = "INFO"                     # Logging level
```

## Examples

### Basic Usage (Drop-in Replacement)
```python
# Replace this OpenAI code:
# completion = openai_client.beta.chat.completions.parse(...)
# result = completion.choices[0].message.parsed

# With this:
completion = await framework.beta.chat.completions.parse(...)
result = completion.choices[0].message.parsed
```

### Basic Usage (Direct Method)
```python
framework = ParallelGPTFramework(api_key="your-key")

result = await framework.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    response_format=YourPydanticModel
)
```

### Custom Configuration
```python
from parallel_gpt_framework import FrameworkConfig

config = FrameworkConfig(
    num_processors=5,
    timeout=45.0,
    max_retries=3
)

framework = ParallelGPTFramework(api_key="your-key", config=config)
```

### Runtime Configuration Updates
```python
# Update any configuration options
framework.update_config(
    num_processors=4,
    timeout=20.0,
    max_retries=3,
    decision_maker_temperature=0.2
)

# Get current configuration
config = framework.get_config()
summary = framework.get_config_summary()
```

### Convenience Functions
```python
from parallel_gpt_framework import create_framework

# Quick setup with custom config
framework = create_framework(
    api_key="your-key",
    num_processors=5,
    timeout=45.0
)
```


## Error Types

```python
ParallelGPTError          # Base exception
ConfigurationError        # Invalid configuration
ProcessingError          # Parallel processing failure
DecisionMakerError       # Decision maker failure
ValidationError          # Response validation failure
APIError                 # OpenAI API errors
TimeoutError            # Request timeout errors
RateLimitError          # Rate limit errors
```

## Best Practices

1. **Choose the right calling method**:
   - Use drop-in replacement for existing OpenAI code
   - Use direct method for new implementations
2. **Set appropriate timeouts** based on your use case (20-60 seconds)
3. **Use 3-5 processors** for optimal balance of speed and cost
4. **Handle errors gracefully** with try-catch blocks
5. **Monitor API usage** to manage costs effectively

## Requirements

- Python 3.8+
- OpenAI API key with access to structured outputs
- Dependencies: `openai>=1.35.0`, `pydantic>=2.0.0`


## Development

Using `uv`:
```bash
uv sync
uv run python example_usage.py
```

Using `pip`:
```bash
pip install -r requirements.txt
python example_usage.py
```

## License

MIT License
#!/usr/bin/env python3
"""
Example usage of the Parallel GPT Framework.

This example demonstrates how to use the framework as a drop-in replacement 
for OpenAI's beta.chat.completions.parse with parallel processing.
"""

import asyncio
import os
from typing import List
from pydantic import BaseModel, Field

from parallel_llm import ParallelLLM, FrameworkConfig


class ProductRecommendation(BaseModel):
    """Product recommendation response model."""
    product_name: str = Field(description="Name of the recommended product")
    price: float = Field(description="Price of the product")
    rating: float = Field(description="Product rating out of 5")
    reason: str = Field(description="Reason for recommendation")
    pros: List[str] = Field(description="List of product advantages")
    cons: List[str] = Field(description="List of product disadvantages")


class WalkingDistanceAnalysis(BaseModel):
    """Walking distance analysis response model."""
    starting_point: str = Field(description="Starting location")
    destination: str = Field(description="Destination location")
    distance_miles: float = Field(description="Walking distance in miles")
    distance_kilometers: float = Field(description="Walking distance in kilometers")
    estimated_time_minutes: int = Field(description="Estimated walking time in minutes")
    difficulty_level: str = Field(description="Walking difficulty (easy/moderate/hard)")
    route_highlights: List[str] = Field(description="Notable points along the route")
    safety_tips: List[str] = Field(description="Safety recommendations for the walk")
    alternative_transport: List[str] = Field(description="Alternative transportation options")
    best_time_to_walk: str = Field(description="Best time of day to make this walk")


async def example_calling_methods():
    """Example: Different ways to call the framework."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    framework = ParallelLLM(api_key=api_key)
    
    messages = [
        {"role": "system", "content": "You are a product recommendation expert."},
        {"role": "user", "content": "Recommend a good wireless headphone under $200 for exercise."}
    ]
    
    try:
        # Method 1: Drop-in replacement (OpenAI-like)
        print("Method 1: Drop-in replacement")
        completion = await framework.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=ProductRecommendation,
            temperature=0
        )
        result = completion.choices[0].message.parsed
        print(f"Product: {result.product_name}, Price: ${result.price}")
        
        # Method 2: Direct call
        print("\nMethod 2: Direct call")
        result_direct = await framework.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=ProductRecommendation,
            temperature=0
        )
        print(f"Product: {result_direct.product_name}, Price: ${result_direct.price}")
        
    except Exception as e:
        print(f"Error: {e}")


async def example_pier_walking_distance():
    """Example: Walking distance analysis between Pier 39 and Pier 80."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    framework = ParallelLLM(api_key=api_key)
    
    system_prompt = """You are a local San Francisco walking route expert and travel guide. 
    You have extensive knowledge of San Francisco's geography, neighborhoods, walking routes, 
    and local transportation options."""
    
    user_prompt = """Analyze the walking route from Pier 39 to Pier 80 in San Francisco. 
    Provide detailed information about the distance, time, difficulty, and route highlights. 
    Consider the terrain, safety, and practical aspects of making this walk. 
    Also suggest alternative transportation options if the walk is too long or difficult."""
    
    try:
        analysis = await framework.parse(
            model="gpt-4o-search-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=WalkingDistanceAnalysis,
            pass_reasoning=True
        )
        
        print("Walking Distance Analysis - Pier 39 to Pier 80")
        print(f"From: {analysis.starting_point}")
        print(f"To: {analysis.destination}")
        print(f"Distance: {analysis.distance_miles} miles ({analysis.distance_kilometers} km)")
        print(f"Time: {analysis.estimated_time_minutes} minutes")
        print(f"Difficulty: {analysis.difficulty_level}")
        print(f"Best time: {analysis.best_time_to_walk}")
        
        print("\nRoute highlights:")
        for highlight in analysis.route_highlights:
            print(f"- {highlight}")
        
        print("\nSafety tips:")
        for tip in analysis.safety_tips:
            print(f"- {tip}")
        
        print("\nAlternative transport:")
        for option in analysis.alternative_transport:
            print(f"- {option}")
        
    except Exception as e:
        print(f"Error: {e}")


async def example_configuration():
    """Example: Configuration options."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Create custom configuration
    config = FrameworkConfig(
        num_processors=5,
        timeout=45.0,
        max_retries=3
    )
    
    framework = ParallelLLM(api_key=api_key, config=config)
    
    print("Configuration:")
    config_summary = framework.get_config_summary()
    for key, value in config_summary.items():
        print(f"  {key}: {value}")





async def main():
    """Run all examples."""
    print("Parallel GPT Framework Examples")
    print("=" * 40)
    
    # await example_calling_methods()
    print("\n" + "=" * 40)
    await example_pier_walking_distance()
    print("\n" + "=" * 40)


if __name__ == "__main__":
    asyncio.run(main()) 
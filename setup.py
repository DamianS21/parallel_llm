#!/usr/bin/env python3
"""
Setup script for Parallel GPT Framework.
"""

import os
from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Core requirements only (without development dependencies)
core_requirements = [
    "openai>=1.35.0",
    "pydantic>=2.0.0",
    "asyncio-extras>=1.3.0",
]

setup(
    name="parallel-gpt-framework",
    version="1.0.0",
    author="Parallel GPT Framework Team",
    author_email="contact@parallalgpt.com",
    description="A framework for parallelizing GPT structured output processing with decision making",
    long_description=open("README.md", "r", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/parallel-gpt-framework",
    packages=find_packages(),
    py_modules=["parallel_gpt_framework"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=1.0.0",
        ],
        "extras": [
            "python-dotenv>=0.19.0",
            "rich>=13.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "parallel-gpt-example=example_usage:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 
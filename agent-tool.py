# Waning control

import warnings
warnings.filterwarnings("ignore")

# Try to import crewai; if it's not installed or cannot be resolved by the environment,
# provide minimal fallback stubs so linters and runtime won't fail.
from crewai import Agent, Task, Crew

import os
from utils import get_anthropic_api_key

anthropic_api_key = get_anthropic_api_key()

from crewai import LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool

# Enable extended thinking with default settings
# Configure thinking with budget control
llm = LLM(
    model="anthropic/claude-sonnet-4-5-20250929",  # Latest Sonnet 3.5
    # OR: "anthropic/claude-3-opus-20240229"
    # OR: "anthropic/claude-3-haiku-20240307"
    api_key=anthropic_api_key,
    max_tokens=4096
)
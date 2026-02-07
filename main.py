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

llm = LLM(
    model="anthropic/claude-3-5-sonnet-20241022",
    api_key=anthropic_api_key,  # Or set ANTHROPIC_API_KEY
    max_tokens=4096  # Required for Anthropic
)

planner = Agent(
    role="Content Planner",
    goal="Plan engaging and factually accurate content on {topic}",
    backstory="You are working on planning a blog article "
              "about the topic: {topic}."
              "You collect information that helps the "
              "audience learn something "
              "and make informed decisions. "
              "Your work is the basis for "
              "the content writer to write an article on this topic",
    allow_delegation=False,
    verbose=True,
    llm=llm,
    api_key=anthropic_api_key
)

writer = Agent(
    role="Content Writer",
    goal="Write insightful and factually accurate "
         "opinion piece about the topic: {topic}",
    backstory="You are working on writing a new opinion piece"
              "about the topic: {topic}."
              "You base your writing on the work of "
              "the content planner, who provides the outline "
              "and relevant context about the topic. "
              "You follow the main objectives and "
              "directions of the outline, "
              "as provided by the content planner. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provided by the content planner. "
              "You acknowledge in your opinion piece "
              "when your statements are opinions "
              "as opposed to facts.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
    api_key=anthropic_api_key
)   

writer = Agent(
    role="Editor",
    goal="Edit a given blog post to align with "
         "the writing style of the organisation.",
    backstory="You are an editor who receives a blog post "
              "from the content writer. "
              "Your goal is to review the blog post "
              "to ensure that it follows journalistic best practices, "
              "provides balanced view points "
              "when providing opinions or assertions, "
              "and also avoids major controversial topics "
              "or opinions when possible."
    allow_delegation=False,
    verbose=True,
    llm=llm,
    api_key=anthropic_api_key
)

plan = Task(
    description=(
        "1. Prioritise the latest trends, key players, "
        "and noteworthy news on {topic} . \n"
        "2. Identify the target audience, considering "
        "their interests and pain points.\n"
        "3. Develop a detailed content outline including"
        "an introduction, key points, and a call to action.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document "
        "with an outline, audience analysis, "
        "SEO keywords and resources.",
    agent=planner
)

write = Task(
    description=(
        "1. Use the content plan top craft a compelling "
        "blog post on {topic}. \n"
        "2. Incorporate SEO keywords naturally. \n"
        "3. Sections/Subtitles are properly named "
        "in an engaging manner.\n"
        "4. Ensure the post is structured with an "
        "engaging introduction, insightful body, "
        "and a summarizing conclusion.\n"
        "5. Proofread for grammatical errors and "
        "alignment with the brand's voice. \n"
    ),
    expected_output="A well-written blog post that "
        "in markdown format, ready for publication,"
        "each section should have 2 or 3 paragraphs.",
    agent=writer
)

edit = Task(
    description=(
        "Proofread the given blog post for "
        "grammatical errors and "
        "alignment with brand's voice."
    ),
    expected_output="A polished and edited blog post "
        "that adheres to the organisation's "
        "editorial standards and should have 2 or 3 paragraphs.",
    agent=editor

)
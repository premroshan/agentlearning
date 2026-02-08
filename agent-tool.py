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
from crewai_tools import SerperDevTool, DirectoryReadTool, FileReadTool

# Enable extended thinking with default settings
# Configure thinking with budget control
llm = LLM(
    model="anthropic/claude-sonnet-4-5-20250929",  # Latest Sonnet 3.5
    # OR: "anthropic/claude-3-opus-20240229"
    # OR: "anthropic/claude-3-haiku-20240307"
    api_key=anthropic_api_key,
    max_tokens=4096
)

from crewai import Agent, Task, Crew

sales_rep_agent = Agent(
    role="Sales Representative",
    goal="Identify high value leads that match our " \
    "ideal customer profile.",
    backstory="As a part of the dynamic sales team at CrewAI, "
              "your mission is to scour "
              "the digital landscape for potential leads. "
              "Armed with cutting-edge-tools "
              "and a strategic mindset, you analyse data, "
              "trends and interactions to "
              "unearth opportunities that others might overlook. "
              "Your work is crucial in paving the way "
              "for meaningful engagements and driving the company's growth.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
    api_key=anthropic_api_key
)

lead_sales_rep = Agent(
    role = "Lead Sales Representative",
    goal = "Nurture leads with personalized compelling communications",
    backstory = (
        "within the vibrant ecosystem of CrewAI's sales department,"
        "you stand out as the bridge between potential clients "
        "and the solutions they need. "
        "By creating engaging, personalized messages, "
        "you not only inform leads about our offerings "
        "but also make them  feel seen and heard."
        "Your role is pivotal in converting interest "
        "into action, guiding leads through the journey "
        "from curiosity to commitment."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
    api_key=anthropic_api_key
)

directory_read_tool = DirectoryReadTool(directory="./instructions")
file_read_tool = FileReadTool()
search_tool = SerperDevTool()



from utils import SentimentAnalysisTool
    
sentiment_analysis_tool = SentimentAnalysisTool()

lead_profiling_task = Task(
    description=(
        "Conduct in depth analysis of {lead_name}, "
        "a company in the {industry} sector "
        "that recently showed interest in our solutions. "
        "Utilize all available data sources "
        "to complie a detailed profile, "
        "focusing on key decision-makers, recent business "
        "developments and potential needs "
        "that align with our offerings. "
        "This task is crucial for tailoring our outreach and engagement strategy, "
        "Dont make assumptions and "
        "only use information you are absolutely sure about. "
    ),
    expected_output=(
        "A comprehensive report on {lead_name}, "
        "including company background, "
        " key personnel, recent news and developments. "
        "Highlight potential areas where "
        "our solutions can provide value, "
        "and suggest personalized engagement strategies "
    ),
    tools=[
        directory_read_tool,
        file_read_tool,
        search_tool
    ],
    agent = sales_rep_agent
)

personalised_outreach_task = Task(
    description=(
        "Using the insights gathered from "
        "the lead profiling report on {lead_name}, "
        "craft a personalized outreach message, "
        "aimed at {key_decision_maker}, "
        "the {position} at {lead_name}. "
        "The campaign should address their recent {milestone} "
        "and how our solutions can support their goals. "
        "Your communications must resonate "
        "with {lead_name}'s company culture and values, "
        "demonstrating a deep understanding of "
        "their business and needs. \n"
        "Dont make assumptions and only "
        "use information you absolutely sure about. "
    ),
    expected_output=(
        "A series of personalized email drafts "
        "tailored to {lead_name}, "
        "specifically targeting {key_decision_maker}. "
        "Each draft should include "
        "a compelling narrative that connects our solutions "
        "with their recent achievements and future goals. "
        "Ensure the tone is engaging, professional, "
        "and aligned with {lead_name}'s company culture."
    ),
    tools=[
        sentiment_analysis_tool,
        search_tool
    ],
    agent = lead_sales_rep
)

crew = Crew(
    agents = [sales_rep_agent, lead_sales_rep],
    tasks=[lead_profiling_task,
         personalised_outreach_task
     ],
    verbose=True,
    memory=True
)

inputs={
    "lead_name": "Tech Innovators Inc.",
    "industry": "Technology",
    "key_decision_maker": "Jane Doe",
    "position": "Chief Technology Officer",
    "milestone": "recent launch of their new AI product line"
}

result = crew.kickoff(inputs=inputs)
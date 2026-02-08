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
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# Enable extended thinking with default settings
# Configure thinking with budget control
llm = LLM(
    model="anthropic/claude-sonnet-4-5-20250929",  # Latest Sonnet 3.5
    # OR: "anthropic/claude-3-opus-20240229"
    # OR: "anthropic/claude-3-haiku-20240307"
    api_key=anthropic_api_key,
    max_tokens=4096
)

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

venue_coordinator = Agent(
    role="Venue Coordinator",
    goal="Identify and book an appropriate venue "
         "based on event requirements.",
    backstory=(
        "With a keen sense of space and "
        "understanding of event logistics, "
        "you excel at finding and securing "
        "the perfect venue that fits the event's theme, "
        "size, and budget constraints."
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    llm=llm,
    api_key=anthropic_api_key
)

logistics_manager = Agent(
    role="Logistics Manager",
    goal="Manage all logistics for the event "
        "including catering and equipment.",
    backstory=(
        "Organized and detail oriented, "
        "you ensure that every logistical aspect of the event "
        "from catering to equipment setup "
        "is flawlessly executed to create a seamless experience for attendees."
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    llm=llm,
    api_key=anthropic_api_key
)

marketing_communications_agent = Agent(
    role="Marketing and Communications Agent",
    goal="Develop and execute marketing strategies for the event.",
    backstory=(
        "Creative and strategic, you excel at crafting compelling narratives "
        "that resonate with diverse audiences. Your expertise in marketing "
        "ensures that the event is effectively promoted across all channels, "
        "driving engagement and attendance."
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    llm=llm,
    api_key=anthropic_api_key
)

from pydantic import BaseModel

class VenueDetails(BaseModel):
    name: str
    address: str
    capacity: int
    booking_status: str

venue_task = Task(
    description=(
        "Find a venue in {event_city} "
        "that meets criteria for {event_topic}."
    ),
    expected_output=(
        "All the details of a specifically chosen "
        "venue you found to accomodate the event."
    ),
    human_input=True,
    agent=venue_coordinator,
    output_json=VenueDetails,
    output_file="venue_details.json"
)

logistics_task = Task(
    description=(
        "Organize logistics for the event, "
        "including catering and equipment "
        "with {expected_participants} participants on {tentative_date}."
    ),
    expected_output=(
        "Confirmation of all logistics arrangements, "
        "including catering and equipment setup."
    ),
    human_input=True,
    agent=logistics_manager
)

marketing_task = Task(
    description=(
        "Promote the event {event_topic} "
        "aiming to engage at least "
        "{expected_participants} potential attendees."
    ),
    expected_output=(
        "Report on marketing activities "
        "and attendee engagement formatted as markdown."
    ),
    async_execution=True,
    output_file="marketing_report.md",
    agent=marketing_communications_agent
)

event_management_crew = Crew(
    agents=[
        venue_coordinator,
        logistics_manager,
        marketing_communications_agent
    ],
    tasks=[
        venue_task,
        logistics_task,
        marketing_task
    ],
    verbose=True
)

event_details = {
    "event_city": "New York",
    "event_topic": "Tech Innovation Conference",
    "expected_participants": 500,
    "tentative_date": "2024-11-15"
}

result = event_management_crew.kickoff(inputs=event_details)
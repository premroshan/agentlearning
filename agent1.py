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

# Set Ollama embedding model as environment variable
os.environ["EMBEDDINGS_OLLAMA_MODEL_NAME"] = "mxbai-embed-large"

# Enable extended thinking with default settings
# Configure thinking with budget control
llm = LLM(
    model="anthropic/claude-sonnet-4-5-20250929",  # Latest Sonnet 3.5
    # OR: "anthropic/claude-3-opus-20240229"
    # OR: "anthropic/claude-3-haiku-20240307"
    api_key=anthropic_api_key,
    max_tokens=4096
)

support_agent = Agent(
    role="Senior suport representative",
    goal="Be the most friendly and helpful support representative in your team ",
    backstory="You work at crew AI and are now "
              " working on providing support to customer {customer}, a "
              "super important customer for your company. You "
              "need to make sure that you provide the best support!"
              "Make sure to provide full complete answers, "
              " and make no assumptions.",
    allow_delegation=False,
    verbose=True,
    llm=llm,
    api_key=anthropic_api_key
)

support_quality_assurance_agent = Agent(
    role="Support quality assurance specialist",
    goal="Get recognition for providing the "
         "best support quality assurance in your team",
    backstory="You work at crew AI and "
        "are now working with your team "
        "on a request from customer {customer} ensuring that "
        " the support representative is providing the best support possible. "
        "You need to make sure that the support representative is providing "
        "full complete answers and make no assumptions.",
    verbose=True,
    llm=llm,
    api_key=anthropic_api_key
)   

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/"
)

inquiry_resolution = Task(
    description=(
    "{customer} just reached out with the following inquiry: {inquiry}. \n"
    "{person} from {customer} is the one that reached out. \n"
    "Make sure to use everything you know "
    "to provide the best support possible. "
    "You must strive to provide a complete "
    "and accurate response to the customer's inquiry. "
    ),
    expected_output=(
        "A detailed informative response to the "
        "customer's inquiry that addresses "
        "all aspects of the inquiry. \n"
        "The response should include references "
        "to everything you used to find the answer, "
        "including external data or solutions. "
        "Ensure the answer is complete, "
        "leaving no questions unanswered and maintain a helpful and "
        "friendly tone throughout the response."
    ),
    tools=[docs_scrape_tool],
    agent=support_agent
)  

quality_assurance_review = Task(
    description=(
    "Review the response provided by the Senior support representative. \n"
    "Ensure that the answer is comprehensive, accurate and adheres to "
    "high quality standards expected for customer support. \n"
    "Verify that all parts of the customer's inquiry have been addressed "
    "thoroughly with a helpful and friendly tone. \n"
    "Check for references an sources used to "
    "find the information, "
    "ensuring the response is well-supported and "
    "leaves no questions unanswered."
    ),
    expected_output=(
        "A final, detailed and informative response "
        "ready to be sent to the customer.\n"
        "This reponse should fully address the "
        "customer's inquiry incorporating all "
        "relevant feedback and improvements.\n"
        "Dont be too formal, we are a chill and cool company "
        "but maintain a professional and friendly tone throughout."
    ),
    agent=support_quality_assurance_agent
)   

crew = Crew(
    agents=[
        support_agent,
        support_quality_assurance_agent
    ],
    tasks=[
        inquiry_resolution,
        quality_assurance_review
    ],
    verbose=True,
    memory=True,
    embedder={
        "provider": "ollama",
        "config": {
            "model": "mxbai-embed-large"
        }
    }
)

inputs = {
    "customer": "Acme Corporation",
    "person": "John Doe",
    "inquiry": (
        "Hi, I'm having trouble integrating Crew AI with our existing "
        "workflow. Can you provide guidance on how to set it up and "
        "ensure it works smoothly with our current tools?"
    )
}

results = crew.kickoff(inputs=inputs)


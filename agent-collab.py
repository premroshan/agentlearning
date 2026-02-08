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

data_analyst_agent = Agent(
    role = "Data Analyst",
    goal = "Monitor and analyze the market data in real-time to identify trends and insights that "
    "can inform investment strategies.",
    backstory = (
        "Specializing in financial markets, this agent uses "
        "stats and data analysis to identify trends and insights that can inform investment strategies."
    ),
    verbose=True,
    aloow_delegation=True,
    tools=[search_tool, scrape_tool],
    llm=llm,
    api_key=anthropic_api_key
)

trading_strategy_agent = Agent(
    role = "Trading Strategy Developer",
    goal = "Develop and test trading strategies based on the insights provided by the data analyst agent.",
    backstory = (
        "Equipped with a deep understanding of financial markets and trading algorithms, this agent crafts "
        "and evaluates trading strategies that leverage the data insights provided by the data analyst agent."
    ),
    verbose=True,
    allow_delegation=True,
    tools=[search_tool, scrape_tool],
    llm=llm,
    api_key=anthropic_api_key
)

risk_management_agent = Agent(
    role = "Risk Management Specialist",
    goal = "Identify and mitigate risks associated with trading strategies.",
    backstory = (
        "Armed with a deep understanding of risk assessment models "
        "and financial regulations, this agent evaluates the potential risks of trading strategies and implements measures to mitigate them, ensuring that the trading activities are compliant and financially sound."
    ),
    verbose=True,
    allow_delegation=True,
    tools=[search_tool, scrape_tool],
    llm=llm,
    api_key=anthropic_api_key
)

data_analysis_task = Task(
    description=(
        "Continuously monitor and analyze market data to identify trends and insights that can inform trading strategies."
        "Use statistical modeling and machine learning to identify trends and insights in the market data. "
    ),
    expected_output=(
        "Insights and alerts about significant market trends, patterns, and anomalies that can inform trading strategies. "
    ),
    agent=data_analyst_agent
)

strategy_development_task = Task(
    description=(
        "Develop and refine trading strategies based on the insights provided by the data analyst agent. "
        "Use backtesting and simulation to evaluate the performance of trading strategies under different market conditions. "
        "Use user defined risk tolerance {risk_tolerance} to ensure that the strategies align with the risk management guidelines.  "
    ),
    expected_output=(
        "A set of trading strategies for {stock_selection} that have been developed and tested based on the insights provided by the data analyst agent, along with performance metrics and recommendations for implementation."
    ),
    agent=trading_strategy_agent
)

execution_planning_task = Task(
    description=(
        "Analyse approved trading strategies and develop a detailed execution plan for implementing them. "
        "Consider factors such as market conditions, liquidity, and transaction costs to optimize the execution of trades. "
    ),
    expected_output=(
        "A detailed execution plan suggesting how and when to implement the approved trading strategies, including recommendations for order types, timing, and risk management measures to optimize trade execution and minimize costs."
    ),
    agent=risk_management_agent
)

from crewai import Crew, Process

financial_trading_crew = Crew(
    agents=[data_analyst_agent, trading_strategy_agent, risk_management_agent],
    tasks=[data_analysis_task, strategy_development_task, execution_planning_task],
    manager_llm=llm,
    process=Process.hierarchical,
    verbose=True
)

financial_trading_inputs = {
    "stock_selection": "technology stocks",
    "risk_tolerance": "moderate"
}

results = financial_trading_crew.kickoff(inputs=financial_trading_inputs)
import os

def get_anthropic_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return key

from crewai.tools import BaseTool

class SentimentAnalysisTool(BaseTool):
    name: str = "Sentiment Analysis Tool"
    description: str = ("Analyzes the sentiment of a given text "
                        "to ensure positive and engaging communication with leads. ")

    def _run(self, text: str) -> str:
        # Placeholder for sentiment analysis logic
        # In a real implementation, this would call an API or use a model to analyze sentiment
        return "Positive"
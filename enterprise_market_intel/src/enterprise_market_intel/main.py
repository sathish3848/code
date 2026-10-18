import os
from dotenv import load_dotenv
from src.enterprise_market_intel.crew import EnterpriseMarketIntelCrew

# Load API keys from .env
load_dotenv()


def run():
    """
    Run the crew with user inputs.
    """
    inputs = {"topic": "AI Code Assistants in Software Engineering"}

    print("Starting Enterprise Market Intelligence Crew...")
    result = EnterpriseMarketIntelCrew().crew().kickoff(inputs=inputs)
    print("\n--- FINAL REPORT ---")
    print(result)


if __name__ == "__main__":
    run()

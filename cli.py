"""Command Line Interface"""
import argparse
from dotenv import find_dotenv, load_dotenv
import os
from src.orchestrator import AgentOrchestrator

load_dotenv()

STYLE_CONFIG = {
    "colors": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"],
    "figure_size": (10, 6),
    "dpi": 100
}


load_dotenv(find_dotenv())

def main():
    parser = argparse.ArgumentParser(description="AI Data Analysis Agents")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--database", "-d", default="chinook", help="Database name")
    parser.add_argument("--role", "-r", default="analyst", help="User role")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator(
        openai_api_key=os.getenv("OPEENAI_API_KEY"),
        style_config=STYLE_CONFIG
    )
    
    # Process request
    result = orchestrator.process_request(
        query=args.query,
        database=args.database,
        user_role=args.role,
        create_viz=not args.no_viz
    )
    
    if result["status"] == "success":
        print(f"\n📊 Results:\n{result['data'].head()}")
    else:
        print(f"\n❌ Error: {result.get('error')}")

if __name__ == "__main__":
    main()
"""Command Line Interface - Updated for Production"""
import argparse
from dotenv import find_dotenv, load_dotenv
import os
from src.langgraph_orchestrator import LangGraphOrchestrator
import sys
import time

load_dotenv(find_dotenv())

# Style configuration for visualizations
STYLE_CONFIG = {
    "colors": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"],
    "figure_size": (10, 6),
    "dpi": 100
}

session_id = f"cli-{int(time.time())}"

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AI Data Analysis Agent with LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
        %(prog)s "Show top 5 artists by sales" --database chinook
        %(prog)s "Monthly revenue for 2009" -d chinook -r analyst
        %(prog)s "List all customers" -d northwind --no-viz
        
        Interactive mode:
        %(prog)s -i --database chinook
        %(prog)s --interactive --session my-session """)
    
    parser.add_argument(
        "query",
        nargs="?",  # NEW: Make optional for interactive mode
        help="Natural language query to analyze"
    )
    parser.add_argument(
        "--database", "-d", 
        default="chinook", 
        help="Database name (default: chinook)"
    )
    parser.add_argument(
        "--role", "-r", 
        default="analyst", 
        choices=["admin", "analyst", "viewer"],
        help="User role for permissions (default: analyst)"
    )
    parser.add_argument(
        "--no-viz", 
        action="store_true", 
        help="Skip visualization creation"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY env variable)"
    )
    parser.add_argument(
        "--api-base",
        default="https://chat.int.bayer.com/api/v2",
        help="Custom OpenAI API base URL"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--session", "-s",
        help="Session ID for conversation continuity"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode for multi-turn conversations"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Configure logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize orchestrator
        print(f"\n Initializing AI Data Analysis Agent...")
        print(f"   Database: {args.database}")
        print(f"   User Role: {args.role}")
        print(f"   Visualization: {'Disabled' if args.no_viz else 'Enabled'}")
        
        orchestrator = LangGraphOrchestrator(
            openai_api_key=api_key,
            style_config=STYLE_CONFIG,
            openai_api_base=args.api_base
        )
        
        session_id = args.session or f"cli-{int(time.time())}"
        
        if args.interactive:
            print(f"   Session: {session_id}")
            print(f"\n Interactive Mode - Type 'exit' or 'quit' to end session\n")
            
            while True:
                try:
                    query = input("You: ").strip()
                    
                    if query.lower() in ['exit', 'quit', 'q']:
                        print("\nGoodbye!")
                        break
                    
                    if not query:
                        continue
                    
                    print() 
                    
                    result = orchestrator.process_request(
                        query=query,
                        database=args.database,
                        user_role=args.role,
                        create_viz=not args.no_viz,
                        session_id=session_id
                    )
                    
                    display_result(result)
                    
                except KeyboardInterrupt:
                    print("\n\nSession ended")
                    break
                except Exception as e:
                    print(f"\nError: {str(e)}\n")
        
        else:
            if not args.query:
                print("Error: Query required in non-interactive mode. Use -i for interactive mode.")
                sys.exit(1)
            
            print(f"\n Processing query: {args.query}\n")
            
            result = orchestrator.process_request(
                query=args.query,
                database=args.database,
                user_role=args.role,
                create_viz=not args.no_viz,
                session_id=session_id
            )
            
            display_result(result)
            
    except Exception as e:
        print(f"\n Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def display_result(result):
    """Display results (factored out for reuse)"""
    if result.status == "success":
        print(f"\n{'='*60}")
        print(" SUCCESS")
        print(f"{'='*60}\n")
        
        if result.intent == "off_topic":
            print(f" {result.message}\n")
        else:
            # SQL Query
            if result.sql_query:
                print("SQL Query:")
                print(f"   {result.sql_query}\n")
            
            # Data Summary
            print(f" Results: {result.row_count} rows returned")
            
            if result.data_preview and result.row_count > 0:
                import pandas as pd
                import json
                df = pd.DataFrame(json.loads(result.data_preview))
                print("\n   Preview (first 10 rows):")
                print(df.to_string(index=False).replace('\n', '\n   '))
            
            # Insights
            if result.insights:
                print(f"\nKey Insights:")
                print(f"   {result.insights.summary}\n")
                for i, finding in enumerate(result.insights.key_findings, 1):
                    print(f"   {i}. {finding}")
            
            # Visualization
            if result.visualization_path:
                print(f"\nVisualization saved: {result.visualization_path}")
                print(f"   Chart type: {result.chart_type}")
        
        # Execution time
        if result.execution_time_seconds:
            print(f"\n Execution time: {result.execution_time_seconds:.2f}s")
        
        print(f"\n{'='*60}\n")
        
    else:
        print(f"\n{'='*60}")
        print(" ERROR!!")
        print(f"{'='*60}\n")
        print(f"   {result.error}\n")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

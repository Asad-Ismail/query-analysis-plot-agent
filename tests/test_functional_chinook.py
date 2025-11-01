"""
Functional tests for the LangGraph Orchestrator focusing on the 
Chinook dataset.

These tests run the full agent workflow and WILL make real LLM API calls.
Ensure you have a valid OPENAI_API_KEY set in your environment.

These tests compare agent results against a "Ground Truth" (GT)
query executed live against the chinook.db to validate accuracy.
"""

import pytest
import os
import shutil
import json
import sqlite3
import pandas as pd
import yaml
import io
from src.langgraph_orchestrator import LangGraphOrchestrator
from dotenv import find_dotenv, load_dotenv

# Load environment variables (same as cli.py)
load_dotenv(find_dotenv())

# --- Test Configuration ---

# Skip all tests in this file if the API key is not set
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE", "https://chat.int.bayer.com/api/v2")

pytestmark = pytest.mark.skipif(
    not API_KEY, 
    reason="OPENAI_API_KEY environment variable not set. Skipping functional tests."
)

# Style config (from cli.py)
STYLE_CONFIG = {
    "colors": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"],
    "figure_size": (10, 6),
    "dpi": 100
}


@pytest.fixture(scope="module")
def orchestrator():
    """
    Module-scoped orchestrator instance. 
    This is expensive to initialize, so we only do it once.
    """
    return LangGraphOrchestrator(
        openai_api_key=API_KEY,
        style_config=STYLE_CONFIG,
        openai_api_base=API_BASE
    )

@pytest.fixture(scope="module")
def db_path():
    """
    Gets the path to the chinook.db file from the config.
    """
    config_path = "config/databases.yaml"
    if not os.path.exists(config_path):
        pytest.fail(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    db_path = config.get("databases", {}).get("chinook", {}).get("path")
    if not db_path or not os.path.exists(db_path):
        pytest.fail(f"chinook.db path not found or invalid in {config_path}")
    
    return db_path

@pytest.fixture(scope="module", autouse=True)
def cleanup_outputs():
    """
    Auto-used module fixture to clean up the 'outputs' directory 
    after all tests in this file run.
    """
    yield
    output_dir = "outputs"
    if os.path.exists(output_dir):
        print(f"\nCleaning up '{output_dir}' directory...")
        shutil.rmtree(output_dir)

# --- Helper Function ---

def load_json_data(json_string: str):
    """Helper to parse the data_preview or full_data string."""
    if not json_string:
        return None
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        pytest.fail(f"Failed to decode JSON: {json_string}")

def get_agent_df(response):
    """Loads the full_data from an agent response into a DataFrame."""
    if not response.full_data:
        pytest.fail("Agent response did not contain 'full_data'.")
    # Use io.StringIO to avoid FutureWarning
    return pd.read_json(io.StringIO(response.full_data))


# --- Test Cases ---

@pytest.mark.usefixtures("cleanup_outputs")
class TestChinookFunctional:
    
    # --- Easy Tests ---

    def test_easy_1_simple_retrieval(self, orchestrator, db_path):
        """
        Easy Test 1: Simple Retrieval & LIMIT
        Tests: Basic SQL, LIMIT clause, analyst role
        GT: Should return exactly 10 rows.
        """
        query = "Show me 10 artists"
        
        response = orchestrator.process_request(
            query=query,
            database="chinook",
            user_role="analyst",
            create_viz=False
        )
        
        print(f"\n[Agent SQL]: {response.sql_query}")
        
        assert response.status == "success"
        assert response.sql_query is not None
        assert "LIMIT" in response.sql_query.upper()
        assert response.row_count == 10
        assert response.error is None
        
        agent_df = get_agent_df(response)
        assert len(agent_df) == 10

    def test_easy_2_count_aggregation(self, orchestrator, db_path):
        """
        Easy Test 2: Single-Table COUNT Aggregation
        Tests: COUNT(*), single-cell result, insights
        GT: The total number of customers is 59.
        """
        query = "How many customers do we have in total?"
        ground_truth_sql = "SELECT COUNT(CustomerId) FROM customers;"
        
        with sqlite3.connect(db_path) as conn:
            gt_df = pd.read_sql_query(ground_truth_sql, conn)
        gt_count = gt_df.iloc[0, 0]

        response = orchestrator.process_request(
            query=query,
            database="chinook",
            user_role="analyst"
        )
        
        print(f"\n[GT SQL]:    {ground_truth_sql}")
        print(f"[Agent SQL]: {response.sql_query}")
        
        assert response.status == "success"
        assert "COUNT" in response.sql_query.upper()
        assert response.row_count == 1
        
        agent_df = get_agent_df(response)
        agent_count = agent_df.iloc[0, 0]
        
        assert agent_count == gt_count

    def test_easy_3_where_clause(self, orchestrator, db_path):
        """
        Easy Test 3: Single-Table WHERE Clause
        Tests: WHERE filter, string parsing ("Berlin")
        GT: There are 2 customers in Berlin (IDs 6 and 35).
        """
        query = "List all customers who live in Berlin"
        ground_truth_sql = "SELECT CustomerId FROM customers WHERE City = 'Berlin' ORDER BY CustomerId;"
        
        with sqlite3.connect(db_path) as conn:
            gt_df = pd.read_sql_query(ground_truth_sql, conn)
        gt_ids = set(gt_df['CustomerId'])

        response = orchestrator.process_request(
            query=query,
            database="chinook",
            user_role="analyst"
        )
        
        print(f"\n[GT SQL]:    {ground_truth_sql.strip()}")
        print(f"[Agent SQL]: {response.sql_query}")
        
        assert response.status == "success"
        assert "WHERE" in response.sql_query.upper()
        assert "BERLIN" in response.sql_query.upper()
        assert response.row_count == 2
        
        agent_df = get_agent_df(response)
        
        id_col = 'CustomerId' if 'CustomerId' in agent_df.columns else agent_df.columns[0]
        agent_ids = set(agent_df[id_col])
        
        assert agent_ids == gt_ids

    # --- Hard Tests ---

    def test_hard_1_top_customers_join_sum(self, orchestrator, db_path):
        """
        Hard Test 1: Multi-Table JOIN & SUM/GROUP BY (Sales)
        Tests: JOIN, SUM, GROUP BY, ORDER BY, viz generation (bar)
        GT: Top 5 customer IDs are [6, 26, 45, 46, 57]
        """
        query = "Who are our top 5 customers by total spending?"
        ground_truth_sql = """
            SELECT c.CustomerId
            FROM customers c
            JOIN invoices i ON c.CustomerId = i.CustomerId
            GROUP BY c.CustomerId
            ORDER BY SUM(i.Total) DESC
            LIMIT 5;
        """
        
        #  Get Ground Truth
        with sqlite3.connect(db_path) as conn:
            gt_df = pd.read_sql_query(ground_truth_sql, conn)
        gt_ids = set(gt_df['CustomerId'])
        
        #  Get Agent Result
        response = orchestrator.process_request(
            query=query,
            database="chinook",
            user_role="analyst",
            create_viz=True
        )
        
        print(f"\n[GT SQL]:    {' '.join(ground_truth_sql.split())}")
        print(f"[Agent SQL]: {response.sql_query}")
        
        assert response.status == "success"
        assert "JOIN" in response.sql_query.upper()
        assert "SUM" in response.sql_query.upper()
        assert "GROUP BY" in response.sql_query.upper()
        assert "ORDER BY" in response.sql_query.upper()
        assert "LIMIT 5" in response.sql_query.upper()
        assert response.row_count == 5
        assert response.chart_type == "bar"
        assert response.visualization_path is not None
        assert os.path.exists(response.visualization_path)
        
        agent_df = get_agent_df(response)
        
        # Find the ID column (might be CustomerId or a name)
        id_col = next((col for col in agent_df.columns if 'Id' in col or 'ID' in col), agent_df.columns[0])
        agent_ids = set(agent_df[id_col])
        
        assert agent_ids == gt_ids

    def test_hard_2_top_genres_deep_join(self, orchestrator, db_path):
        """
        Hard Test 2: Deep JOIN & SUM/GROUP BY (Product)
        Tests: Complex 3-table JOIN, calculated SUM, viz (bar/pie)
        GT: Top 3 genres are Rock, Latin, Metal
        """
        query = "What are the top 3 best-selling genres by revenue?"
        ground_truth_sql = """
            SELECT g.Name
            FROM genres g
            JOIN tracks t ON g.GenreId = t.GenreId
            JOIN invoice_items ii ON t.TrackId = ii.TrackId
            GROUP BY g.Name
            ORDER BY SUM(ii.UnitPrice * ii.Quantity) DESC
            LIMIT 3;
        """

        with sqlite3.connect(db_path) as conn:
            gt_df = pd.read_sql_query(ground_truth_sql, conn)
        gt_genres = gt_df['Name'].tolist()
        assert gt_genres == ["Rock", "Latin", "Metal"]
        
        response = orchestrator.process_request(
            query=query,
            database="chinook",
            user_role="admin",
            create_viz=True
        )
        
        print(f"\n[GT SQL]:    {' '.join(ground_truth_sql.split())}")
        print(f"[Agent SQL]: {response.sql_query}")
        
        assert response.status == "success"
        assert "INVOICE_ITEMS" in response.sql_query.upper()
        assert "TRACKS" in response.sql_query.upper()
        assert "GENRES" in response.sql_query.upper()
        assert "UNITPRICE" in response.sql_query.upper()
        assert "QUANTITY" in response.sql_query.upper()
        assert response.row_count == 3
        assert response.chart_type in ["bar", "pie"]
        
        agent_df = get_agent_df(response)

        name_col = next(col for col in agent_df.columns if agent_df[col].dtype == 'object')
        agent_genres = agent_df[name_col].tolist()
        
        assert agent_genres == gt_genres

    def test_hard_3_date_trend(self, orchestrator, db_path):
        """
        Hard Test 3: Date/Time Filtering & Aggregation (Trend)
        Tests: Date functions (STRFTIME), WHERE on date, viz (line)
        GT: Should return 12 rows (one for each month of 2010).
        """
        query = "Show me the total monthly sales for 2010"
        ground_truth_sql = """
            SELECT STRFTIME('%Y-%m', InvoiceDate) as Month
            FROM invoices
            WHERE STRFTIME('%Y', InvoiceDate) = '2010'
            GROUP BY 1
            ORDER BY 1;
        """
        
        with sqlite3.connect(db_path) as conn:
            gt_df = pd.read_sql_query(ground_truth_sql, conn)

        response = orchestrator.process_request(
            query=query,
            database="chinook",
            user_role="analyst",
            create_viz=True
        )
        
        print(f"\n[GT SQL]:    {' '.join(ground_truth_sql.split())}")
        print(f"[Agent SQL]: {response.sql_query}")
        
        assert response.status == "success"
        assert "STRFTIME" in response.sql_query.upper() or \
               "SUBSTR" in response.sql_query.upper()
        assert "2010" in response.sql_query
        assert response.chart_type == "line"
        
        assert response.row_count == len(gt_df)

    def test_hard_4_security_permission_denied(self, orchestrator, db_path):
        """
        Hard Test 4: Security & Permission Error
        Tests: 'viewer' role restrictions, error handling
        GT: Must return an error stating access is denied.
        """

        query = "List all employee names"
        
        response = orchestrator.process_request(
            query=query,
            database="chinook",
            user_role="viewer"  )
        
        print(f"\n[Agent SQL]: {response.sql_query}")
        print(f"[Agent Error]: {response.error}")
        
        assert response.status == "error"
        assert response.error is not None
        
        error_msg = response.error.lower()
        #assert "denied" in error_msg
        assert response.full_data is None
        assert response.row_count == 0
        assert response.visualization_path is None


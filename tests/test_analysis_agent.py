"""Tests for refactored Analysis Agent with structured outputs"""
import pytest
import pandas as pd
import sqlite3
from unittest.mock import Mock, MagicMock, patch
from langchain_openai import ChatOpenAI
from src.analysis_agent import DataAnalysisAgent
from src.models import SQLQueryOutput, InsightsOutput


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing"""
    return Mock(spec=ChatOpenAI)


@pytest.fixture
def analysis_agent(mock_llm):
    """Create analysis agent with mock LLM"""
    return DataAnalysisAgent(mock_llm)


@pytest.fixture
def sample_schema():
    """Sample database schema"""
    return {
        "artists": ["ArtistId", "Name"],
        "albums": ["AlbumId", "Title", "ArtistId"],
        "tracks": ["TrackId", "Name", "AlbumId", "GenreId"]
    }


@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database with sample data"""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    # Create artists table
    cursor.execute("""
        CREATE TABLE artists (
            ArtistId INTEGER PRIMARY KEY,
            Name TEXT
        )
    """)
    
    # Insert sample data
    artists_data = [
        (1, "AC/DC"),
        (2, "Accept"),
        (3, "Aerosmith")
    ]
    cursor.executemany("INSERT INTO artists VALUES (?, ?)", artists_data)
    
    conn.commit()
    return conn


class TestDataAnalysisAgent:
    """Test suite for DataAnalysisAgent"""
    
    def test_initialization(self, mock_llm):
        """Test agent initializes correctly"""
        agent = DataAnalysisAgent(mock_llm)
        assert agent.llm == mock_llm
        assert len(agent.DANGEROUS_KEYWORDS) > 0
    
    def test_is_safe_query_with_select(self, analysis_agent):
        """Test safe SELECT query validation"""
        safe_queries = [
            "SELECT * FROM artists",
            "SELECT name, id FROM albums LIMIT 10",
            "  SELECT COUNT(*) FROM tracks  "
        ]
        
        for query in safe_queries:
            assert analysis_agent._is_safe_query(query) is True
    
    def test_is_safe_query_with_dangerous_keywords(self, analysis_agent):
        """Test that dangerous queries are rejected"""
        dangerous_queries = [
            "DROP TABLE artists",
            "DELETE FROM albums",
            "UPDATE tracks SET name='hacked'",
            "INSERT INTO artists VALUES (99, 'test')",
            "ALTER TABLE albums ADD COLUMN test TEXT",
            "SELECT * FROM artists; DROP TABLE albums;"
        ]
        
        for query in dangerous_queries:
            assert analysis_agent._is_safe_query(query) is False
    
    def test_is_safe_query_non_select(self, analysis_agent):
        """Test that non-SELECT statements are rejected"""
        assert analysis_agent._is_safe_query("SHOW TABLES") is False
        assert analysis_agent._is_safe_query("PRAGMA table_info(artists)") is False
    
    def test_validate_table_access_allowed(self, analysis_agent):
        """Test table access validation when tables are allowed"""
        tables_used = ["artists", "albums"]
        allowed_tables = ["artists", "albums", "tracks"]
        
        assert analysis_agent._validate_table_access(tables_used, allowed_tables) is True
    
    def test_validate_table_access_denied(self, analysis_agent):
        """Test table access validation when tables are not allowed"""
        tables_used = ["artists", "invoices"]
        allowed_tables = ["artists", "albums", "tracks"]
        
        assert analysis_agent._validate_table_access(tables_used, allowed_tables) is False
    
    def test_format_schema(self, analysis_agent, sample_schema):
        """Test schema formatting for prompts"""
        formatted = analysis_agent._format_schema(sample_schema)
        
        assert "Table: artists" in formatted
        assert "Columns: ArtistId, Name" in formatted
        assert "Table: albums" in formatted
    
    def test_prepare_data_summary(self, analysis_agent):
        """Test data summary preparation"""
        df = pd.DataFrame({
            "name": ["Artist1", "Artist2", "Artist3"],
            "sales": [100, 200, 150],
            "rating": [4.5, 3.8, 4.2]
        })
        
        summary = analysis_agent._prepare_data_summary(df)
        
        assert "Total Rows: 3" in summary
        assert "Total Columns: 3" in summary
        assert "sales" in summary
        assert "min=" in summary
        assert "max=" in summary
    
    @patch('src.analysis_agent_v2.pd.read_sql_query')
    def test_analyze_success_flow(self, mock_read_sql, analysis_agent, sample_schema, in_memory_db):
        """Test successful analysis workflow"""
        # Mock SQL generation
        mock_sql_output = SQLQueryOutput(
            sql_query="SELECT * FROM artists LIMIT 10",
            explanation="Retrieves all artists with a limit",
            tables_used=["artists"]
        )
        
        # Mock insights generation
        mock_insights = InsightsOutput(
            key_findings=["Finding 1", "Finding 2"],
            summary="Test summary"
        )
        
        # Setup mock LLM to return structured outputs
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.side_effect = [mock_sql_output, mock_insights]
        analysis_agent.llm.with_structured_output = Mock(return_value=mock_structured_llm)
        
        # Mock DataFrame result
        mock_df = pd.DataFrame({"Name": ["AC/DC", "Accept"]})
        mock_read_sql.return_value = mock_df
        
        # Execute analysis
        result = analysis_agent.analyze(
            query="Show all artists",
            conn=in_memory_db,
            schema=sample_schema,
            allowed_tables=["artists"]
        )
        
        # Assertions
        assert result["status"] == "success"
        assert result["sql"] == "SELECT * FROM artists LIMIT 10"
        assert result["row_count"] == 2
        assert "insights" in result
    
    @patch('src.analysis_agent_v2.pd.read_sql_query')
    def test_analyze_unsafe_query(self, mock_read_sql, analysis_agent, sample_schema, in_memory_db):
        """Test that unsafe queries are rejected"""
        # Mock SQL generation with unsafe query
        mock_sql_output = SQLQueryOutput(
            sql_query="DROP TABLE artists",
            explanation="This should be blocked",
            tables_used=["artists"]
        )
        
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_sql_output
        analysis_agent.llm.with_structured_output = Mock(return_value=mock_structured_llm)
        
        # Execute analysis
        result = analysis_agent.analyze(
            query="Delete all artists",
            conn=in_memory_db,
            schema=sample_schema,
            allowed_tables=["artists"]
        )
        
        # Assertions
        assert result["status"] == "error"
        assert "unsafe" in result["error"].lower()
        mock_read_sql.assert_not_called()  # Should not execute query
    
    @patch('src.analysis_agent_v2.pd.read_sql_query')
    def test_analyze_permission_denied(self, mock_read_sql, analysis_agent, sample_schema, in_memory_db):
        """Test that permission restrictions are enforced"""
        # Mock SQL generation using unauthorized table
        mock_sql_output = SQLQueryOutput(
            sql_query="SELECT * FROM invoices",
            explanation="Query on invoices table",
            tables_used=["invoices"]
        )
        
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_sql_output
        analysis_agent.llm.with_structured_output = Mock(return_value=mock_structured_llm)
        
        # Execute analysis with limited permissions
        result = analysis_agent.analyze(
            query="Show invoices",
            conn=in_memory_db,
            schema=sample_schema,
            allowed_tables=["artists", "albums"]  # No invoices access
        )
        
        # Assertions
        assert result["status"] == "error"
        assert "denied" in result["error"].lower()
        mock_read_sql.assert_not_called()
    
    @patch('src.analysis_agent_v2.pd.read_sql_query')
    def test_analyze_database_error(self, mock_read_sql, analysis_agent, sample_schema, in_memory_db):
        """Test handling of database errors"""
        # Mock SQL generation
        mock_sql_output = SQLQueryOutput(
            sql_query="SELECT * FROM nonexistent_table",
            explanation="Query on non-existent table",
            tables_used=["nonexistent_table"]
        )
        
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_sql_output
        analysis_agent.llm.with_structured_output = Mock(return_value=mock_structured_llm)
        
        # Mock database error
        mock_read_sql.side_effect = pd.errors.DatabaseError("Table not found")
        
        # Execute analysis
        result = analysis_agent.analyze(
            query="Show data from nowhere",
            conn=in_memory_db,
            schema=sample_schema,
            allowed_tables=None
        )
        
        # Assertions
        assert result["status"] == "error"
        assert "database error" in result["error"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

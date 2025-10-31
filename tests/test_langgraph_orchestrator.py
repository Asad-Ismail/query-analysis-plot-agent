"""Tests for LangGraph Orchestrator"""
import pytest
import os
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from src.langgraph_orchestrator import LangGraphOrchestrator
from src.models import AgentResponse


@pytest.fixture
def style_config():
    """Sample style configuration"""
    return {
        "colors": ["#2E86AB", "#A23B72"],
        "figure_size": (8, 5),
        "dpi": 80
    }


@pytest.fixture
def test_db_path(tmp_path):
    """Create a test database"""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create test table
    cursor.execute("""
        CREATE TABLE artists (
            ArtistId INTEGER PRIMARY KEY,
            Name TEXT
        )
    """)
    
    # Insert test data
    test_data = [
        (1, "AC/DC"),
        (2, "Accept"),
        (3, "Aerosmith")
    ]
    cursor.executemany("INSERT INTO artists VALUES (?, ?)", test_data)
    
    conn.commit()
    conn.close()
    
    return str(db_path)


@pytest.fixture
def mock_config_files(tmp_path, test_db_path):
    """Create mock config files"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    # Create databases.yaml
    databases_yaml = config_dir / "databases.yaml"
    databases_yaml.write_text(f"""databases:
  test:
    path: "{test_db_path}"
    type: "sqlite"
    description: "Test database"
""")
    
    # Create permissions.yaml
    permissions_yaml = config_dir / "permissions.yaml"
    permissions_yaml.write_text("""roles:
  admin:
    test:
      tables: null
  analyst:
    test:
      tables: ["artists"]
  viewer:
    test:
      tables: ["artists"]
""")
    
    return str(config_dir)


class TestLangGraphOrchestrator:
    """Test suite for LangGraph Orchestrator"""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.langgraph_orchestrator.ChatOpenAI')
    @patch('src.langgraph_orchestrator.DatabaseManager')
    @patch('src.langgraph_orchestrator.PermissionManager')
    def test_initialization(self, mock_perm_class, mock_db_class, mock_openai, style_config):
        """Test orchestrator initializes correctly"""
        orchestrator = LangGraphOrchestrator(
            openai_api_key="test-key",
            style_config=style_config
        )
        
        assert orchestrator.llm is not None
        assert orchestrator.analysis_agent is not None
        assert orchestrator.viz_agent is not None
        assert orchestrator.graph is not None
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.langgraph_orchestrator.ChatOpenAI')
    @patch('src.langgraph_orchestrator.DatabaseManager')
    @patch('src.langgraph_orchestrator.PermissionManager')
    def test_check_permissions_node_success(
        self, 
        mock_perm_class, 
        mock_db_class, 
        mock_openai, 
        style_config
    ):
        """Test permission checking node"""
        # Setup mocks
        mock_perm_instance = Mock()
        mock_perm_instance.get_allowed_tables.return_value = ["artists"]
        mock_perm_class.return_value = mock_perm_instance
        
        mock_db_instance = Mock()
        mock_db_instance.get_schema.return_value = {"artists": ["ArtistId", "Name"]}
        mock_db_class.return_value = mock_db_instance
        
        orchestrator = LangGraphOrchestrator(
            openai_api_key="test-key",
            style_config=style_config
        )
        
        # Test state
        state = {
            "user_query": "Test",
            "database": "test",
            "user_role": "analyst",
            "status": "pending"
        }
        
        # Execute node
        result = orchestrator._check_permissions_node(state)
        
        assert "allowed_tables" in result
        assert "database_schema" in result
        assert result["status"] == "pending"  # Should not change to error
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.langgraph_orchestrator.ChatOpenAI')
    @patch('src.langgraph_orchestrator.DatabaseManager')
    @patch('src.langgraph_orchestrator.PermissionManager')
    def test_routing_after_permissions_error(
        self, 
        mock_perm_class, 
        mock_db_class, 
        mock_openai, 
        style_config
    ):
        """Test routing after permission error"""
        orchestrator = LangGraphOrchestrator(
            openai_api_key="test-key",
            style_config=style_config
        )
        
        # State with error
        state = {"status": "error"}
        route = orchestrator._route_after_permissions(state)
        assert route == "error"
        
        # State without error
        state = {"status": "pending"}
        route = orchestrator._route_after_permissions(state)
        assert route == "continue"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.langgraph_orchestrator.ChatOpenAI')
    @patch('src.langgraph_orchestrator.DatabaseManager')
    @patch('src.langgraph_orchestrator.PermissionManager')
    def test_routing_after_insights_with_viz(
        self, 
        mock_perm_class, 
        mock_db_class, 
        mock_openai, 
        style_config
    ):
        """Test routing after insights - with visualization"""
        orchestrator = LangGraphOrchestrator(
            openai_api_key="test-key",
            style_config=style_config
        )
        
        state = {
            "create_visualization": True,
            "row_count": 10
        }
        route = orchestrator._route_after_insights(state)
        assert route == "visualize"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.langgraph_orchestrator.ChatOpenAI')
    @patch('src.langgraph_orchestrator.DatabaseManager')
    @patch('src.langgraph_orchestrator.PermissionManager')
    def test_routing_after_insights_skip_viz(
        self, 
        mock_perm_class, 
        mock_db_class, 
        mock_openai, 
        style_config
    ):
        """Test routing after insights - skip visualization"""
        orchestrator = LangGraphOrchestrator(
            openai_api_key="test-key",
            style_config=style_config
        )
        
        # Skip viz when not requested
        state = {
            "create_visualization": False,
            "row_count": 10
        }
        route = orchestrator._route_after_insights(state)
        assert route == "skip_viz"
        
        # Skip viz when no rows
        state = {
            "create_visualization": True,
            "row_count": 0
        }
        route = orchestrator._route_after_insights(state)
        assert route == "skip_viz"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.langgraph_orchestrator.ChatOpenAI')
    @patch('src.langgraph_orchestrator.DatabaseManager')
    @patch('src.langgraph_orchestrator.PermissionManager')
    def test_finalize_nodes(
        self, 
        mock_perm_class, 
        mock_db_class, 
        mock_openai, 
        style_config
    ):
        """Test finalize success and error nodes"""
        orchestrator = LangGraphOrchestrator(
            openai_api_key="test-key",
            style_config=style_config
        )
        
        # Test success finalization
        state = {"status": "pending"}
        result = orchestrator._finalize_success_node(state)
        assert result["status"] == "success"
        assert result["current_step"] == "completed"
        
        # Test error finalization
        state = {"status": "pending", "error_message": "Test error"}
        result = orchestrator._finalize_error_node(state)
        assert result["status"] == "error"
        assert result["current_step"] == "failed"
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.langgraph_orchestrator.ChatOpenAI')
    @patch('src.langgraph_orchestrator.DatabaseManager')
    @patch('src.langgraph_orchestrator.PermissionManager')
    @patch('src.langgraph_orchestrator.DataAnalysisAgent')
    @patch('src.langgraph_orchestrator.VisualizationAgent')
    def test_process_request_handles_errors(
        self,
        mock_viz_agent_class,
        mock_analysis_agent_class,
        mock_perm_class,
        mock_db_class,
        mock_openai,
        style_config
    ):
        """Test that process_request handles errors gracefully"""
        # Setup mock to raise exception
        mock_perm_instance = Mock()
        mock_perm_instance.get_allowed_tables.side_effect = Exception("Database not found")
        mock_perm_class.return_value = mock_perm_instance
        
        orchestrator = LangGraphOrchestrator(
            openai_api_key="test-key",
            style_config=style_config
        )
        
        # Process request
        response = orchestrator.process_request(
            query="Test query",
            database="nonexistent",
            user_role="analyst"
        )
        
        # Should return error response
        assert isinstance(response, AgentResponse)
        assert response.status == "error"
        assert response.error is not None
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.langgraph_orchestrator.ChatOpenAI')
    @patch('src.langgraph_orchestrator.DatabaseManager')
    @patch('src.langgraph_orchestrator.PermissionManager')
    def test_graph_structure(
        self, 
        mock_perm_class, 
        mock_db_class, 
        mock_openai, 
        style_config
    ):
        """Test that graph is built with correct structure"""
        orchestrator = LangGraphOrchestrator(
            openai_api_key="test-key",
            style_config=style_config
        )
        
        # Graph should be compiled and ready
        assert orchestrator.graph is not None
        
        # Graph should have nodes
        # Note: Checking the graph structure depends on LangGraph's internal API
        # For now, we just verify it's been created
        assert hasattr(orchestrator.graph, 'invoke')


class TestOrchestratorIntegration:
    """Integration tests with real workflow (mocked LLM responses)"""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('src.langgraph_orchestrator.ChatOpenAI')
    @patch('src.analysis_agent_v2.pd.read_sql_query')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_full_workflow_success(
        self,
        mock_plt_close,
        mock_plt_savefig,
        mock_read_sql,
        mock_openai_class,
        style_config,
        mock_config_files,
        test_db_path,
        tmp_path
    ):
        """Test complete successful workflow"""
        import pandas as pd
        from src.models import SQLQueryOutput, InsightsOutput, ChartRecommendation
        
        # Setup mocks
        mock_llm = Mock()
        mock_openai_class.return_value = mock_llm
        
        # Mock structured outputs
        mock_sql = SQLQueryOutput(
            sql_query="SELECT * FROM artists LIMIT 10",
            explanation="Get artists",
            tables_used=["artists"]
        )
        
        mock_insights = InsightsOutput(
            key_findings=["Found 3 artists", "AC/DC is first"],
            summary="Artist data retrieved"
        )
        
        mock_chart = ChartRecommendation(
            chart_type="bar",
            reasoning="Bar chart for categories",
            x_column="Name",
            y_column="ArtistId"
        )
        
        # Setup structured LLM responses
        mock_structured = Mock()
        mock_structured.invoke.side_effect = [mock_sql, mock_insights, mock_chart]
        mock_llm.with_structured_output.return_value = mock_structured
        
        # Mock SQL execution
        mock_df = pd.DataFrame({
            "ArtistId": [1, 2, 3],
            "Name": ["AC/DC", "Accept", "Aerosmith"]
        })
        mock_read_sql.return_value = mock_df
        
        # Patch config paths
        with patch('src.database_manager.DatabaseManager.load_config'):
            with patch('src.database_manager.PermissionManager.load_config'):
                with patch('src.database_manager.DatabaseManager.get_connection') as mock_conn:
                    with patch('src.database_manager.DatabaseManager.get_schema') as mock_schema:
                        with patch('src.database_manager.PermissionManager.get_allowed_tables') as mock_allowed:
                            
                            # Setup mocks
                            mock_conn.return_value = Mock()
                            mock_schema.return_value = {"artists": ["ArtistId", "Name"]}
                            mock_allowed.return_value = ["artists"]
                            
                            # Create orchestrator
                            orchestrator = LangGraphOrchestrator(
                                openai_api_key="test-key",
                                style_config=style_config
                            )
                            
                            # Process request
                            response = orchestrator.process_request(
                                query="Show all artists",
                                database="test",
                                user_role="analyst",
                                create_viz=True
                            )
                            
                            # Verify success
                            assert isinstance(response, AgentResponse)
                            assert response.status == "success"
                            assert response.row_count == 3
                            assert response.sql_query is not None
                            assert response.insights is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

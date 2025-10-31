"""Tests for Pydantic models"""
import pytest
from src.models import (
    SQLQueryOutput,
    InsightsOutput,
    ChartRecommendation,
    AnalysisState,
    AgentResponse
)


class TestSQLQueryOutput:
    """Test SQL query output model"""
    
    def test_valid_sql_output(self):
        """Test creating valid SQL output"""
        output = SQLQueryOutput(
            sql_query="SELECT * FROM artists LIMIT 10",
            explanation="Retrieves first 10 artists",
            tables_used=["artists"],
            estimated_rows=10
        )
        assert output.sql_query == "SELECT * FROM artists LIMIT 10"
        assert "artists" in output.tables_used
    
    def test_sql_output_without_optional_fields(self):
        """Test SQL output with only required fields"""
        output = SQLQueryOutput(
            sql_query="SELECT name FROM albums",
            explanation="Gets album names",
            tables_used=["albums"]
        )
        assert output.estimated_rows is None


class TestInsightsOutput:
    """Test insights output model"""
    
    def test_valid_insights(self):
        """Test creating valid insights"""
        insights = InsightsOutput(
            key_findings=[
                "Rock music is the top-selling genre",
                "Sales increased by 20% in Q4"
            ],
            summary="Rock dominates music sales with strong Q4 growth"
        )
        assert len(insights.key_findings) == 2
        assert insights.summary
    
    def test_insights_with_recommendations(self):
        """Test insights with optional recommendations"""
        insights = InsightsOutput(
            key_findings=["Finding 1", "Finding 2"],
            summary="Summary",
            recommendations=["Focus on rock music", "Expand Q4 promotions"]
        )
        assert len(insights.recommendations) == 2
    
    def test_minimum_findings_validation(self):
        """Test that at least 2 findings are required"""
        with pytest.raises(ValueError):
            InsightsOutput(
                key_findings=["Only one finding"],
                summary="Summary"
            )


class TestChartRecommendation:
    """Test chart recommendation model"""
    
    def test_valid_chart_recommendation(self):
        """Test creating valid chart recommendation"""
        rec = ChartRecommendation(
            chart_type="bar",
            reasoning="Bar charts are best for categorical comparisons",
            x_column="artist_name",
            y_column="total_sales"
        )
        assert rec.chart_type == "bar"
        assert rec.x_column == "artist_name"
    
    def test_invalid_chart_type(self):
        """Test that invalid chart types are rejected"""
        with pytest.raises(ValueError):
            ChartRecommendation(
                chart_type="invalid_type",
                reasoning="Some reasoning"
            )
    
    def test_chart_recommendation_without_axes(self):
        """Test chart recommendation for pie chart (no axes)"""
        rec = ChartRecommendation(
            chart_type="pie",
            reasoning="Pie charts show proportions"
        )
        assert rec.x_column is None
        assert rec.y_column is None


class TestAnalysisState:
    """Test analysis state model"""
    
    def test_initial_state(self):
        """Test creating initial state"""
        state = AnalysisState(
            user_query="Show top artists",
            database="chinook"
        )
        assert state.status == "pending"
        assert state.current_step == "initialized"
        assert state.user_role == "analyst"
    
    def test_state_progression(self):
        """Test state updates through workflow"""
        state = AnalysisState(
            user_query="Show top artists",
            database="chinook"
        )
        
        # Simulate workflow progression
        state.allowed_tables = ["artists", "albums"]
        state.current_step = "permissions_checked"
        
        state.sql_output = SQLQueryOutput(
            sql_query="SELECT * FROM artists",
            explanation="Get artists",
            tables_used=["artists"]
        )
        state.current_step = "sql_generated"
        
        assert state.sql_output.sql_query == "SELECT * FROM artists"
        assert state.current_step == "sql_generated"


class TestAgentResponse:
    """Test final agent response model"""
    
    def test_success_response(self):
        """Test successful response"""
        response = AgentResponse(
            status="success",
            query="Top 5 artists",
            sql_query="SELECT * FROM artists LIMIT 5",
            row_count=5,
            execution_time_seconds=1.5
        )
        assert response.status == "success"
        assert response.error is None
    
    def test_error_response(self):
        """Test error response"""
        response = AgentResponse(
            status="error",
            query="Invalid query",
            error="Database connection failed",
            row_count=0
        )
        assert response.status == "error"
        assert response.error == "Database connection failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for refactored Visualization Agent with structured outputs"""
import pytest
import pandas as pd
import os
import shutil
from unittest.mock import Mock, patch
from langchain_openai import ChatOpenAI
from src.visualization_agent import VisualizationAgent
from src.models import ChartRecommendation


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing"""
    return Mock(spec=ChatOpenAI)


@pytest.fixture
def style_config():
    """Sample style configuration"""
    return {
        "colors": ["#2E86AB", "#A23B72", "#F18F01"],
        "figure_size": (10, 6),
        "dpi": 100
    }


@pytest.fixture
def viz_agent(mock_llm, style_config):
    """Create visualization agent with mock LLM"""
    return VisualizationAgent(mock_llm, style_config)


@pytest.fixture
def sample_data():
    """Sample DataFrames for testing"""
    return {
        "categorical": pd.DataFrame({
            "Artist": ["AC/DC", "Accept", "Aerosmith", "Beatles"],
            "Sales": [1000, 800, 1200, 1500]
        }),
        "time_series": pd.DataFrame({
            "Month": ["Jan", "Feb", "Mar", "Apr"],
            "Revenue": [10000, 12000, 11000, 13000]
        }),
        "proportions": pd.DataFrame({
            "Genre": ["Rock", "Jazz", "Pop", "Classical"],
            "Percentage": [40, 20, 25, 15]
        }),
        "scatter": pd.DataFrame({
            "Price": [9.99, 14.99, 19.99, 24.99],
            "Units_Sold": [100, 80, 60, 40]
        })
    }


@pytest.fixture(scope="function")
def temp_output_dir(tmp_path):
    """Create temporary output directory"""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    yield str(output_dir)
    # Cleanup after test
    if output_dir.exists():
        shutil.rmtree(output_dir)


class TestVisualizationAgent:
    """Test suite for VisualizationAgent"""
    
    def test_initialization(self, mock_llm, style_config):
        """Test agent initializes correctly"""
        agent = VisualizationAgent(mock_llm, style_config)
        assert agent.llm == mock_llm
        assert agent.style == style_config
    
    def test_analyze_data_characteristics(self, viz_agent, sample_data):
        """Test data characteristic analysis"""
        df = sample_data["categorical"]
        characteristics = viz_agent._analyze_data_characteristics(df)
        
        assert "Numeric columns" in characteristics
        assert "Categorical columns" in characteristics
        assert "Sales" in characteristics
        assert "Artist" in characteristics
    
    def test_analyze_data_with_datetime(self, viz_agent):
        """Test characteristic analysis with datetime columns"""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=5),
            "Value": [100, 200, 150, 300, 250]
        })
        
        characteristics = viz_agent._analyze_data_characteristics(df)
        assert "DateTime columns" in characteristics
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_visualize_bar_chart(
        self, 
        mock_close, 
        mock_savefig, 
        viz_agent, 
        sample_data, 
        temp_output_dir
    ):
        """Test bar chart creation"""
        # Mock chart recommendation
        mock_recommendation = ChartRecommendation(
            chart_type="bar",
            reasoning="Bar charts are best for categorical comparisons",
            x_column="Artist",
            y_column="Sales"
        )
        
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_recommendation
        viz_agent.llm.with_structured_output = Mock(return_value=mock_structured_llm)
        
        # Create visualization
        result = viz_agent.visualize(
            data=sample_data["categorical"],
            query="Show sales by artist",
            output_dir=temp_output_dir
        )
        
        # Assertions
        assert result["status"] == "success"
        assert result["chart_type"] == "bar"
        assert "filepath" in result
        assert mock_savefig.called
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_visualize_line_chart(
        self, 
        mock_close, 
        mock_savefig, 
        viz_agent, 
        sample_data, 
        temp_output_dir
    ):
        """Test line chart creation"""
        mock_recommendation = ChartRecommendation(
            chart_type="line",
            reasoning="Line charts show trends over time",
            x_column="Month",
            y_column="Revenue"
        )
        
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_recommendation
        viz_agent.llm.with_structured_output = Mock(return_value=mock_structured_llm)
        
        result = viz_agent.visualize(
            data=sample_data["time_series"],
            query="Show revenue trend",
            output_dir=temp_output_dir
        )
        
        assert result["status"] == "success"
        assert result["chart_type"] == "line"
        assert mock_savefig.called
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_visualize_pie_chart(
        self, 
        mock_close, 
        mock_savefig, 
        viz_agent, 
        sample_data, 
        temp_output_dir
    ):
        """Test pie chart creation"""
        mock_recommendation = ChartRecommendation(
            chart_type="pie",
            reasoning="Pie charts show proportions",
            x_column="Genre",
            y_column="Percentage"
        )
        
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_recommendation
        viz_agent.llm.with_structured_output = Mock(return_value=mock_structured_llm)
        
        result = viz_agent.visualize(
            data=sample_data["proportions"],
            query="Show genre distribution",
            output_dir=temp_output_dir
        )
        
        assert result["status"] == "success"
        assert result["chart_type"] == "pie"
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_visualize_scatter_chart(
        self, 
        mock_close, 
        mock_savefig, 
        viz_agent, 
        sample_data, 
        temp_output_dir
    ):
        """Test scatter plot creation"""
        mock_recommendation = ChartRecommendation(
            chart_type="scatter",
            reasoning="Scatter plots show relationships",
            x_column="Price",
            y_column="Units_Sold"
        )
        
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_recommendation
        viz_agent.llm.with_structured_output = Mock(return_value=mock_structured_llm)
        
        result = viz_agent.visualize(
            data=sample_data["scatter"],
            query="Show price vs sales",
            output_dir=temp_output_dir
        )
        
        assert result["status"] == "success"
        assert result["chart_type"] == "scatter"
    
    def test_visualize_empty_data(self, viz_agent, temp_output_dir):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        
        result = viz_agent.visualize(
            data=empty_df,
            query="Show nothing",
            output_dir=temp_output_dir
        )
        
        assert result["status"] == "error"
        assert "empty" in result["error"].lower()
    
    def test_visualize_insufficient_columns(self, viz_agent, temp_output_dir):
        """Test handling of DataFrame with insufficient columns"""
        df = pd.DataFrame({"OnlyOne": [1, 2, 3]})
        
        result = viz_agent.visualize(
            data=df,
            query="Show data",
            output_dir=temp_output_dir
        )
        
        assert result["status"] == "error"
        assert "columns" in result["error"].lower()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_column_validation_in_recommendation(
        self, 
        mock_close, 
        mock_savefig, 
        viz_agent, 
        sample_data, 
        temp_output_dir
    ):
        """Test that invalid column names in recommendation are corrected"""
        # Mock recommendation with invalid column names
        mock_recommendation = ChartRecommendation(
            chart_type="bar",
            reasoning="Test",
            x_column="NonExistentColumn",
            y_column="AnotherBadColumn"
        )
        
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_recommendation
        viz_agent.llm.with_structured_output = Mock(return_value=mock_structured_llm)
        
        # Should not crash - should fall back to df.columns[0] and df.columns[1]
        result = viz_agent.visualize(
            data=sample_data["categorical"],
            query="Test query",
            output_dir=temp_output_dir
        )
        
        assert result["status"] == "success"
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_large_dataset_limiting(
        self, 
        mock_close, 
        mock_savefig, 
        viz_agent, 
        temp_output_dir
    ):
        """Test that large datasets are limited for bar charts"""
        # Create dataset with 30 rows
        large_df = pd.DataFrame({
            "Category": [f"Cat_{i}" for i in range(30)],
            "Value": range(30, 0, -1)
        })
        
        mock_recommendation = ChartRecommendation(
            chart_type="bar",
            reasoning="Bar chart",
            x_column="Category",
            y_column="Value"
        )
        
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = mock_recommendation
        viz_agent.llm.with_structured_output = Mock(return_value=mock_structured_llm)
        
        result = viz_agent.visualize(
            data=large_df,
            query="Show all categories",
            output_dir=temp_output_dir
        )
        
        # Should succeed and limit to top 15
        assert result["status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Pydantic models for structured outputs and state management"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd


class SQLQueryOutput(BaseModel):
    """Structured output for SQL query generation"""
    sql_query: str = Field(
        description="The generated SQL SELECT query. Must be a valid SQLite query."
    )
    explanation: str = Field(
        description="Brief explanation of what the query does"
    )
    tables_used: List[str] = Field(
        description="List of table names used in the query"
    )
    estimated_rows: Optional[int] = Field(
        default=None,
        description="Estimated number of rows this query might return"
    )


class InsightsOutput(BaseModel):
    """Structured output for data insights"""
    key_findings: List[str] = Field(
        description="List of 2-4 key findings from the data analysis",
        min_length=2,
        max_length=4
    )
    summary: str = Field(
        description="One-sentence summary of the overall finding"
    )
    recommendations: Optional[List[str]] = Field(
        default=None,
        description="Optional actionable recommendations based on the data"
    )


class ChartRecommendation(BaseModel):
    """Structured output for chart type recommendation"""
    chart_type: Literal["bar", "line", "pie", "scatter", "table"] = Field(
        description="Recommended chart type based on data characteristics"
    )
    reasoning: str = Field(
        description="Brief explanation of why this chart type is appropriate"
    )
    x_column: Optional[str] = Field(
        default=None,
        description="Recommended column for x-axis (if applicable)"
    )
    y_column: Optional[str] = Field(
        default=None,
        description="Recommended column for y-axis (if applicable)"
    )


class AnalysisState(BaseModel):
    """State object for LangGraph workflow"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Input fields
    user_query: str
    database: str
    user_role: str = "analyst"
    create_visualization: bool = True
    
    # Intermediate fields
    allowed_tables: Optional[List[str]] = None
    database_schema: Optional[dict] = None
    
    # SQL generation
    sql_output: Optional[SQLQueryOutput] = None
    sql_validated: bool = False
    
    # Data results
    result_data: Optional[str] = None  # JSON string of DataFrame
    row_count: int = 0
    
    # Analysis
    insights: Optional[InsightsOutput] = None
    
    # Visualization
    chart_recommendation: Optional[ChartRecommendation] = None
    visualization_path: Optional[str] = None
    
    # Status tracking
    status: Literal["pending", "processing", "success", "error"] = "pending"
    error_message: Optional[str] = None
    current_step: str = "initialized"


class AgentResponse(BaseModel):
    """Final response model for API/CLI"""
    status: Literal["success", "error"]
    query: str
    sql_query: Optional[str] = None
    data_preview: Optional[str] = None  # JSON string of first 10 rows
    full_data: Optional[str] = None     # JSON string of full DataFrame
    row_count: int = 0
    insights: Optional[InsightsOutput] = None
    visualization_path: Optional[str] = None
    chart_type: Optional[str] = None
    error: Optional[str] = None
    execution_time_seconds: Optional[float] = None
"""Pydantic models for structured outputs and state management"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field

class IntentOutput(BaseModel):
    """Intent classification for incoming queries"""
    intent: Literal["data_query", "off_topic", "follow_up"] = Field(
        description="Query intent: data_query (needs SQL), off_topic (general chat), follow_up (needs context)"
    )
    reasoning: str = Field(
        description="Brief explanation of classification"
    )
    requires_context: bool = Field(
        default=False,
        description="Whether this query needs previous conversation context"
    )


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

    intent: Optional[str] = None
    message: Optional[str] = None
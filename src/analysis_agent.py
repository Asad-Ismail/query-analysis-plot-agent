"""Data Analysis Agent with Structured Outputs"""
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List, Optional, Tuple
import logging
from src.models import SQLQueryOutput, InsightsOutput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAnalysisAgent:
    """Agent for analyzing data using LLM-generated SQL with structured outputs"""
    
    # SQL keywords that indicate unsafe operations
    DANGEROUS_KEYWORDS = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 
        'CREATE', 'TRUNCATE', 'REPLACE', 'MERGE'
    ]
    
    def __init__(self, llm: ChatOpenAI):
        """Initialize the analysis agent
        
        Args:
            llm: ChatOpenAI instance for LLM calls
        """
        self.llm = llm
        logger.info("DataAnalysisAgent initialized")
    
    def _generate_sql_structured(
        self, 
        query: str, 
        schema: Dict[str, dict], 
        allowed_tables: Optional[List[str]],
        error_history: Optional[List[str]] = None
    ) -> SQLQueryOutput:
        """Generate SQL using LLM with structured output
        Args:
            query: User's natural language query
            schema: Full database schema
            allowed_tables: Tables user can access (None = all tables)
            
        Returns:
            SQLQueryOutput with validated structure
        """
        # Filter schema based on permissions
        if allowed_tables:
            filtered_schema = {
                table: info 
                for table, info in schema.items() 
                if table in allowed_tables
            }
        else:
            filtered_schema = schema
        # Format schema for prompt
        schema_text = self._format_schema(filtered_schema)

        history_text = ""
        if error_history:
            history_text = f"""
            ***PREVIOUS ATTEMPT FAILED*** 
            Your last SQL query failed. Please review the error and generate a new, corrected query.
            The error was:
            {'\n'.join(error_history)}
            ***REVIEW AND FIX THE QUERY***
            """
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert. Generate a SQLite SELECT query based on the user's request.
                IMPORTANT RULES:
                1. ONLY generate SELECT statements - no modifications allowed
                2. Always include LIMIT clause (max 100 rows unless user specifies)
                3. Use proper JOINs when multiple tables are needed
                4. Include helpful column aliases for readability
                5. Only use tables and columns from the provided schema
                6. When calculating revenue or sales: Use UnitPrice * Quantity 

                ***CRITICAL RULE***:
                If the user's query cannot be answered *exactly* using ONLY the tables and columns in the "Available Schema", you MUST respond with an error. 
                DO NOT substitute other tables. For example, if the user asks for "employees" but that table is not in the schema, you must fail, not query "artists".

                Available Schema:
                {schema}"""),
                
                ("user", """{history}
                User Query: {query} """)
                ])
        
        # Use structured output
        structured_llm = self.llm.with_structured_output(SQLQueryOutput)
        
        # Generate SQL
        result = structured_llm.invoke(prompt.format_messages(schema=schema_text, history=history_text, query=query))
        
        return result
    
    def _generate_insights_structured(
        self, 
        query: str, 
        df: pd.DataFrame,
        sql_explanation: str
    ) -> InsightsOutput:
        """Generate insights using LLM with structured output
        
        Args:
            query: Original user query
            df: DataFrame with query results
            sql_explanation: Explanation of what the SQL does
            
        Returns:
            InsightsOutput with key findings and summary
        """
        # Prepare data summary
        data_summary = self._prepare_data_summary(df)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analyst. Provide clear, actionable insights from the data.

            Generate:
            1. 2-4 key findings (specific observations from the data)
            2. A one-sentence summary

            Be specific with numbers and patterns you observe."""),
            ("user", """Original Question: {query}

            SQL Explanation: {sql_explanation}

            Data Summary:
            {data_summary}

            Full Data Preview (first 10 rows):
            {data_preview}""") ])
        
        # Use structured output
        structured_llm = self.llm.with_structured_output(InsightsOutput)
        
        result = structured_llm.invoke(
            prompt.format_messages(
                query=query,
                sql_explanation=sql_explanation,
                data_summary=data_summary,
                data_preview=df.head(10).to_string()
            )
        )      
        return result
    
    def _format_schema(self, schema: Dict[str, dict]) -> str: 
            """Format schema with types for prompt"""
            lines = []
            for table_name, table_info in schema.items():
                lines.append(f"Table: {table_name}")
                
                # Format columns with types (Task 1)
                column_lines = [
                    f"{col['name']} ({col['type']})" 
                    for col in table_info['columns']
                ]
                lines.append(f"  Columns: {', '.join(column_lines)}")
            
            return "\n".join(lines) # 
    
    def _prepare_data_summary(self, df: pd.DataFrame) -> str:
        """Prepare statistical summary of data"""
        summary_parts = [
            f"Total Rows: {len(df)}",
            f"Total Columns: {len(df.columns)}",
            f"Columns: {', '.join(df.columns.tolist())}"
        ]
        
        # Add numeric column stats if present
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary_parts.append("\nNumeric Column Statistics:")
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                summary_parts.append(
                    f"  {col}: min={df[col].min():.2f}, "
                    f"max={df[col].max():.2f}, "
                    f"mean={df[col].mean():.2f}"
                )
        
        return "\n".join(summary_parts)
    
    def _is_safe_query(self, sql: str) -> bool:
        """Validate query safety - only SELECT statements allowed
        
        Args:
            sql: SQL query to validate
            
        Returns:
            True if safe, False otherwise
        """
        sql_upper = sql.upper().strip()
        
        # Check for dangerous keywords
        for keyword in self.DANGEROUS_KEYWORDS:
            if keyword in sql_upper:
                return False
        
        # Must start with SELECT
        if not sql_upper.startswith('SELECT'):
            return False
        
        return True
    
    def _validate_table_access(
        self, 
        tables_used: List[str], 
        allowed_tables: List[str]
    ) -> bool:
        """Validate user has access to all tables in query
        
        Args:
            tables_used: Tables used in the query
            allowed_tables: Tables user is permitted to access
            
        Returns:
            True if all tables are allowed, False otherwise
        """
        return all(table in allowed_tables for table in tables_used)
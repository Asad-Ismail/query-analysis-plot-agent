"""Data Analysis Agent"""
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List, Optional

class DataAnalysisAgent:
    """Agent for analyzing data using LLM-generated SQL"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze(
        self, 
        query: str, 
        conn, 
        schema: Dict[str, List[str]],
        allowed_tables: Optional[List[str]] = None
    ) -> Dict:
        """Main analysis workflow"""
        # Generate SQL
        sql_query = self._generate_sql(query, schema, allowed_tables)
        
        # Safety check
        if not self._is_safe_query(sql_query):
            return {
                "status": "error",
                "error": "Unsafe query detected"
            }
        
        # Execute
        try:
            df = pd.read_sql_query(sql_query, conn)
            insights = self._generate_insights(query, df)
            
            return {
                "status": "success",
                "query": query,
                "sql": sql_query,
                "data": df,
                "insights": insights
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_sql(self, query: str, schema: Dict, allowed_tables: Optional[List[str]]) -> str:
        """Generate SQL using LLM"""
        # Filter schema based on permissions
        if allowed_tables:
            schema = {k: v for k, v in schema.items() if k in allowed_tables}
        
        tables_info = "\n".join([
            f"- {table}: {', '.join(cols)}"
            for table, cols in schema.items()
        ])
        
        prompt = ChatPromptTemplate.from_template(
            """Generate a SQLite SELECT query.
                SCHEMA:
                {schema}
                QUERY: {query}
                Rules:
                - ONLY SELECT statements
                - Include LIMIT 100 if not specified
                - Return ONLY SQL, no explanation
                SQL:""")
        
        response = self.llm.invoke(prompt.format(schema=tables_info, query=query))
        return response.content.strip().replace("```sql", "").replace("```", "").strip()
    
    def _is_safe_query(self, sql: str) -> bool:
        """Validate query safety"""
        dangerous = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        return not any(kw in sql.upper() for kw in dangerous)
    
    def _generate_insights(self, query: str, df: pd.DataFrame) -> str:
        """Generate insights using LLM"""
        prompt = ChatPromptTemplate.from_template(
            """Provide 2-3 key insights from this data.
                QUERY: {query}
                DATA PREVIEW: {data}
                Insights:""")
        
        response = self.llm.invoke(prompt.format(query=query, data=df.head().to_string()))
        return response.content.strip()
from typing import TypedDict, Annotated, Literal,Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import pandas as pd
from io import StringIO
import time
import logging
from langfuse import Langfuse
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
from src.models import  AgentResponse, SQLQueryOutput, InsightsOutput
from src.database_manager import DatabaseManager, PermissionManager
from src.analysis_agent import DataAnalysisAgent
from src.visualization_agent import VisualizationAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangGraphOrchestrator:
    """LangGraph for workflow coordination"""
    
    def __init__(self, openai_api_key: str, style_config: dict, openai_api_base: str = None):
        """Initialize orchestrator with agents and managers
        
        Args:
            openai_api_key: OpenAI API key
            style_config: Visualization styling configuration
            openai_api_base: Optional custom API base URL
        """

        # Initialize Langfuse 
        self.langfuse = Langfuse()
        self.langfuse_handler = LangfuseCallbackHandler()

        # Initialize LLM
        llm_kwargs = {
            "model": "gpt-4o-mini",
            "temperature": 0,
            "api_key": openai_api_key,
            "callbacks": [self.langfuse_handler]
        }
        
        if openai_api_base:
            llm_kwargs["openai_api_base"] = openai_api_base
        
        self.llm = ChatOpenAI(**llm_kwargs)
        
        self.db_manager = DatabaseManager()
        self.perm_manager = PermissionManager()
        self.analysis_agent = DataAnalysisAgent(self.llm)
        self.viz_agent = VisualizationAgent(self.llm, style_config)

        self.max_retries = 3
        self.graph = self._build_graph()
        
        logger.info("LangGraph Orchestrator initialized!")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow
        
        Returns:
            Compiled StateGraph
        """
        # Create workflow graph
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("check_permissions", self._check_permissions_node)
        workflow.add_node("generate_sql", self._generate_sql_node)
        workflow.add_node("execute_query", self._execute_query_node)
        workflow.add_node("generate_insights", self._generate_insights_node)
        workflow.add_node("create_visualization", self._create_visualization_node)
        workflow.add_node("finalize_success", self._finalize_success_node)
        workflow.add_node("finalize_error", self._finalize_error_node)
        
        # Set entry point
        workflow.set_entry_point("check_permissions")
        
        # Add edges
        workflow.add_conditional_edges(
            "check_permissions",
            self._route_after_permissions,
            {
                "continue": "generate_sql",
                "error": "finalize_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_sql",
            self._route_after_sql,
            {
                "continue": "execute_query",
                "error": "finalize_error"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_query",
            self._route_sql_execution, 
            {
                "generate_insights": "generate_insights",
                "retry_sql": "generate_sql", 
                "error": "finalize_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_insights",
            self._route_after_insights,
            {
                "visualize": "create_visualization",
                "skip_viz": "finalize_success"
            }
        )
        
        workflow.add_edge("create_visualization", "finalize_success")
        workflow.add_edge("finalize_success", END)
        workflow.add_edge("finalize_error", END)
        
        return workflow.compile()
    
    def process_request(
        self,
        query: str,
        database: str,
        user_role: str = "analyst",
        create_viz: bool = True,
        session_id: str = None,
        chart_type_override: Optional[str] = None
    ) -> AgentResponse:
        """Process analysis request through LangGraph workflow
        
        Args:
            query: Natural language query
            database: Database name
            user_role: User role for permissions
            create_viz: Whether to create visualization
            
        Returns:
            AgentResponse with results or error
        """
        start_time = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {query}")
        logger.info(f"Database: {database} | Role: {user_role}")
        logger.info(f"{'='*60}\n")

        trace = self.langfuse.trace(
        name="query_analysis_request",
        user_id=user_role,
        session_id=session_id,
        metadata={
            "database": database,
            "user_role": user_role,
            "create_visualization": create_viz,
            "query": query
        },
        tags=[database, user_role, "cli_request"],
        input={"query": query})
    
        # Initialize state
        initial_state = {
            "user_query": query,
            "database": database,
            "user_role": user_role,
            "create_visualization": create_viz,
            "status": "pending",
            "current_step": "initialized",
            "allowed_tables": None,
            "database_schema": None,
            "sql_output": None,
            "sql_validated": False,
            "result_data": None,
            "row_count": 0,
            "insights": None,
            "chart_recommendation": None,
            "visualization_path": None,
            "error_message": None,
            "chart_type_override": chart_type_override
        }
        
        # Execute workflow
        try:
            final_state = self.graph.invoke(initial_state)
            execution_time = time.time() - start_time
            if final_state["status"] == "success":
                data_preview = None
                full_data_json = None
                if final_state.get("result_data"):
                    full_data_json = final_state.get("result_data") 
                    df = pd.read_json(StringIO(final_state["result_data"]))
                    data_preview = df.head(10).to_json(orient='records')
                
                response = AgentResponse(
                    status="success",
                    query=query,
                    sql_query=final_state.get("sql_output", {}).get("sql_query") if final_state.get("sql_output") else None,
                    data_preview=data_preview,
                    full_data=full_data_json,
                    row_count=final_state.get("row_count", 0),
                    insights=final_state.get("insights"),
                    visualization_path=final_state.get("visualization_path"),
                    chart_type=final_state.get("chart_recommendation", {}).get("chart_type") if final_state.get("chart_recommendation") else None,
                    execution_time_seconds=execution_time
                )

                trace.update(
                output={
                    "status": "success",
                    "row_count": final_state.get("row_count", 0),
                    "sql_query": final_state.get("sql_output", {}).get("sql_query") if final_state.get("sql_output") else None,
                    "chart_type": final_state.get("chart_recommendation", {}).get("chart_type") if final_state.get("chart_recommendation") else None,
                    "execution_time": execution_time
                },
                metadata={
                    "tables_used": final_state.get("sql_output", {}).get("tables_used") if final_state.get("sql_output") else [],
                    "visualization_created": final_state.get("visualization_path") is not None})

            else:
                response = AgentResponse(
                    status="error",
                    query=query,
                    error=final_state.get("error_message", "Unknown error occurred"),
                    execution_time_seconds=execution_time,
                    row_count=0
                )
                trace.update(
                output={
                    "status": "error",
                    "error": final_state.get("error_message"),
                    "execution_time": execution_time
                },
                level="ERROR")
            
            logger.info(f"\n Request completed in {execution_time:.2f}s")
            return response
            
        except Exception as e:
            trace.update(
            output={
                "status": "error",
                "error": str(e),
                "execution_time": execution_time
            },
            level="ERROR")
            logger.error(f"Orchestration error: {str(e)}")
            return AgentResponse(
                status="error",
                query=query,
                error=f"Orchestration failed: {str(e)}",
                execution_time_seconds=time.time() - start_time,
                row_count=0
            )

    
    def _check_permissions_node(self, state: dict) -> dict:
        """Check user permissions and load database schema"""
        logger.info("Checking permissions...")
        state["current_step"] = "checking_permissions"
        
        try:
            # Get allowed tables
            allowed_tables = self.perm_manager.get_allowed_tables(
                state["user_role"], 
                state["database"]
            )
            
            # Get schema
            schema = self.db_manager.get_schema(state["database"])
            
            state["allowed_tables"] = allowed_tables
            state["database_schema"] = schema
            
            logger.info(f"Permissions loaded. Allowed tables: {allowed_tables or 'ALL'}")
            return state
            
        except Exception as e:
            logger.error(f"Permission check failed: {str(e)}")
            state["status"] = "error"
            state["error_message"] = f"Permission check failed: {str(e)}"
            return state
    
    def _generate_sql_node(self, state: dict) -> dict:
        """Generate SQL query using analysis agent"""
        logger.info(" Generating SQL...")
        state["current_step"] = "generating_sql"
        
        try:
            sql_output = self.analysis_agent._generate_sql_structured(
                query=state["user_query"],
                schema=state["database_schema"],
                allowed_tables=state["allowed_tables"]
            )
            
            # Validate SQL safety
            if not self.analysis_agent._is_safe_query(sql_output.sql_query):
                query_lower = sql_output.sql_query.lower()

                print(f"Output query is {query_lower}")

                if not query_lower:
                    logger.warning(f"OOD query detected!")
                    state["status"] = "error"
                    state["error_message"] = "Out of Domain Query. Please ask relevan questions!"

                elif query_lower.startswith(('error', 'sorry', 'cannot', 'access denied')):
                    logger.warning(f"LLM returned a soft error: {sql_output.sql_query}")
                    state["status"] = "error"
                    state["error_message"] = sql_output.sql_query
                else:
                    # It's a genuinely unsafe query
                    logger.warning(f"Unsafe query detected: {sql_output.sql_query}")
                    state["status"] = "error"
                    state["error_message"] = "Generated query contains unsafe operations"
                return state

            
            # Validate table permissions
            if state["allowed_tables"] and not self.analysis_agent._validate_table_access(
                sql_output.tables_used, state["allowed_tables"]
            ):
                state["status"] = "error"
                state["error_message"] = f"Access denied to tables: {sql_output.tables_used}"
                return state
            
            state["sql_output"] = {
                "sql_query": sql_output.sql_query,
                "explanation": sql_output.explanation,
                "tables_used": sql_output.tables_used
            }
            state["sql_validated"] = True
            
            logger.info(f" SQL generated: {sql_output.sql_query}")
            return state
            
        except Exception as e:
            logger.error(f"SQL generation failed: {str(e)}")
            state["status"] = "error"
            state["error_message"] = f"SQL generation failed: {str(e)}"
            return state
    
    def _execute_query_node(self, state: dict) -> dict:
        """Execute SQL query against database"""
        logger.info("Executing query...")
        state["current_step"] = "executing_query"
        
        try:
            conn = self.db_manager.get_connection(state["database"])
            
            df = pd.read_sql_query(state["sql_output"]["sql_query"], conn)
            conn.close()
            
            state["result_data"] = df.to_json(orient='records')
            state["row_count"] = len(df)
            
            logger.info(f"Successfully Query executed: {len(df)} rows returned")
            return state

        except (pd.errors.DatabaseError, sqlite3.Error) as e:
            logger.warning(f"SQL execution error: {str(e)}")
            state["error_history"] = state.get("error_history", []) + [str(e)]
            state["status"] = "retry_sql" 
            return state
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            state["status"] = "error"
            state["error_message"] = f"Query execution failed: {str(e)}"
            return state
    
    def _generate_insights_node(self, state: dict) -> dict:
        """Generate insights from query results"""
        logger.info(" Generating insights...")
        state["current_step"] = "generating_insights"
        
        try:
            df = pd.read_json(StringIO(state["result_data"]))
            
            insights = self.analysis_agent._generate_insights_structured(
                query=state["user_query"],
                df=df,
                sql_explanation=state["sql_output"]["explanation"]
            )
            
            state["insights"] = {
                "key_findings": insights.key_findings,
                "summary": insights.summary,
            }
            
            logger.info(" Insights generated")
            logger.info(f"Summary: {insights.summary}")
            return state
            
        except Exception as e:
            logger.error(f"Insight generation failed: {str(e)}")
            # Non-critical error - continue without insights
            state["insights"] = {
                "key_findings": ["Insight generation failed"],
                "summary": "Unable to generate insights",
            }
            return state
    
    def _create_visualization_node(self, state: dict) -> dict:
        """Create visualization"""
        logger.info("Creating visualization...")
        state["current_step"] = "creating_visualization"
        
        try:
            df = pd.read_json(StringIO(state["result_data"]))
            
            viz_result = self.viz_agent.visualize(
                data=df,
                query=state["user_query"],
                chart_type_override=state.get("chart_type_override")
            )
            
            if viz_result["status"] == "success":
                state["visualization_path"] = viz_result["filepath"]
                state["chart_recommendation"] = {
                    "chart_type": viz_result["chart_type"],
                    "reasoning": viz_result["reasoning"]
                }
                logger.info(f" Visualization saved: {viz_result['filepath']}")
            else:
                logger.warning(f"Visualization failed: {viz_result.get('error')}")
            
            return state
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
            # Non-critical error - continue without visualization
            return state
    
    def _finalize_success_node(self, state: dict) -> dict:
        """Finalize successful execution"""
        state["status"] = "success"
        state["current_step"] = "completed"
        logger.info(" Workflow completed successfully")
        return state
    
    def _finalize_error_node(self, state: dict) -> dict:
        """Finalize error state"""
        state["status"] = "error"
        state["current_step"] = "failed"
        logger.error(f" Workflow failed: {state.get('error_message')}")
        return state
    

    def _route_after_permissions(self, state: dict) -> Literal["continue", "error"]:
        """Route after permission check"""
        if state["status"] == "error":
            return "error"
        return "continue"
    
    def _route_after_sql(self, state: dict) -> Literal["continue", "error"]:
        """Route after SQL generation"""
        if state["status"] == "error":
            return "error"
        return "continue"
    

    def _route_sql_execution(self, state: dict) -> Literal["generate_insights", "retry_sql", "error"]:
        """Route after query execution, checking for retries."""
        
        if state.get("status") == "retry_sql":
            state["status"] = "pending" # Reset status for the next attempt
            retry_count = state.get("retry_count", 0) + 1
            state["retry_count"] = retry_count
            
            if retry_count > self.max_retries:
                logger.error("Max SQL retries exceeded.")
                state["error_message"] = "Max SQL retries exceeded after query failed."
                return "error"
            
            logger.info(f"Retrying SQL generation (attempt {retry_count})...")
            return "retry_sql"
        
        if state.get("status") == "error":
            return "error"
            
        return "generate_insights"
    
    def _route_after_insights(self, state: dict) -> Literal["visualize", "skip_viz"]:
        """Route after insights generation"""
        if state["create_visualization"] and state["row_count"] > 0:
            return "visualize"
        return "skip_viz"
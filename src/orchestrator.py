"""Agent Orchestrator - coordinates both agents"""
from langchain_openai import ChatOpenAI
from .database_manager import DatabaseManager, PermissionManager
from .analysis_agent import DataAnalysisAgent
from .visualization_agent import VisualizationAgent

class AgentOrchestrator:
    """Coordinates analysis and visualization agents"""
    
    def __init__(self, openai_api_key: str, style_config: dict):
        #self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
        model = 'o4-mini'
        self.llm = ChatOpenAI(openai_api_base="https://chat.int.bayer.com/api/v2",openai_api_key=openai_api_key,model=model,temperature=0.0)
        self.db_manager = DatabaseManager()
        self.perm_manager = PermissionManager()
        self.analysis_agent = DataAnalysisAgent(self.llm)
        self.viz_agent = VisualizationAgent(self.llm, style_config)
    
    def process_request(
        self, 
        query: str, 
        database: str, 
        user_role: str = "analyst",
        create_viz: bool = True
    ) -> dict:
        """Process complete analysis request"""
        
        print(f"\n{'='*60}")
        print(f"Processing: {query}")
        print(f"Database: {database} | Role: {user_role}")
        print(f"{'='*60}\n")
        
        # Get permissions
        allowed_tables = self.perm_manager.get_allowed_tables(user_role, database)
        
        # Get connection and schema
        conn = self.db_manager.get_connection(database)
        schema = self.db_manager.get_schema(database)
        
        # Analysis phase
        analysis_result = self.analysis_agent.analyze(query, conn, schema, allowed_tables)
        
        if analysis_result["status"] != "success":
            return analysis_result
        
        print(f"✅ Analysis complete: {analysis_result['data'].shape[0]} rows")
        print(f"\nInsights:\n{analysis_result['insights']}\n")
        
        # Visualization phase
        if create_viz:
            viz_result = self.viz_agent.visualize(analysis_result["data"], query)
            print(f"✅ Visualization saved: {viz_result['filepath']}")
            analysis_result["visualization"] = viz_result
        
        conn.close()
        return analysis_result
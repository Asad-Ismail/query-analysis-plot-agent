"""Flask Backend for Data Analysis Agent"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import find_dotenv, load_dotenv
import os
from src.langgraph_orchestrator import LangGraphOrchestrator
from src.database_manager import DatabaseManager
import time

load_dotenv(find_dotenv())

app = Flask(__name__, static_folder='.')
CORS(app)

# Style configuration
STYLE_CONFIG = {
    "colors": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"],
    "figure_size": (10, 6),
    "dpi": 100
}

# Initialize components
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable required")

orchestrator = LangGraphOrchestrator(
    openai_api_key=api_key,
    style_config=STYLE_CONFIG,
    openai_api_base=api_base
)

db_manager = DatabaseManager()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/databases', methods=['GET'])
def get_databases():
    """Get list of available databases"""
    return jsonify({
        "databases": list(db_manager.databases.keys())
    })

@app.route('/api/schema', methods=['POST'])
def get_schema():
    """Get schema for a specific database"""
    data = request.json
    db_name = data.get('database', 'chinook')
    
    try:
        schema = db_manager.get_schema(db_name)
        return jsonify({
            "status": "success",
            "schema": schema
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 400

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process natural language query"""
    data = request.json
    query = data.get('query')
    database = data.get('database', 'chinook')
    user_role = data.get('role', 'analyst')
    create_viz = data.get('create_viz', True)
    chart_type = data.get('chart_type')
    session_id = data.get('session_id') 
    
    if not query:
        return jsonify({
            "status": "error",
            "error": "Query is required"
        }), 400
    
    if not session_id:
        session_id = f"web-{int(time.time())}"
    
    try:
        result = orchestrator.process_request(
            query=query,
            database=database,
            user_role=user_role,
            create_viz=create_viz,
            session_id=session_id,
            chart_type_override=chart_type
        )
        
        # Convert result to dict
        response_dict = {
            "status": result.status,
            "query": result.query,
            "intent": result.intent,  
            "message": result.message, 
            "sql_query": result.sql_query,
            "data_preview": result.data_preview,
            "full_data": result.full_data,
            "row_count": result.row_count,
            "insights": result.insights.dict() if result.insights else None,
            "visualization_path": result.visualization_path,
            "chart_type": result.chart_type,
            "error": result.error,
            "execution_time": result.execution_time_seconds,
            "session_id": session_id  # NEW: Return session to client
        }
        
        return jsonify(response_dict)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    """Clear conversation history for a session"""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({
            "status": "error",
            "error": "Session ID required"
        }), 400
    
    try:
        orchestrator.conversation_manager.clear_session(session_id)
        return jsonify({
            "status": "success",
            "message": "Session cleared"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """Serve generated visualizations"""
    return send_from_directory('outputs', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
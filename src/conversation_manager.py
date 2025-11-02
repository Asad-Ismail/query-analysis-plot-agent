"""Simple in-memory conversation manager"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    query: str
    response_summary: str
    sql_query: Optional[str] = None
    row_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class ConversationManager:
    """Manages conversation history per session"""
    
    def __init__(self, max_history: int = 10):
        self.sessions: Dict[str, List[ConversationTurn]] = {}
        self.max_history = max_history
    
    def add_turn(
        self, 
        session_id: str, 
        query: str, 
        response_summary: str,
        sql_query: Optional[str] = None,
        row_count: int = 0
    ):
        """Add a turn to session history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        turn = ConversationTurn(
            query=query,
            response_summary=response_summary,
            sql_query=sql_query,
            row_count=row_count
        )
        
        self.sessions[session_id].append(turn)
        
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
    
    def get_history(self, session_id: str, last_n: int = 3) -> List[ConversationTurn]:
        """Get recent conversation history"""
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id][-last_n:]
    
    def get_context_string(self, session_id: str, last_n: int = 3) -> str:
        """Format history as context string"""
        history = self.get_history(session_id, last_n)
        if not history:
            return "No previous conversation."
        
        context_parts = ["Previous conversation:"]
        for i, turn in enumerate(history, 1):
            context_parts.append(f"\nTurn {i}:")
            context_parts.append(f"  User: {turn.query}")
            context_parts.append(f"  Result: {turn.response_summary}")
            if turn.sql_query:
                context_parts.append(f"  SQL: {turn.sql_query}")
        
        return "\n".join(context_parts)
    
    def clear_session(self, session_id: str):
        """Clear session history"""
        if session_id in self.sessions:
            del self.sessions[session_id]

"""Intent classification for routing queries"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.models import IntentOutput
import logging

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classifies user intent to route appropriately"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def classify(self, query: str, conversation_context: str = None) -> IntentOutput:
        """Classify query intent with structured output
        
        Args:
            query: User's query
            conversation_context: Optional previous conversation
            
        Returns:
            IntentOutput with classification
        """
        context_hint = ""
        if conversation_context:
            context_hint = f"\n\nPrevious conversation:\n{conversation_context}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intent classifier for a data analysis system.

Classify queries into:

1. **data_query**: Questions about data that need database queries
   - "Show top 5 customers"
   - "What's the total revenue?"
   - "List all products"

2. **follow_up**: Queries referencing previous results
   - "Show me more"
   - "What about 2011?" (when 2010 was just queried)
   - "Break that down by region"

3. **off_topic**: General questions not related to data analysis
   - "What's the meaning of life?"
   - "How do I cook pasta?"
   - "Tell me a joke"
   - Greetings like "hello" should be treated as off_topic

IMPORTANT: 
- If query references previous results ("more", "also", "what about"), mark requires_context=True
- Be strict: only data_query if it clearly needs database access"""),
            ("user", "Query: {query}{context}")
        ])
        
        structured_llm = self.llm.with_structured_output(IntentOutput)
        
        result = structured_llm.invoke(
            prompt.format_messages(
                query=query,
                context=context_hint
            )
        )
        
        logger.info(f"Intent classified as '{result.intent}': {result.reasoning}")
        return result

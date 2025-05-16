# graph_state.py
from typing import TypedDict, Optional, List, Dict, Any

class AgentState(TypedDict):
    original_query: str
    intent: Optional[str]           # e.g., "SQL", "RETRIEVAL", "META", "GREETING", "UNKNOWN"
    entities: Optional[Dict[str, Any]] # e.g., {"order_id": "12345"}
    
    sql_query_generated: Optional[str]
    sql_query_result: Optional[List[Any]] # List of tuples or dicts
    
    retrieved_contexts: Optional[List[Dict[str, Any]]] # List of {'source': str, 'text': str}
    rag_summary: Optional[str]
    
    intermediate_response: Optional[str] # Could be direct result from SQL/RAG or meta answer
    final_answer: Optional[str]
    
    error_message: Optional[str]
    history: List[Dict[str, str]] # To maintain conversation history (optional for now)

    # To track which version of a node processed a step (for meta-queries)
    # This could be populated by each node based on its loaded config
    processing_steps_versions: Dict[str, str]

    # For benchmarking
    node_latencies: Optional[Dict[str, float]] # Stores latency for each executed node
    node_execution_order: Optional[List[str]]  # Stores the order of node execution
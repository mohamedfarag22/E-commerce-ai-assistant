# app_graph.py
from langgraph.graph import StateGraph, END
from graph_state import AgentState
from agents.intent_parser_node import parse_intent_node
from agents.sql_node import sql_node
from agents.retrieval_node import retrieval_node
from agents.meta_query_node import meta_query_node
from agents.response_node import response_synthesis_node
from utils import logger

# Define nodes
workflow = StateGraph(AgentState)

workflow.add_node("intent_parser", parse_intent_node)
workflow.add_node("sql_processor", sql_node)
workflow.add_node("retrieval_processor", retrieval_node)
workflow.add_node("meta_query_handler", meta_query_node)
workflow.add_node("response_synthesizer", response_synthesis_node)

# Define edges
workflow.set_entry_point("intent_parser")

def route_after_intent(state: AgentState):
    intent = state.get("intent")
    logger.info(f"Routing based on intent: {intent}")
    if intent in ["SQL_QUERY", "SQL_QUERY_GENERAL", "ORDER_STATUS", "PRODUCT_AVAILABILITY"]:
        return "sql_processor"
    elif intent in ["RETURN_INFO", "SHIPPING_INFO","PROBLEM_REPORT"]:
        return "retrieval_processor"
    elif intent == "META_QUERY":
        return "meta_query_handler"
    elif intent == "OUT_OF_CONTEXT":
        return "response_synthesizer"
    else:
        state["intermediate_response"] = "I'm not sure how to help with that. Could you please rephrase?"
        logger.warning(f"Unknown or unhandled intent: {intent}")
        return "response_synthesizer"


workflow.add_conditional_edges(
    "intent_parser",
    route_after_intent,
    {
        "sql_processor": "sql_processor",
        "retrieval_processor": "retrieval_processor",
        "meta_query_handler": "meta_query_handler",
        "response_synthesizer": "response_synthesizer" # For direct routing like GREETING or UNKNOWN
    }
)

# After SQL or Retrieval, go to Response Synthesizer
workflow.add_edge("sql_processor", "response_synthesizer")
workflow.add_edge("retrieval_processor", "response_synthesizer")

# After Meta Query, also go to Response Synthesizer (it will use intermediate_response)
workflow.add_edge("meta_query_handler", "response_synthesizer")

# Response synthesizer is the final step before END
workflow.add_edge("response_synthesizer", END)


# Compile the graph
app = workflow.compile()

# For visualization (optional, requires `pip install pygraphviz` or `pydot`)
# try:
#     app.get_graph().draw_mermaid_png(output_file_path="graph_visualization.png")
#     logger.info("Graph visualization saved to graph_visualization.png")
# except Exception as e:
#     logger.warning(f"Could not generate graph visualization: {e}. Ensure pygraphviz or pydot is installed.")
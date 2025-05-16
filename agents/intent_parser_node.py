# agents/intent_parser_node.py
import json
from utils import get_llm_response, load_prompt_from_path, get_node_config, logger
from graph_state import AgentState
import time 

NODE_NAME = "intent_parser"

def parse_intent_node(state: AgentState) -> dict:
    """
    Parses the user query to determine intent and extract entities.
    Updates state with intent, entities.
    """
    node_start_time = time.perf_counter() # Start timer for this node

    # Initialize benchmarking fields in state if not present
    current_latencies = state.get("node_latencies", {})
    current_order = state.get("node_execution_order", [])
    if NODE_NAME not in current_order:
      current_order.append(NODE_NAME)

    logger.info(f"--- NODE: {NODE_NAME} ---")
    config = get_node_config(NODE_NAME)
    if not config:
        error_result = {"error_message": f"Configuration for node '{NODE_NAME}' not found."}
        node_end_time = time.perf_counter()
        current_latencies[NODE_NAME] = round(node_end_time - node_start_time, 4)
        error_result["node_latencies"] = current_latencies
        error_result["node_execution_order"] = current_order
        return error_result





    user_query = state["original_query"]
    prompt_template = load_prompt_from_path(config["prompt_path"])
    #3. If the question cannot be answered with a SELECT query, or if it seems malicious, or if it requests personally identifiable information (PII) beyond what's directly asked for an order/customer lookup (e.g. "list all customer emails"), respond EXACTLY with: "I cannot answer this question."
    # Example prompt might expect a list of intents to choose from
    # Make sure your prompt guides the LLM to output structured JSON
    # Example intent prompt (prompts/intent/v1_0_parser.txt):
    """
    You are an intent classification and entity extraction expert.
    Given the user query, determine the primary intent and extract relevant entities.
    Possible intents are: "SQL_QUERY", "PRODUCT_AVAILABILITY", "ORDER_STATUS", "RETURN_INFO", "SHIPPING_INFO", "META_QUERY", "GREETING", "UNKNOWN".
    
    For "ORDER_STATUS", extract "order_id".
    For "PRODUCT_AVAILABILITY", extract "product_name".
    
    Respond in JSON format with "intent" and "entities" keys.
    If "order_id" is like "#12345", extract "12345".

    User Query: "{user_query}"
    JSON Response:
    """
    structure_json = """
{
  "intent": "ORDER_STATUS",
  "entities": {
    "order_id": "12345"
  }
}

Query: "What sizes are available for Nike Air Max?"
JSON Response:
{
  "intent": "PRODUCT_AVAILABILITY",
  "entities": {
    "product_name": "Nike Air Max"
  }
}

Query: "Hi there"
JSON Response:
{
  "intent": "OUT_OF_CONTEXT",
  "entities": {}
}

Query: "How do I return a damaged item from order 789?"
JSON Response:
{
  "intent": "RETURN_INFO",
  "entities": {
    "order_id": "789"
  }
}

Query: "Which version of the SQL agent is active?"
JSON Response:
{
  "intent": "META_QUERY",
  "entities": {}
}

Query: "My package for order 555 arrived broken."
JSON Response:
{
  "intent": "PROBLEM_REPORT",
  "entities": {
    "order_id": "555"
  }
}

Query: "Tell me about your best-selling shoes."
JSON Response:
{
  "intent": "SQL_QUERY_GENERAL",
  "entities": {}
}

Query: "Can I cancel an order after payment?"
JSON Response:
{
  "intent": "PROBLEM_REPORT",
  "entities": {}
}
""" 
    formatted_prompt = prompt_template.replace("{user_query}", user_query).replace("{structure_json}", structure_json)

    # formatted_prompt = prompt_template.format(user_query=user_query,struture_json=structure_json)

    llm_response_str = get_llm_response(
        prompt=formatted_prompt,
        model=config.get("llm_model"), # Use model from config
        json_mode=False # Request JSON output
    )



    if "Error:" in llm_response_str: # Check if LLM call failed
        logger.error(f"{NODE_NAME}: LLM error: {llm_response_str}")
        return {"intent": "UNKNOWN", "error_message": llm_response_str, "processing_steps_versions": {NODE_NAME: config.get("version")}}

    try:
        partial_result = {}
        parsed_response = json.loads(llm_response_str)
        intent = parsed_response.get("intent", "UNKNOWN")
        entities = parsed_response.get("entities", {})
        logger.info(f"{NODE_NAME}: Intent='{intent}', Entities='{entities}'")
        
        # Update processing steps versions
        current_versions = state.get("processing_steps_versions", {})
        current_versions[NODE_NAME] = config.get("version")

        node_end_time = time.perf_counter()
        current_latencies[NODE_NAME] = round(node_end_time - node_start_time, 4)
        partial_result["node_latencies"] = current_latencies
        partial_result["node_execution_order"] = current_order

        return {"intent": intent, "entities": entities, "processing_steps_versions": current_versions,**partial_result}
    except json.JSONDecodeError:
        logger.error(f"{NODE_NAME}: Failed to parse LLM JSON response: {llm_response_str}")
        return {"intent": "UNKNOWN", "error_message": "Failed to parse intent from LLM.", "processing_steps_versions": {NODE_NAME: config.get("version")}}
    except Exception as e:
        logger.error(f"{NODE_NAME}: Unexpected error: {e}")
        return {"intent": "UNKNOWN", "error_message": str(e), "processing_steps_versions": {NODE_NAME: config.get("version")}}
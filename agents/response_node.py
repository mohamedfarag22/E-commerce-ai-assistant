# agents/response_node.py
from utils import get_llm_response, load_prompt_from_path, get_node_config, logger
from graph_state import AgentState
import time 

NODE_NAME = "response_synthesizer"

def response_synthesis_node(state: AgentState) -> dict:
    """
    Synthesizes a final user-facing response from intermediate results.
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
    intent = state.get("intent")  # Get the intent from state
    sql_result = state.get("sql_query_result")
    rag_summary = state.get("rag_summary")
    error_msg = state.get("error_message")
    intermediate_response = state.get("intermediate_response")

    # Handle greeting intent first
    if intent == "OUT_OF_CONTEXT":
        greeting_prompt = """You are a friendly e-commerce assistant. Generate a warm, professional greeting response.
        Examples:
        - "Hello! How can I assist you with your order today?"
        - "Hi there! Welcome to our support. How may I help you?"
        - "Greetings! What can I do for you today?"
        
        Your greeting response:"""
        final_answer = get_llm_response(
            prompt=greeting_prompt,
            model=config.get("llm_model")
        )
        logger.info(f"{NODE_NAME}: Generated greeting response: {final_answer}")
    elif intermediate_response:
        logger.info(f"{NODE_NAME}: Using intermediate response: {intermediate_response}")
        final_answer = intermediate_response
    elif error_msg and not (sql_result or rag_summary):
        logger.warning(f"{NODE_NAME}: Error from previous step: {error_msg}")
        final_answer = f"I encountered an issue trying to process your request: {error_msg}. Please try rephrasing or ask something else."
    else:
        # Prepare context for non-greeting responses that need LLM formatting
        context_for_llm = ""
        if sql_result is not None:
            context_for_llm = f"The database query for '{user_query}' returned: {sql_result}"
        elif rag_summary:
            context_for_llm = f"Information found regarding '{user_query}': {rag_summary}"
        else:
            context_for_llm = "I couldn't find a specific answer for your query. Please try rephrasing or asking something else."

        if not intermediate_response and not error_msg:
            prompt_template = load_prompt_from_path(config["prompt_path"])
            formatted_prompt = prompt_template.format(user_query=user_query, information=context_for_llm)
            logger.info(f"formatted prompt of response node {formatted_prompt}")

            try:
                final_answer = get_llm_response(
                    prompt=formatted_prompt,
                    model=config.get("llm_model")
                )
                if "Error:" in final_answer:
                    logger.error(f"{NODE_NAME}: LLM error during final response synthesis: {final_answer}")
                    final_answer = f"I encountered an issue while processing your request. Here's what I found: {context_for_llm[:200]}..."
            except Exception as e:
                logger.error(f"{NODE_NAME}: Error occurred while generating response: {str(e)}")
                final_answer = "I encountered an issue while processing your request. Please try again later."

    logger.info(f"{NODE_NAME}: Final answer: {final_answer}")

    # Update processing steps versions
    current_versions = state.get("processing_steps_versions", {})
    current_versions[NODE_NAME] = config.get("version")
    partial_result = {}
    node_end_time = time.perf_counter()
    current_latencies[NODE_NAME] = round(node_end_time - node_start_time, 4)
    partial_result["node_latencies"] = current_latencies
    partial_result["node_execution_order"] = current_order

    return {"final_answer": final_answer, "processing_steps_versions": current_versions,**partial_result}
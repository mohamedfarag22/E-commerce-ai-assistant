# agents/meta_query_node.py
from utils import get_node_config, load_agent_registry, logger, get_llm_response, load_prompt_from_path
from graph_state import AgentState
import json
import time 

NODE_NAME = "meta_query_handler"

def classify_meta_intent(user_query: str, agent_types: list, model: str = "gpt-4o") -> dict:
    agent_list_str = agent_types
    classification_prompt = f"""
You are an intelligent AI system that helps route internal meta-queries.
DO NOT include markdown code formatting (like ```json or ```) in your output. Only return plain.

Given this user query:
\"{user_query}\"

From the following list of agent types:
{agent_list_str}

Determine:
1. Is the user asking about version info? (yes/no)
2. Which agent type from the list (if any) is being referred to?

Respond strictly in JSON format like this:
{{
  "is_version_query": true,
  "target_node": "retrieval_processor"
}}

If unsure about the agent type, set "target_node" to null.
"""
    try:
        response = get_llm_response(prompt=classification_prompt, model=model)
        return json.loads(response)
    except Exception as e:
        logger.warning(f"Failed to parse LLM classification: {e}")
        return {"is_version_query": False, "target_node": None}

def meta_query_node(state: AgentState) -> dict:

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
    agent_types = """
agent_types:
- intent_parser
- sql_processor
- retrieval_processor
- response_synthesizer
- meta_query_handler

"""
    model = config.get("llm_model", "gpt-4o")

    intent_result = classify_meta_intent(user_query, agent_types, model)

    answer = ""
    if intent_result["is_version_query"]:
        target_node = intent_result.get("target_node")
        if target_node:
            version = state.get("processing_steps_versions", {}).get(
                target_node, "N/A (not used yet or unknown)"
            )
            if version == "N/A (not used yet or unknown)":
                node_config = get_node_config(target_node)
                version = node_config.get("version", "N/A") if node_config else "N/A"
            answer = f"The '{target_node}' agent is configured to use version: {version}. \n don't mention the number of the version \n if version of agent is 1.0 named 'Goodie1' ,if version 1.1 named 'Goodie2' etc."
        else:
            registry = load_agent_registry()
            active_versions = registry.get("active_node_versions", {})
            if active_versions:
                answer = "Currently active component versions are:\n"
                for node, ver in active_versions.items():
                    answer += f"- {node}: {ver}\n"
            else:
                answer = "Version information is not readily available."
    else:
        answer = "I can answer questions about my system component versions. What would you like to know?"
    partial_result = {}

    prompt_template = load_prompt_from_path(config["prompt_path"])
    if prompt_template and config.get("llm_model"):
        formatted_prompt = prompt_template.format(information_found=answer, user_query=user_query)
        final_meta_answer = get_llm_response(prompt=formatted_prompt, model=model)
    else:
        final_meta_answer = answer
    node_end_time = time.perf_counter()
    current_latencies[NODE_NAME] = round(node_end_time - node_start_time, 4)
    partial_result["node_latencies"] = current_latencies
    partial_result["node_execution_order"] = current_order
    logger.info(f"{NODE_NAME}: Meta answer: {final_meta_answer}")
    current_versions = state.get("processing_steps_versions", {})
    current_versions[NODE_NAME] = config.get("version")


    return {"intermediate_response": final_meta_answer, "processing_steps_versions": current_versions,**partial_result}
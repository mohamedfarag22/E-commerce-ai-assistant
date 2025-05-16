# agents/sql_node.py
import sqlite3
import time
import json
from utils import get_llm_response, load_prompt_from_path, get_node_config, logger, DB_SCHEMA_FOR_PROMPT
from graph_state import AgentState
import time

NODE_NAME = "sql_processor"

def sql_node(state: AgentState) -> dict:
    """
    Generates SQL from user query (if intent is SQL-related), executes it.
    Updates state with sql_query_generated, sql_query_result.
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
    entities = state.get("entities", {})
    
    # Construct a more targeted query for the LLM if entities are present
    # This depends on how the intent parser and this node are designed to interact
    # For now, we pass the original query and expect the SQL prompt to handle it.

    prompt_template = load_prompt_from_path(config["prompt_path"])
    # The prompt should include the DB_SCHEMA_FOR_PROMPT placeholder or have it embedded
    # prompts/sql/v1_0_schema.txt:
    """
    You are an AI assistant that translates natural language questions into SQL queries for an e-commerce database.
    {db_schema}

    Only generate valid SQLite SELECT queries. Do not generate INSERT, UPDATE, or DELETE queries.
    If the question cannot be answered with a SELECT query, or seems malicious, respond with "I cannot answer this question."

    User Question: {user_query}
    SQL Query:
    """
    serach_term = """
'%{search_term}%'
"""
    formatted_prompt = prompt_template.format(serach_term=serach_term,db_schema=DB_SCHEMA_FOR_PROMPT, user_query=user_query)
    
    generated_sql = get_llm_response(
        prompt=formatted_prompt,
        model=config.get("llm_model")
    )

    if "Error:" in generated_sql or "I cannot answer this question" in generated_sql:
        logger.warning(f"{NODE_NAME}: SQL generation failed or refused: {generated_sql}")
        return {
            "sql_query_generated": generated_sql, 
            "sql_query_result": None, 
            "error_message": "SQL generation failed or request refused.",
            "processing_steps_versions": {**state.get("processing_steps_versions", {}), NODE_NAME: config.get("version")}
        }
    
    logger.info(f"{NODE_NAME}: Generated SQL: {generated_sql}")

    # Execute SQL
    db_path = config["db_path"]
    results = None
    error_msg = None
    try:
        conn = sqlite3.connect(db_path)
        # To return results as dictionaries instead of tuples
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        cursor.execute(generated_sql)
        query_results_raw = cursor.fetchall()
        # Convert Row objects to simple dictionaries for JSON serialization if needed later
        results = [dict(row) for row in query_results_raw]
        conn.close()
        logger.info(f"{NODE_NAME}: SQL execution successful, {len(results)} rows returned.")
        logger.info(f"the result of SQL {results}: ")

    except sqlite3.Error as e:
        logger.error(f"{NODE_NAME}: Database error: {e} for query: {generated_sql}")
        error_msg = f"Database error: {e}"
        results = None # Ensure results is None on error
    except Exception as e: # Catch other potential errors
        logger.error(f"{NODE_NAME}: Unexpected error executing SQL: {e} for query: {generated_sql}")
        error_msg = f"Unexpected error during SQL execution: {e}"
        results = None
    partial_result = {}
    # Update processing steps versions
    current_versions = state.get("processing_steps_versions", {})
    current_versions[NODE_NAME] = config.get("version")
    node_end_time = time.perf_counter()
    current_latencies[NODE_NAME] = round(node_end_time - node_start_time, 4)
    partial_result["node_latencies"] = current_latencies
    partial_result["node_execution_order"] = current_order
    return {
        "sql_query_generated": generated_sql,
        "sql_query_result": results,
        "error_message": error_msg,
        "processing_steps_versions": current_versions,
        **partial_result
    }
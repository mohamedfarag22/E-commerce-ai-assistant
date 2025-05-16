import pytest
import json
import os
from graph_state import AgentState
from run_evaluation import compare_sql_queries, compare_sql_results

# --- Load JSON data directly at module level for parametrization ---
def _load_json_for_test_module(relative_path):
    test_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(test_dir, '..', '..'))
    full_path = os.path.join(project_root, relative_path)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback for CI if files are at root during checkout for some reason
        alt_full_path = os.path.join(project_root, os.path.basename(relative_path))
        if os.path.exists(alt_full_path):
            with open(alt_full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        pytest.fail(f"Failed to load golden query JSON for parametrize: {full_path} or {alt_full_path} not found.", pytrace=False)
    except Exception as e:
        pytest.fail(f"Failed to load/parse golden query JSON {full_path}. Error: {e}", pytrace=False)
    return []

SQL_TEST_DATA = _load_json_for_test_module("data/golden_queries_sql.json")
# --- End of data loading ---

@pytest.mark.parametrize("golden_query_data", SQL_TEST_DATA)
def test_sql_pipeline_from_json_with_latency_check(golden_query_data, langgraph_app, mock_initial_state, mocker):
    assert isinstance(golden_query_data, dict), \
        f"Test setup error: golden_query_data is not a dict. Got: {golden_query_data} (type: {type(golden_query_data)})"
        
    user_query = golden_query_data["query"]
    expected_sql = golden_query_data["expected_sql"]
    expected_result_str = golden_query_data["expected_result"]

    llm_call_log = []

    def mock_llm_router_sql(prompt: str, model: str, json_mode: bool = False, **kwargs):
        # print(f"\nDEBUG SQL MOCK LLM ROUTER. Prompt starts with: {prompt[:200]}\n")
        
        # Check for CI-specific dummy prompt first, then for actual prompt content.
        # --- INTENT PARSER ---
        if ("CI Intent Prompt:" in prompt and "{user_query}" in prompt and "{structure_json}" in prompt) or \
           ("classifying user intent" in prompt and "## User Query:" in prompt):
            llm_call_log.append("intent_parser_triggered")
            # Determine mocked intent based on the current user_query for this test case
            if "status of order" in user_query.lower() or "order #" in user_query.lower():
                order_id_match = user_query.split("#")[-1].split("?")[0].strip().split(" ")[0]
                return json.dumps({"intent": "ORDER_STATUS", "entities": {"order_id": order_id_match}})
            elif "nike air max" in user_query.lower() or "in stock" in user_query.lower():
                 return json.dumps({"intent": "PRODUCT_AVAILABILITY", "entities": {"product_name": "Nike Air Max"}})
            elif "email of the customer" in user_query.lower() and "order #" in user_query.lower():
                 parts = user_query.split("#")
                 order_id_match = "UNKNOWN_ORDER_ID"
                 if len(parts) > 1: order_id_match = parts[1].split("?")[0].strip().split(" ")[0]
                 return json.dumps({"intent": "SQL_QUERY_GENERAL", "entities": {"order_id": order_id_match}})
            # Default intent for SQL queries if not matched above (including dummy CI query)
            return json.dumps({"intent": "SQL_QUERY_GENERAL", "entities": {}})

        # --- SQL GENERATOR ---
        elif ("CI SQL Prompt:" in prompt and "{db_schema}" in prompt and "{user_query}" in prompt) or \
             ("translates natural language questions" in prompt and "into SQLite SELECT queries" in prompt and "## Database Schema" in prompt):
            llm_call_log.append("sql_generator_triggered")
            return expected_sql

        # --- RESPONSE SYNTHESIZER ---
        elif ("CI Response Prompt:" in prompt and "{information}" in prompt and "{user_query}" in prompt) or \
             ("friendly and professional AI customer support assistant" in prompt and "Information Found by the System:" in prompt):
            llm_call_log.append("response_synthesizer_triggered")
            try:
                results_data = json.loads(expected_result_str)
                return f"Mocked final response based on SQL. Data: {results_data}."
            except: # Fallback if expected_result_str is not valid JSON for some reason
                return f"Mocked final response for query: {user_query}"
        
        llm_call_log.append(f"fallback_triggered_sql: {prompt[:100]}")
        return "Fallback: UNMATCHED PROMPT in SQL mock_llm_router"

    # Patch get_llm_response in each module where it's imported and used by the nodes in this pipeline
    mocker.patch('agents.intent_parser_node.get_llm_response', side_effect=mock_llm_router_sql)
    mocker.patch('agents.sql_node.get_llm_response', side_effect=mock_llm_router_sql)
    mocker.patch('agents.response_node.get_llm_response', side_effect=mock_llm_router_sql)

    current_initial_state = mock_initial_state.copy()
    current_initial_state["original_query"] = user_query
    
    final_state = langgraph_app.invoke(current_initial_state)

    # Assertions
    generated_sql = final_state.get("sql_query_generated")
    generated_results_from_db = final_state.get("sql_query_result")

    assert "intent_parser_triggered" in llm_call_log, \
        f"Intent parser LLM mock not triggered for query '{user_query}'. LLM Log: {llm_call_log}. Final State: {final_state}"
    
    mocked_intent = final_state.get("intent")
    sql_related_intents = ["ORDER_STATUS", "PRODUCT_AVAILABILITY", "SQL_QUERY_GENERAL"]

    if mocked_intent in sql_related_intents:
        assert "sql_generator_triggered" in llm_call_log, \
            f"SQL generator LLM mock not triggered. Query: '{user_query}', Intent: {mocked_intent}. Log: {llm_call_log}"
        assert generated_sql is not None, \
            f"SQL query was not generated. Query: '{user_query}', Intent: {mocked_intent}. Log: {llm_call_log}"
        
        is_sql_query_correct, _ = compare_sql_queries(generated_sql, expected_sql)
        assert is_sql_query_correct, \
            f"Generated SQL did not match expected SQL for query: '{user_query}'.\n" \
            f"Expected: '{expected_sql}'\nGot: '{generated_sql}'"

        is_sql_result_correct, _ = compare_sql_results(generated_results_from_db, expected_result_str)
        assert is_sql_result_correct, \
            f"Actual SQL result from DB did not match expected. Query: '{user_query}'.\n" \
            f"Expected (from JSON): '{expected_result_str}'\nGot (from DB execution of '{generated_sql}'): '{json.dumps(generated_results_from_db)}'"
        
        assert "response_synthesizer_triggered" in llm_call_log, \
            f"Response synthesizer LLM mock not triggered for SQL query '{user_query}'. Log: {llm_call_log}"
    
    assert final_state.get("final_answer") is not None, "A final answer should be generated."
    
    if mocked_intent in sql_related_intents:
        assert final_state.get("error_message") is None, \
            f"Pipeline produced an error for query '{user_query}' with intent '{mocked_intent}': {final_state.get('error_message')}. LLM Log: {llm_call_log}"

    node_latencies = final_state.get("node_latencies")
    node_execution_order = final_state.get("node_execution_order")
    assert node_latencies, "Node latencies dictionary is empty or missing."
    assert node_execution_order, "Node execution order list is empty."

    expected_nodes_in_path = ["intent_parser"]
    if mocked_intent in sql_related_intents:
        expected_nodes_in_path.extend(["sql_processor", "response_synthesizer"])
    
    for node_name in expected_nodes_in_path:
        assert node_name in node_execution_order, f"'{node_name}' not in execution order for query '{user_query}'. Order: {node_execution_order}"
        assert node_name in node_latencies, f"Latency for '{node_name}' missing for query '{user_query}'. Latencies: {node_latencies}"
        assert isinstance(node_latencies[node_name], float) and node_latencies[node_name] >= 0
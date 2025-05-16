import pytest
import json
import os
from graph_state import AgentState
from run_evaluation import compare_sql_queries, compare_sql_results

def _load_json_for_test_module(relative_path):
    test_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(test_dir, '..', '..'))
    full_path = os.path.join(project_root, relative_path)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        pytest.fail(f"Failed to load golden query JSON: {full_path}. Error: {e}", pytrace=False)
    return []

SQL_TEST_DATA = _load_json_for_test_module("data/golden_queries_sql.json")

@pytest.mark.parametrize("golden_query_data", SQL_TEST_DATA)
def test_sql_pipeline_from_json_with_latency_check(golden_query_data, langgraph_app, mock_initial_state, mocker):
    assert isinstance(golden_query_data, dict), \
        f"Test setup error: golden_query_data is not a dict. Got: {golden_query_data} (type: {type(golden_query_data)})"
        
    user_query = golden_query_data["query"]
    expected_sql = golden_query_data["expected_sql"]
    expected_result_str = golden_query_data["expected_result"]

    llm_call_log = []
    def mock_llm_router(prompt: str, model: str, json_mode: bool = False, **kwargs):
            # print(f"\nDEBUG SQL MOCK LLM ROUTER. Prompt starts with: {prompt[:250]}\n") # Increased preview
            
            # Intent Parser Node
            if "classifying user intent" in prompt and "## User Query:" in prompt:
                llm_call_log.append("intent_parser_triggered")
                # ... (your intent logic as before, ensuring it returns a JSON string) ...
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
                return json.dumps({"intent": "SQL_QUERY_GENERAL", "entities": {}})

            # SQL Generator Node
            # Check for phrases unique to the SQL generation prompt (prompts/sql/v1_0_schema.txt)
            elif ("translates natural language questions" in prompt and 
                  "into SQLite SELECT queries" in prompt and 
                  "## Database Schema" in prompt): # More specific check
                llm_call_log.append("sql_generator_triggered")
                return expected_sql

            # Response Synthesizer Node
            elif "friendly and professional AI customer support assistant" in prompt and "Information Found by the System:" in prompt:
                llm_call_log.append("response_synthesizer_triggered")
                # ... (your response synth logic) ...
                try:
                    results_data = json.loads(expected_result_str)
                    return f"Mocked final response based on SQL. Data: {results_data}."
                except:
                    return f"Mocked final response for query: {user_query}"
            
            llm_call_log.append(f"fallback_triggered_sql: {prompt[:200]}") # Increased preview
            return "Fallback: UNMATCHED PROMPT in SQL mock_llm_router"

    # Patch get_llm_response in each module where it's imported and used
    mocker.patch('agents.intent_parser_node.get_llm_response', side_effect=mock_llm_router)
    mocker.patch('agents.sql_node.get_llm_response', side_effect=mock_llm_router)
    mocker.patch('agents.response_node.get_llm_response', side_effect=mock_llm_router)
    # No need to patch 'utils.get_llm_response' if all nodes import from utils and we patch the node's import.
    # However, patching 'utils.get_llm_response' is fine if all nodes use that same reference.
    # The key is consistency. The logs showed 'utils.get_llm_response' being called, so patching that
    # should have worked if the reference wasn't already "cached" by the importing modules.
    # Patching at the point of use is safest.

    current_initial_state = mock_initial_state.copy()
    current_initial_state["original_query"] = user_query
    final_state = langgraph_app.invoke(current_initial_state)

    # Assertions (as before, with more context in error messages)
    # ... (your assertions for generated_sql, results, final_answer, errors, latency)
    generated_sql = final_state.get("sql_query_generated")
    generated_results_from_db = final_state.get("sql_query_result")

    assert "intent_parser_triggered" in llm_call_log, f"Intent parser LLM mock not triggered for query '{user_query}'. LLM Log: {llm_call_log}. Final State: {final_state}"
    mocked_intent = final_state.get("intent")
    sql_related_intents = ["ORDER_STATUS", "PRODUCT_AVAILABILITY", "SQL_QUERY_GENERAL"]

    if mocked_intent in sql_related_intents:
        assert "sql_generator_triggered" in llm_call_log, f"SQL generator LLM mock not triggered. Query: '{user_query}', Intent: {mocked_intent}. Log: {llm_call_log}"
        assert generated_sql is not None, f"SQL query was not generated. Query: '{user_query}', Intent: {mocked_intent}. Log: {llm_call_log}"
        # ... (rest of SQL assertions) ...
        is_sql_query_correct, _ = compare_sql_queries(generated_sql, expected_sql)
        assert is_sql_query_correct, f"Generated SQL mismatch for '{user_query}'. Expected: '{expected_sql}', Got: '{generated_sql}'"
        is_sql_result_correct, _ = compare_sql_results(generated_results_from_db, expected_result_str)
        assert is_sql_result_correct, f"SQL result mismatch for '{user_query}'. Expected: '{expected_result_str}', Got: '{json.dumps(generated_results_from_db)}'"
        assert "response_synthesizer_triggered" in llm_call_log, f"Response synthesizer for SQL not triggered. Query: '{user_query}'. Log: {llm_call_log}"
        assert final_state.get("error_message") is None, f"Error during SQL path: {final_state.get('error_message')}. Log: {llm_call_log}"
    
    assert final_state.get("final_answer") is not None, "Final answer missing."
    # ... (latency assertions)
    node_latencies = final_state.get("node_latencies")
    node_execution_order = final_state.get("node_execution_order")
    assert node_latencies and node_execution_order, "Latency/order info missing."
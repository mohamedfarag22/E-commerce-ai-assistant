import pytest
import json
import os
from graph_state import AgentState
from run_evaluation import compare_retrieval_sources

def _load_json_for_test_module(relative_path):
    test_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(test_dir, '..', '..'))
    full_path = os.path.join(project_root, relative_path)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        alt_full_path = os.path.join(project_root, os.path.basename(relative_path))
        if os.path.exists(alt_full_path):
             with open(alt_full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        pytest.fail(f"Failed to load golden query JSON: {full_path} or {alt_full_path} not found.", pytrace=False)
    except Exception as e:
        pytest.fail(f"Failed to load/parse golden query JSON {full_path}. Error: {e}", pytrace=False)
    return []

RETRIEVAL_TEST_DATA = _load_json_for_test_module("data/golden_queries_retrieval.json")

@pytest.mark.parametrize("golden_query_data", RETRIEVAL_TEST_DATA)
def test_retrieval_pipeline_from_json_with_latency_check(golden_query_data, langgraph_app, mock_initial_state, mocker):
    assert isinstance(golden_query_data, dict), \
        f"Test setup error: golden_query_data is not a dict. Got: {golden_query_data} (type: {type(golden_query_data)})"

    user_query = golden_query_data["query"]
    expected_source_filename = golden_query_data["expected_answer_source"]

    llm_call_log = []
    expected_rag_summary_content = f"Mocked RAG summary for '{user_query}' from {expected_source_filename}."

    def mock_llm_router_retrieval(prompt: str, model: str, json_mode: bool = False, **kwargs):
        # print(f"\nDEBUG RETRIEVAL MOCK LLM ROUTER. Prompt starts with: {prompt[:200]}\n")
        
        # --- INTENT PARSER ---
        if ("CI Intent Prompt:" in prompt and user_query in prompt) or \
           ("classifying user intent" in prompt and "## User Query:" in prompt):
            llm_call_log.append("intent_parser_triggered")
            intent_payload = {"intent": "PROBLEM_REPORT", "entities": {}} # Default
            if "return" in user_query.lower() or "refund" in user_query.lower() or "wrong item" in user_query.lower() or "cancel" in user_query.lower() or ("dummy ci retrieval query" in user_query.lower() and "return" in expected_source_filename): # Make dummy CI query more specific
                 intent_payload["intent"] = "RETURN_INFO"
            elif "shipping" in user_query.lower() or "package arrives late" in user_query.lower() or ("dummy ci retrieval query" in user_query.lower() and "shipping" in expected_source_filename):
                 intent_payload["intent"] = "SHIPPING_INFO"
            return json.dumps(intent_payload)
        
        # --- RAG SUMMARIZER (within retrieval_node) ---
        elif ("CI RAG Prompt:" in prompt and "{context_str}" in prompt) or \
             ("Advanced AI assistant providing strictly context-based answers" in prompt and "**Inputs**:" in prompt):
            llm_call_log.append("rag_summarizer_triggered")
            return expected_rag_summary_content
        
        # --- RESPONSE SYNTHESIZER ---
        # Assumes response_node uses intermediate_response (RAG summary) directly
        # If it reformats, this condition needs to match that prompt.
        # For this test, we primarily check if it's *not* called for the standard formatting prompt
        # if the Option A (direct use of RAG summary) is correct.
        elif ("CI Response Prompt:" in prompt and "{information}" in prompt) or \
             ("friendly and professional AI customer support assistant" in prompt and "Information Found by the System:" in prompt):
            llm_call_log.append("response_synthesizer_triggered_for_formatting") # Specific name
            return f"Formatted final answer for retrieval query: {user_query}"
            
        llm_call_log.append(f"fallback_triggered_retrieval: {prompt[:100]}")
        return "Fallback: UNMATCHED PROMPT in Retrieval mock_llm_router"

    mocker.patch('agents.intent_parser_node.get_llm_response', side_effect=mock_llm_router_retrieval)
    mocker.patch('agents.retrieval_node.get_llm_response', side_effect=mock_llm_router_retrieval)
    mocker.patch('agents.response_node.get_llm_response', side_effect=mock_llm_router_retrieval)

    mocker.patch('utils.get_embeddings', return_value=[[0.05] * 1536])
    mock_indices_array = mocker.MagicMock(name="mock_faiss_indices_array"); mock_indices_array.size = 1
    mock_indices_array.__getitem__ = lambda s, k: [0] if k == 0 else mocker.MagicMock()
    mock_distances_array = mocker.MagicMock(name="mock_faiss_distances_array"); mock_distances_array.size = 1
    mock_distances_array.__getitem__ = lambda s, k: [0.05] if k == 0 else mocker.MagicMock()
    mock_faiss_search_method = mocker.MagicMock(return_value=(mock_distances_array, mock_indices_array), name="mock_faiss_search_method")
    mock_faiss_index_object = mocker.MagicMock(name="mock_faiss_index_object_for_test"); mock_faiss_index_object.search = mock_faiss_search_method
    mocker.patch('agents.retrieval_node._load_retrieval_assets', return_value=True)
    mocker.patch('agents.retrieval_node._faiss_index', mock_faiss_index_object)
    mock_retrieved_text_content = f"This is mocked text from '{expected_source_filename}' for '{user_query}'."
    mock_node_metadata_list = [{"source": expected_source_filename, "text": mock_retrieved_text_content}]
    mocker.patch('agents.retrieval_node._metadata', mock_node_metadata_list)

    current_initial_state = mock_initial_state.copy()
    current_initial_state["original_query"] = user_query
    
    final_state = langgraph_app.invoke(current_initial_state)

    retrieved_contexts = final_state.get("retrieved_contexts")
    assert "intent_parser_triggered" in llm_call_log, f"Intent parser mock not triggered. Query: '{user_query}'. Log: {llm_call_log}. State: {final_state}"
    
    mocked_intent = final_state.get("intent")
    retrieval_related_intents = ["RETURN_INFO", "SHIPPING_INFO", "PROBLEM_REPORT"]

    if mocked_intent in retrieval_related_intents:
        assert retrieved_contexts and len(retrieved_contexts) > 0, f"No contexts retrieved. Query: '{user_query}'. Log: {llm_call_log}"
        assert retrieved_contexts[0]["text"] == mock_retrieved_text_content
        is_source_correct, _, actual_top_source = compare_retrieval_sources(retrieved_contexts, expected_source_filename)
        assert is_source_correct, f"Source mismatch. Expected: '{expected_source_filename}', Got: '{actual_top_source}'. Query: '{user_query}'"
        
        assert "rag_summarizer_triggered" in llm_call_log, f"RAG summarizer mock not triggered. Query: '{user_query}'. Log: {llm_call_log}"
        
        # Assuming response_node uses intermediate_response (RAG summary) directly (Option A)
        assert final_state.get("final_answer") == expected_rag_summary_content, \
            f"Final answer mismatch. Expected RAG summary: '{expected_rag_summary_content}', Got: '{final_state.get('final_answer')}'"
        
        # If response_node uses intermediate_response directly, its LLM for formatting shouldn't be called.
        assert "response_synthesizer_triggered_for_formatting" not in llm_call_log, \
             f"Response synthesizer LLM (formatting) should NOT have been called if intermediate_response from RAG was used. Log: {llm_call_log}"
        
        assert final_state.get("error_message") is None, \
            f"Pipeline error for '{user_query}' with intent '{mocked_intent}': {final_state.get('error_message')}. Log: {llm_call_log}"

    node_latencies = final_state.get("node_latencies")
    node_execution_order = final_state.get("node_execution_order")
    assert node_latencies and node_execution_order, "Latency/order info missing."
    
    expected_nodes_in_path = ["intent_parser"]
    if mocked_intent in retrieval_related_intents:
         expected_nodes_in_path.extend(["retrieval_processor", "response_synthesizer"])
    
    for node_name in expected_nodes_in_path:
        assert node_name in node_execution_order, f"'{node_name}' not in execution order {node_execution_order} for '{user_query}'"
        assert node_name in node_latencies, f"Latency for '{node_name}' missing for '{user_query}'. Latencies: {node_latencies}"
        assert isinstance(node_latencies[node_name], float) and node_latencies[node_name] >= 0
# tests/agents/test_retrieval_node.py
import pytest
from jsonschema import validate, ValidationError
from agents.retrieval_node import retrieval_node, NODE_NAME # Make sure NODE_NAME is defined
from graph_state import AgentState # For type hinting

# ... (your schema definitions: RETRIEVED_CONTEXT_ITEM_SCHEMA, RETRIEVAL_NODE_FUNCTION_OUTPUT_SCHEMA) ...

@pytest.fixture
def retrieval_node_config_fixture(): # Example fixture for node config
    return {
        "version": "1.0-test-retrieval",
        "vector_store_path": "fake/faiss.index",
        "metadata_store_path": "fake/metadata.json",
        "embedding_model": "text-embedding-test",
        "rag_prompt_path": "fake/rag_prompt.txt",
        "llm_model_for_rag": "gpt-test-rag",
        "top_k": 1
    }


# Use the general mock_initial_state fixture from conftest.py
def test_retrieval_node_success_and_schema(mocker, mock_initial_state, retrieval_node_config_fixture):
    # Customize the initial state for this specific test
    current_test_state = mock_initial_state.copy()
    current_test_state["original_query"] = "Tell me about return policy for damaged goods."
    # For a unit test of retrieval_node, you might assume intent is already classified
    current_test_state["intent"] = "RETURN_INFO"
    current_test_state["entities"] = {"item_condition": "damaged"}


    # 1. Mock dependencies of retrieval_node
    mocker.patch('agents.retrieval_node.get_node_config', return_value=retrieval_node_config_fixture)
    mocker.patch('agents.retrieval_node._load_retrieval_assets', return_value=True) # Assume assets load

    # Mock get_embeddings from utils (or wherever retrieval_node imports it from)
    mocker.patch('agents.retrieval_node.get_embeddings', return_value=[[0.1]*1536]) # Dummy embedding

    # Mock FAISS search (as done in integration tests, but simpler for unit)
    mock_indices_array = mocker.MagicMock(name="mock_indices_array_unit")
    mock_indices_array.size = 1
    mock_indices_array.__getitem__ = lambda s, k: [0] if k == 0 else [] # Return index 0
    mock_distances_array = mocker.MagicMock(name="mock_distances_array_unit")
    mock_distances_array.__getitem__ = lambda s, k: [0.1] if k == 0 else []
    
    mock_faiss_search = mocker.MagicMock(return_value=(mock_distances_array, mock_indices_array))
    mock_faiss_index_instance = mocker.MagicMock(name="mock_faiss_index_unit")
    mock_faiss_index_instance.search = mock_faiss_search
    mocker.patch('agents.retrieval_node._faiss_index', mock_faiss_index_instance)

    # Mock metadata
    mock_retrieved_doc = {"source": "return_policy.txt", "text": "Damaged goods can be returned...", "distance": 0.1}
    mocker.patch('agents.retrieval_node._metadata', [mock_retrieved_doc]) # _metadata is a list

    # Mock the RAG LLM call within retrieval_node
    mocked_rag_summary = "Summary: return damaged goods according to policy."
    mocker.patch('agents.retrieval_node.get_llm_response', return_value=mocked_rag_summary)

    # 2. Call the node function with the customized state
    output_dict = retrieval_node(current_test_state)

    # 3. Validate schema
    # (Ensure RETRIEVAL_NODE_FUNCTION_OUTPUT_SCHEMA is defined in this file or imported)
    # try:
    #     validate(instance=output_dict, schema=RETRIEVAL_NODE_FUNCTION_OUTPUT_SCHEMA)
    # except ValidationError as e:
    #     pytest.fail(f"retrieval_node output schema validation failed: {e.message}\nInstance: {output_dict}")

    # 4. Assert specific values
    assert output_dict.get("rag_summary") == mocked_rag_summary
    assert output_dict.get("intermediate_response") == mocked_rag_summary
    assert len(output_dict.get("retrieved_contexts", [])) == 1
    assert output_dict.get("retrieved_contexts")[0]["source"] == "return_policy.txt"
    assert output_dict.get("error_message") is None
    assert NODE_NAME in output_dict.get("node_latencies", {})
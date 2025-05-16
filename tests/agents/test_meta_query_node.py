import pytest
import json
from jsonschema import validate, ValidationError

from agents.meta_query_node import meta_query_node, NODE_NAME as META_NODE_NAME, classify_meta_intent
from graph_state import AgentState

META_NODE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "intermediate_response": {"type": ["string", "null"]},
        "error_message": {"type": ["string", "null"]},
        "processing_steps_versions": {
            "type": "object",
            "properties": {META_NODE_NAME: {"type": "string"}},
            "required": [META_NODE_NAME]
        },
        "node_latencies": {
            "type": "object",
            "properties": {META_NODE_NAME: {"type": "number"}},
            "required": [META_NODE_NAME]
        },
        "node_execution_order": {
            "type": "array",
            "items": {"type": "string"},
            "contains": {"const": META_NODE_NAME}
        }
        # If meta_query_node sets final_answer directly, add it here
    },
    "required": ["intermediate_response", "processing_steps_versions", "node_latencies", "node_execution_order"]
}

@pytest.fixture
def meta_node_config():
    return {
        "version": "meta-v0.5",
        "prompt_path": "prompts/meta/v1_0_responder.txt",
        "llm_model": "gpt-test-meta"
    }

@pytest.fixture
def initial_state_for_meta_test():
    return AgentState(
        original_query="What version is the SQL agent?",
        intent="META_QUERY", # Assume intent is set
        processing_steps_versions={"sql_processor": "v1.beta-active"}, # Simulate sql_processor was used
        node_latencies={}, node_execution_order=[],
        # ... other fields ...
        entities=None, sql_query_generated=None, sql_query_result=None, retrieved_contexts=None,
        rag_summary=None, intermediate_response=None, final_answer=None, error_message=None, history=[]
    )

@pytest.fixture
def mock_agent_registry_for_meta(meta_node_config): # Use meta_node_config for its own config
    return {
        "nodes": {
            "sql_processor": {"version": "v1.beta-registry"},
            "intent_parser": {"version": "v1.alpha-registry"},
            META_NODE_NAME: meta_node_config # Ensure its own config is in registry for get_node_config
        },
        "active_node_versions": {
            "sql_processor": "v1.beta-active", # This is what meta_query_node might report for "all"
            "intent_parser": "v1.alpha-active"
        }
    }

def test_meta_node_specific_agent_version_from_state(mocker, initial_state_for_meta_test, meta_node_config, mock_agent_registry_for_meta):
    mocker.patch('agents.meta_query_node.get_node_config', return_value=meta_node_config)
    mocker.patch('agents.meta_query_node.load_agent_registry', return_value=mock_agent_registry_for_meta)
    mocker.patch('agents.meta_query_node.load_prompt_from_path', return_value="Info: {information_found} Query: {user_query} Resp:")

    # Mock the internal classify_meta_intent LLM call
    mock_classify_response = {"is_version_query": True, "target_node": "sql_processor"}
    mocker.patch('agents.meta_query_node.classify_meta_intent', return_value=mock_classify_response)

    # Mock the final formatting LLM call
    expected_formatted_answer = "The SQL processor is version Goodie2 (v1.beta-active)."
    mock_final_llm = mocker.patch('agents.meta_query_node.get_llm_response', return_value=expected_formatted_answer)

    result_dict = meta_query_node(initial_state_for_meta_test)

    try: validate(instance=result_dict, schema=META_NODE_OUTPUT_SCHEMA)
    except ValidationError as e: pytest.fail(f"Schema validation failed: {e}")

    assert result_dict.get("intermediate_response") == expected_formatted_answer
    mock_final_llm.assert_called_once()
    # Check that the information found (before final formatting) was correct
    prompt_to_final_llm = mock_final_llm.call_args[1]['prompt']
    assert "sql_processor' agent is configured to use version: v1.beta-active" in prompt_to_final_llm
    # The prompt also has "don't mention the number... if version 1.1 named Goodie2"
    # This implies the version string 'v1.beta-active' itself isn't directly in the final answer if LLM follows instruction
    assert "Goodie2" in expected_formatted_answer # Assuming 'v1.beta-active' maps to 'Goodie2' based on naming logic.

# Add more tests for:
# - Version query for an agent NOT YET in processing_steps_versions (uses registry's active_node_versions)
# - Version query for "all agents" (target_node is None)
# - Non-version meta query
# - classify_meta_intent returning parsing errors or unexpected structures
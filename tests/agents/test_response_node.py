import pytest
import json
from jsonschema import validate, ValidationError

from agents.response_node import response_synthesis_node, NODE_NAME as RESPONSE_NODE_NAME
from graph_state import AgentState

RESPONSE_NODE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer": {"type": ["string", "null"]},
        "error_message": {"type": ["string", "null"]}, # May pass through errors
        "processing_steps_versions": {
            "type": "object",
            "properties": {RESPONSE_NODE_NAME: {"type": "string"}},
            "required": [RESPONSE_NODE_NAME]
        },
        "node_latencies": {
            "type": "object",
            "properties": {RESPONSE_NODE_NAME: {"type": "number"}},
            "required": [RESPONSE_NODE_NAME]
        },
        "node_execution_order": {
            "type": "array",
            "items": {"type": "string"},
            "contains": {"const": RESPONSE_NODE_NAME}
        }
        # Add other state keys if response_node modifies them
    },
    "required": ["final_answer", "processing_steps_versions", "node_latencies", "node_execution_order"]
}

@pytest.fixture
def response_node_config():
    return {
        "version": "resp-v2.0",
        "prompt_path": "prompts/response/v1_0_format.txt", # Path used by the node
        "llm_model": "gpt-test-response"
    }

@pytest.fixture
def state_with_sql_result():
    return AgentState(
        original_query="What is order #123?",
        intent="SQL_QUERY",
        sql_query_result=[{"status": "Processed", "item_count": 2}],
        processing_steps_versions={}, node_latencies={}, node_execution_order=[],
        # Fill other fields as None or empty
        entities=None, sql_query_generated=None, retrieved_contexts=None,
        rag_summary=None, intermediate_response=None, final_answer=None, error_message=None, history=[]
    )

@pytest.fixture
def state_with_rag_summary():
    return AgentState(
        original_query="How to return?",
        intent="RETURN_INFO",
        rag_summary="You can return items within 30 days.",
        intermediate_response="You can return items within 30 days.", # Often set by retrieval_node
        processing_steps_versions={}, node_latencies={}, node_execution_order=[],
        entities=None, sql_query_generated=None, sql_query_result=None, retrieved_contexts=None,
        final_answer=None, error_message=None, history=[]
    )

@pytest.fixture
def state_with_intermediate_response():
    return AgentState(
        original_query="What version is X?",
        intent="META_QUERY",
        intermediate_response="Component X is version 1.0.",
        processing_steps_versions={}, node_latencies={}, node_execution_order=[],
        entities=None, sql_query_generated=None, sql_query_result=None, retrieved_contexts=None,
        rag_summary=None, final_answer=None, error_message=None, history=[]
    )

@pytest.fixture
def state_for_greeting():
    return AgentState(
        original_query="Hello there",
        intent="OUT_OF_CONTEXT", # Or GREETING
        processing_steps_versions={}, node_latencies={}, node_execution_order=[],
        entities=None, sql_query_generated=None, sql_query_result=None, retrieved_contexts=None,
        rag_summary=None, intermediate_response=None, final_answer=None, error_message=None, history=[]
    )


def test_response_node_with_sql_data(mocker, state_with_sql_result, response_node_config):
    mocker.patch('agents.response_node.get_node_config', return_value=response_node_config)
    mocker.patch('agents.response_node.load_prompt_from_path', return_value="Query: {user_query} Info: {information} Answer:")
    mocked_final_llm_answer = "Based on SQL, status is Processed, 2 items."
    mock_get_llm = mocker.patch('agents.response_node.get_llm_response', return_value=mocked_final_llm_answer)

    result_dict = response_synthesis_node(state_with_sql_result)

    try: validate(instance=result_dict, schema=RESPONSE_NODE_OUTPUT_SCHEMA)
    except ValidationError as e: pytest.fail(f"Schema validation failed (SQL data): {e}")
    
    assert result_dict.get("final_answer") == mocked_final_llm_answer
    mock_get_llm.assert_called_once() # LLM called for formatting
    args, kwargs = mock_get_llm.call_args
    assert "The database query for 'What is order #123?' returned: " in kwargs.get('prompt')


# Test case for when response_node uses intermediate_response directly (Option A from pipeline tests)
def test_response_node_uses_intermediate_response_directly(mocker, state_with_intermediate_response, response_node_config):
    mocker.patch('agents.response_node.get_node_config', return_value=response_node_config)
    mock_get_llm = mocker.patch('agents.response_node.get_llm_response') # Should NOT be called

    result_dict = response_synthesis_node(state_with_intermediate_response)

    try: validate(instance=result_dict, schema=RESPONSE_NODE_OUTPUT_SCHEMA)
    except ValidationError as e: pytest.fail(f"Schema validation failed (intermediate_response): {e}")

    assert result_dict.get("final_answer") == state_with_intermediate_response["intermediate_response"]
    mock_get_llm.assert_not_called() # Crucial check for this path

def test_response_node_greeting(mocker, state_for_greeting, response_node_config):
    mocker.patch('agents.response_node.get_node_config', return_value=response_node_config)
    mocked_greeting = "Hello! How can I help you today?"
    # This get_llm_response is for generating the greeting itself
    mock_get_llm_greeting = mocker.patch('agents.response_node.get_llm_response', return_value=mocked_greeting)
    
    result_dict = response_synthesis_node(state_for_greeting)

    try: validate(instance=result_dict, schema=RESPONSE_NODE_OUTPUT_SCHEMA)
    except ValidationError as e: pytest.fail(f"Schema validation failed (greeting): {e}")

    assert result_dict.get("final_answer") == mocked_greeting
    mock_get_llm_greeting.assert_called_once()
    args, kwargs = mock_get_llm_greeting.call_args
    assert "friendly e-commerce assistant" in kwargs.get('prompt').lower() # Check greeting prompt

# Add tests for error message handling, no data found scenario, etc.
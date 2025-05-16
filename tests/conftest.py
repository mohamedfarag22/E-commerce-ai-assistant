import sys
import os
import pytest
import json

# --- Python Path Management ---
# Ensure project root is on sys.path. This should be the very first executable code.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    # print(f"INFO [conftest.py]: Prepending project root to sys.path: {PROJECT_ROOT}")
    sys.path.insert(0, PROJECT_ROOT)
# else:
#     print(f"INFO [conftest.py]: Project root already in sys.path: {PROJECT_ROOT}")
# --- End of Python Path Management ---

# Now import project modules
try:
    from graph_state import AgentState
    from app_graph import app as langgraph_application
    from utils import load_agent_registry # For langgraph_app fixture
    # print("INFO [conftest.py]: Successfully imported project-specific modules.")
except ImportError as e:
    print(f"CRITICAL ERROR [conftest.py]: Still failed to import project modules. Check PYTHONPATH. Error: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root calculated as: {PROJECT_ROOT}")
    if 'graph_state' not in sys.modules:
        print(f"graph_state.py exists at {os.path.join(PROJECT_ROOT, 'graph_state.py')}? {os.path.exists(os.path.join(PROJECT_ROOT, 'graph_state.py'))}")
    raise

@pytest.fixture(scope="function", autouse=True)
def set_dummy_openai_api_key_for_tests(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_pk_for_testing_conftest")

@pytest.fixture
def mock_initial_state():
    return AgentState(
        original_query="Test query", intent=None, entities=None,
        sql_query_generated=None, sql_query_result=None, retrieved_contexts=None,
        rag_summary=None, intermediate_response=None, final_answer=None,
        error_message=None, history=[], processing_steps_versions={},
        node_latencies={}, node_execution_order=[]
    )

def _load_golden_json_data(file_path_relative_to_project_root):
    full_path = os.path.join(PROJECT_ROOT, file_path_relative_to_project_root)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        pytest.fail(f"Golden query JSON file not found: {full_path}.", pytrace=False)
    except json.JSONDecodeError as e:
        pytest.fail(f"Error decoding JSON from file: {full_path}. Error: {e}", pytrace=False)
    return []


@pytest.fixture(scope="session")
def golden_sql_queries_json():
    return _load_golden_json_data("data/golden_queries_sql.json")

@pytest.fixture(scope="session")
def golden_retrieval_queries_json():
    return _load_golden_json_data("data/golden_queries_retrieval.json")

@pytest.fixture
def langgraph_app():
    load_agent_registry(force_reload=True)
    return langgraph_application

@pytest.fixture
def mock_openai_client(mocker):
    mock_client = mocker.MagicMock(name="mock_openai_client_instance")
    mock_chat_message = mocker.MagicMock(name="mock_chat_message")
    # This default content will be used if utils.get_llm_response is called WITHOUT
    # being specifically patched by a test's side_effect.
    mock_chat_message.content = "Default Content from mock_openai_client in conftest.py"
    mock_chat_response = mocker.MagicMock(name="mock_chat_response")
    mock_chat_response.choices = [mocker.MagicMock(message=mock_chat_message, name="mock_chat_choice")]
    mock_client.chat.completions.create.return_value = mock_chat_response
    mock_embedding_data_item = mocker.MagicMock(name="mock_embedding_data_item")
    mock_embedding_data_item.embedding = [0.01] * 1536
    mock_embedding_response = mocker.MagicMock(name="mock_embedding_response")
    mock_embedding_response.data = [mock_embedding_data_item]
    mock_client.embeddings.create.return_value = mock_embedding_response
    return mock_client

@pytest.fixture(scope="function", autouse=True)
def patch_utils_openai_client_globally(monkeypatch, mock_openai_client):
    # This ensures utils.client is always our mock_openai_client
    monkeypatch.setattr("utils.client", mock_openai_client)
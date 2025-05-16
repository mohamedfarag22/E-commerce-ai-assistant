# E-commerce AI Assistant (LangGraph Version)

[![Python CI Pipeline](https://github.com/<YOUR_USERNAME>/<YOUR_REPONAME>/actions/workflows/ci.yml/badge.svg)](https://github.com/<YOUR_USERNAME>/<YOUR_REPONAME>/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/<YOUR_USERNAME>/<YOUR_REPONAME>/graph/badge.svg?token=<YOUR_CODECOV_BADGE_TOKEN_IF_PRIVATE_OR_NEEDED>)](https://codecov.io/gh/<YOUR_USERNAME>/<YOUR_REPONAME>)

A modular, version-controlled AI assistant designed to support customer service operations for an e-commerce platform. This system leverages LangGraph to orchestrate Large Language Model (LLM)-based agents, SQL querying capabilities, and semantic document retrieval (RAG).

**Current Test Status: 16 unit tests passed!** (Adjust this number based on your actual passing unit tests for individual agent nodes). The pipeline integration tests (5 for SQL, 5 for Retrieval) are also passing with mocked dependencies.

## Core Features

*   **Modular Agent Architecture:** Built with LangGraph, allowing for clear separation of concerns (intent parsing, SQL processing, RAG, meta-queries, response synthesis).
*   **Version Control for Agents:** Agent configurations (prompts, models, versions) are managed via `agent_registry.yaml`.
*   **Intent Recognition:** Understands user queries to determine primary intent and extracts relevant entities.
*   **SQL Querying:** Answers structured data questions by generating and executing SQLite queries against an e-commerce database.
*   **Document Retrieval (RAG):** Provides answers from unstructured documents (FAQs, policies) using FAISS for semantic search and LLM-based summarization.
*   **Meta-Awareness:** Can respond to queries about its own component versions.
*   **Benchmarking & Evaluation:** Includes scripts and golden data for evaluating performance and accuracy.
*   **Comprehensive Testing:** Unit tests for individual agent nodes (including JSON schema validation for outputs) and integration tests for key pipelines.
*   **CI/CD Integration:** GitHub Actions workflow for automated linting, testing, and code coverage reporting.

## Project Structure
ecommerce_ai_assistant/
├── .env # Environment variables (e.g., OPENAI_API_KEY)
├── README.md # This file
├── DESIGN_JUSTIFICATION.md # Architectural design choices
├── requirements.txt # Application dependencies
├── agent_registry.yaml # Agent configurations
├── eval_config.yaml # Configuration for run_evaluation.py
├── data/
│   ├── ecommerce_support.db
│   ├── documents/
│   │   ├── return_policy.txt
│   │   └── shipping_faq.txt
│   ├── doc_index/
│   ├── golden_queries_sql.csv       # For SQL agent/pipeline testing
│   └── golden_queries_retrieval.csv # For Retrieval agent/pipeline testing
├── app.py # Flask API entry point
├── main.py # CLI entry point
├── utils.py # Shared utilities
├── graph_state.py # AgentState definition
├── app_graph.py # LangGraph workflow definition
├── build_document_index.py # Script for FAISS index creation
├── run_evaluation.py # Script for benchmarking
├── agents/                     # Your agent node implementations (MUST be instrumented for latency)
│   ├── __init__.py
│   ├── intent_parser_node.py
│   ├── retrieval_node.py
│   ├── sql_node.py
│   ├── meta_query_node.py
│   └── response_node.py├── data/ # Database, documents, FAISS index, golden JSON queries
├── prompts/    # (Will need updates for gpt-4o and new nodes)
│   ├── intent/
│   │   └── v1_0_parser.txt
│   ├── meta/
│   │   └── v1_0_responder.txt
│   ├── retrieval/
│   │   └── v1_0_rag.txt
│   ├── response/
│   │   └── v1_0_format.txt
│   └── sql/
│       └── v1_0_schema.txt
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Shared fixtures
│   ├── agents/                      # Unit tests for individual agent nodes
│   │   ├── __init__.py
│   │   ├── test_retrieval_node.py
│   │   ├── test_meta_query_node.py
│   │   └── test_response_node.py
│   ├── integration/                 # Tests for the whole graph/pipeline
│   │   ├── __init__.py
│   │   ├── test_sql_pipeline.py     # Uses golden_queries_sql.csv
│   │   └── test_retrieval_pipeline.py # Uses golden_queries_retrieval.csv
└── .github/workflows/ci.yml # GitHub Actions CI workflow


## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPONAME>.git
    cd ecommerce_ai_assistant
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```
4.  **Set up environment variables:**
    Create a `.env` file in the project root:
    ```env
    OPENAI_API_KEY="your_actual_openai_api_key_here_for_running_app_and_eval"
    ```
    (For `pytest` runs, a dummy key is used as LLM calls are mocked).

5.  **Build Document Index (for RAG):**
    Required before first run or after document updates if not using a pre-built index.
    ```bash
    python build_document_index.py
    ```
    *(Note: For `pytest` integration tests, FAISS interactions are mocked and do not require a real index during the test run itself).*

6.  **Prepare Database:**
    Ensure `data/ecommerce_support.db` exists and is populated with the necessary schema and sample data. The SQL pipeline tests will execute queries against this database.

## Running the Application

### CLI
```bash
python main.py

### ```Flask API
python app.py

The API will be available at http://127.0.0.1:5000. Send POST requests to /chat:
{"message": "What is the status of my order #1002?"}

Running Tests
Ensure you are in the project root directory and your virtual environment is activated.
Run all tests (Unit & Mocked Integration):
pytest
Use code with caution.
Bash
For verbose output:
pytest -v
Use code with caution.
Bash
To see print statements from tests (useful for debugging):
pytest -s
Use code with caution.
Bash
Run specific test files:
pytest tests/agents/test_intent_parser_node.py
pytest tests/integration/test_sql_pipeline.py
Use code with caution.
Bash
Test Coverage:
The CI pipeline is configured to generate a code coverage report using pytest-cov. You can also run this locally:
pytest --cov=. --cov-report=html
# Then open htmlcov/index.html in your browser
Use code with caution.
Bash
Running Evaluations (Benchmarking)
To perform a more end-to-end evaluation using actual LLM calls (ensure OPENAI_API_KEY is set in .env):
python run_evaluation.py
Use code with caution.
Bash
This script uses eval_config.yaml and your golden JSON query files (data/golden_*.json) to generate an evaluation_report.md with metrics like accuracy and latency.
Key Technologies
Python 3.9+
LangGraph: For orchestrating agents.
OpenAI API: For LLM capabilities.
FAISS: For vector similarity search (RAG).
SQLite: For the e-commerce database.
Flask: For the web API.
Pytest: For automated testing.
Jsonschema: For validating JSON structures in tests.
GitHub Actions: For CI/CD.
Codecov (Optional): For code coverage tracking.
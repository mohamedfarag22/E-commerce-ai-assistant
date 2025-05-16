# E-commerce AI Assistant (LangGraph Version)

[![Python CI Pipeline](https://github.com/mohamedfarag22/ecommerce-ai-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/mohamedfarag22/ecommerce-ai-assistant/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mohamedfarag22/<YOUR_REPONAME>/graph/badge.svg?token=<YOUR_CODECOV_BADGE_TOKEN_IF_PRIVATE_OR_NEEDED>)](https://codecov.io/gh/mohamedfarag22/ecommerce-ai-assistant)

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

```
ecommerce_ai_assistant/
├── .env
├── README.md
├── DESIGN_JUSTIFICATION.md
├── requirements.txt
├── agent_registry.yaml
├── eval_config.yaml
├── data/
│   ├── ecommerce_support.db
│   ├── documents/
│   ├── doc_index/
│   ├── golden_queries_sql.csv
│   └── golden_queries_retrieval.csv
├── app.py
├── main.py
├── utils.py
├── graph_state.py
├── app_graph.py
├── build_document_index.py
├── run_evaluation.py
├── agents/
│   ├── intent_parser_node.py
│   ├── retrieval_node.py
│   ├── sql_node.py
│   ├── meta_query_node.py
│   └── response_node.py
├── prompts/
├── tests/
│   ├── agents/
│   ├── integration/
├── .github/workflows/ci.yml
```

## Flow graph Agents

![Flow Graph Agents](https://github.com/mohamedfarag22/ecommerce-ai-assistant/raw/main/Graph_flow_Agents.png)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mohamedfarag22/<YOUR_REPONAME>.git
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

**CLI**:
```bash
python main.py
```

**Flask API with Front-End**:
```bash
python app.py
# Access at http://127.0.0.1:5000
# POST request body: {"message": "Where is my order #123?"}
```

## 🧪 Running Tests

```bash
pytest             # All tests
pytest -v          # Verbose output
pytest -s          # Show print statements
pytest tests/agents/test_intent_parser_node.py
pytest tests/integration/test_sql_pipeline.py
```

**Test Coverage**:

```bash
pytest --cov=. --cov-report=html
# Open htmlcov/index.html in your browser
```

## 📊 Evaluation

Run full evaluation (uses OpenAI API, ensure `.env` is set):

```bash
python run_evaluation.py

```
This script uses eval_config.yaml and your golden JSON query files (data/golden_*.json) to generate an evaluation_report.md with metrics like accuracy and latency.

# Result of Evalution to pass the tests units :
![Evalation Report](https://github.com/mohamedfarag22/ecommerce-ai-assistant/raw/main/Evaluation_result_pyTest.png)

# LangGraph Agent Benchmarking Tool
```
pytho
```
This tool evaluates the performance of a LangGraph agent by running it against a benchmark dataset of queries and comparing the actual responses with expected results using GPT-4o for semantic evaluation.

## Features

- Automated benchmarking of LangGraph agent responses
- Semantic comparison of expected vs. actual results using GPT-4o
- JSON output of benchmark results
- Handles both structured and natural language responses

## Dependencies

- Python 3.12
- Required packages:
  - `langchain`
  - `openai`
  - Your custom LangGraph app (`app_graph`)
  - Your custom state definition (`graph_state`)

## 🛠️ Tech Stack

- Python 3.9+
- langchain
- LangGraph
- OpenAI API (GPT)
- FAISS
- SQLite
- Flask
- Pytest + Jsonschema
- GitHub Actions + Codecov

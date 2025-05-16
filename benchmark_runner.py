from langchain.embeddings import OpenAIEmbeddings
from app_graph import app  # your LangGraph compiled app
from graph_state import AgentState
from openai import OpenAI
import json

# Your benchmark dataset
benchmark_data = [
    {
        "query": "Where is my order #1002?",
        "expected_result": '[{"status": "Shipped"}]'
    },
    {
        "query": "What is the email of the customer who placed order #1003?",
        "expected_result": '[{"email": "john@example.com"}]'
    },
    {
        "query": "Do you have Nike Air Max in stock?",
        "expected_result": '[{"inventory_count": 12}]'
    },
    {
        "query": "Which orders were returned and why?",
        "expected_result": '[{"id": 1005, "reason": "Item damaged"},{"id": 1007, "reason": "Wrong size"},{"id": 1011, "reason": "Wrong item sent"},{"id": 12345, "reason": "Damaged item"}]'
    },
    {
        "query": "What’s the most recent order placed by Alice Smith?",
        "expected_result": '[{"id": 1010, "order_date": "2024-12-01"}]'
    }
]

# Initialize OpenAI Chat client
client = OpenAI()

def evaluate_with_gpt(expected, actual):
    """Use GPT to determine if the actual result matches the expected result even if phrased differently."""
    prompt = (
        "Your task is to compare two pieces of text:\n\n"
        f"Expected (JSON format or structured data):\n{expected}\n\n"
        f"Actual (natural language or structured):\n{actual}\n\n"
        "Determine whether the actual response conveys the **same factual information** as the expected result, "
        "even if the wording or format is different. Ignore formatting, extra explanations, or polite phrases.\n\n"
        "ONLY return 'True' if the actual response accurately contains all the core facts from the expected result or partial answer.\n"
        "If have big missing or incorrect, return 'False'.\n\n"
        "Respond ONLY with: True or False."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise evaluator who checks if two responses have equivalent meaning."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip() == "True"


def run_benchmark():
    results = []

    for sample in benchmark_data:
        query = sample["query"]
        expected = sample["expected_result"]

        print(f"\n🔍 Running query: {query}")

        # Initialize state
        state: AgentState = {
            "original_query": query,
            "intent": None,
            "entities": None,
            "sql_query_generated": None,
            "sql_query_result": None,
            "retrieved_contexts": None,
            "rag_summary": None,
            "intermediate_response": None,
            "final_answer": None,
            "error_message": None,
            "history": [],
            "processing_steps_versions": {},
            "node_latencies": {},
            "node_execution_order": []
        }

        # Run through LangGraph
        final_state = app.invoke(state)
        actual_response = final_state.get("final_answer", "")

        # Evaluate with GPT
        gpt_equivalent = evaluate_with_gpt(expected, actual_response)

        results.append({
            "query": query,
            "expected_result": expected,
            "actual_response": actual_response,
            "match": gpt_equivalent
        })

    return results

# Run benchmark and save to file
if __name__ == "__main__":
    benchmark_results = run_benchmark()

    # Save to JSON file
    with open("benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\n📁 Results saved to benchmark_results.json")

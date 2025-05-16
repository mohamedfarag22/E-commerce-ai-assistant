from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from app_graph import app as langgraph_app
from graph_state import AgentState
from utils import logger, load_agent_registry
import os
import time # Import time module

app = Flask(__name__)
load_dotenv()

# Check API key at startup
if not os.getenv("OPENAI_API_KEY"):
    logger.critical("OPENAI_API_KEY not set. Exiting.")
    raise RuntimeError("OPENAI_API_KEY is not set.")

# Load registry once
load_agent_registry()

@app.route("/")
def index():
    return render_template("index.html")  # Optional HTML interface

@app.route("/chat", methods=["POST"])
def ask():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    # Ensure node_latencies and node_execution_order are part of initial_state if AgentState includes them
    # This is good practice, though they are primarily populated by the nodes themselves.
    initial_state: AgentState = {
        "original_query": user_input,
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
        "node_latencies": {},  # Initialize for benchmarking data
        "node_execution_order": [] # Initialize for benchmarking data
    }

    try:
        start_time = time.perf_counter() # Start timer for total processing
        final_state = langgraph_app.invoke(initial_state)
        end_time = time.perf_counter() # End timer for total processing
        processing_time = end_time - start_time
        
        logger.info(f"Total LangGraph processing time for query '{user_input[:50]}...': {processing_time:.4f} seconds")

        # Log per-node latencies and execution order if available
        node_latencies = final_state.get("node_latencies")
        node_execution_order = final_state.get("node_execution_order")

        if node_latencies:
            logger.info(f"Per-node latencies: {node_latencies}")
        if node_execution_order:
            logger.info(f"Node execution order: {' -> '.join(node_execution_order)}")


        response_data = {}
        if final_state.get("final_answer"):
            response_data = {"response": final_state["final_answer"]}
        elif final_state.get("error_message"):
            response_data = {"response": f"I encountered an error: {final_state['error_message']}"}
        else:
            response_data = {"response": "I'm not sure how to respond to that."}
        
        # Optionally add processing time and benchmark data to response for debugging
        # response_data["debug_total_processing_time_seconds"] = round(processing_time, 4)
        # if node_latencies:
        #     response_data["debug_node_latencies"] = node_latencies
        # if node_execution_order:
        #     response_data["debug_node_execution_order"] = node_execution_order
            
        return jsonify(response_data)

    except Exception as e:
        logger.error("Error during processing", exc_info=True)
        return jsonify({"response": "A critical error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True)
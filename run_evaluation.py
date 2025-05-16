import yaml
import json
import time
import os
from pathlib import Path
from difflib import SequenceMatcher
import sqlite3

from dotenv import load_dotenv
load_dotenv() # Load .env for OPENAI_API_KEY, critical for this script

from app_graph import app as langgraph_app
from graph_state import AgentState
from utils import logger, load_agent_registry

# --- Configuration ---
DEFAULT_EVAL_CONFIG_PATH = "eval_config.yaml"
DEFAULT_DB_PATH = "data/ecommerce_support.db"

def load_eval_config(config_path=DEFAULT_EVAL_CONFIG_PATH):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_golden_queries_from_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        if not isinstance(queries, list):
            logger.error(f"Golden query file {json_path} should contain a JSON list.")
            return []
        return queries
    except FileNotFoundError:
        logger.error(f"Golden query file not found: {json_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {json_path}")
        return []

def compare_sql_queries(generated_sql, expected_sql):
    if generated_sql is None and expected_sql is None: return True, 1.0
    if generated_sql is None or expected_sql is None: return False, 0.0
    norm_gen = generated_sql.lower().strip().rstrip(';')
    norm_exp = expected_sql.lower().strip().rstrip(';')
    return norm_gen == norm_exp, SequenceMatcher(None, norm_gen, norm_exp).ratio()

def _normalize_value_for_comparison(value):
    """Helper to normalize string values to lowercase for comparison."""
    if isinstance(value, str):
        return value.lower()
    return value

def _normalize_dict_for_comparison(d):
    """Normalizes string values in a dictionary to lowercase."""
    if not isinstance(d, dict):
        return d
    return {k: _normalize_value_for_comparison(v) for k, v in d.items()}

def compare_sql_results(generated_results, expected_results_str):
    # Normalize None/empty scenarios
    gen_is_empty = generated_results is None or (isinstance(generated_results, list) and not generated_results)
    exp_is_empty = not expected_results_str or expected_results_str.lower() in ['null', 'none', '[]']

    if gen_is_empty and exp_is_empty:
        return True, 1.0
    if gen_is_empty or exp_is_empty: # One is empty, the other is not
        return False, 0.0

    try:
        expected_results = json.loads(expected_results_str)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse expected_results_str as JSON: {expected_results_str}")
        return False, 0.0

    # Normalize generated results (especially for case-insensitivity of strings)
    if isinstance(generated_results, list):
        normalized_generated_results = [_normalize_dict_for_comparison(item) for item in generated_results]
    else: # Should ideally be a list, but handle other types by direct comparison after normalization
        normalized_generated_results = _normalize_value_for_comparison(generated_results)


    # Normalize expected results if they are lists of dicts
    if isinstance(expected_results, list):
        normalized_expected_results = [_normalize_dict_for_comparison(item) for item in expected_results]
    else:
        normalized_expected_results = _normalize_value_for_comparison(expected_results)


    if not isinstance(normalized_generated_results, list) or not isinstance(normalized_expected_results, list):
        # Fallback for non-list comparison (e.g., single value expected/generated)
        match = normalized_generated_results == normalized_expected_results
        return match, 1.0 if match else 0.0

    # If both are lists, compare length first
    if len(normalized_generated_results) != len(normalized_expected_results):
        logger.info(f"SQL Result Length Mismatch: Expected {len(normalized_expected_results)}, Got {len(normalized_generated_results)}")
        return False, 0.0

    # For list of dicts, sort them to handle order-agnostic comparison
    try:
        # Sort based on a canonical string representation of each (normalized) dict
        sorted_gen_str = sorted([json.dumps(item, sort_keys=True) for item in normalized_generated_results])
        sorted_exp_str = sorted([json.dumps(item, sort_keys=True) for item in normalized_expected_results])
        
        match = sorted_gen_str == sorted_exp_str
        if not match:
            logger.info(f"SQL Result Content Mismatch (after normalization & sort):")
            logger.info(f"Expected (sorted str): {sorted_exp_str}")
            logger.info(f"Got (sorted str): {sorted_gen_str}")

        return match, 1.0 if match else 0.0
    except TypeError:
        logger.warning("Could not sort SQL results by JSON string for comparison (e.g. unhashable types); performing direct list comparison of normalized results.")
        match = normalized_generated_results == normalized_expected_results # Order matters here
        return match, 1.0 if match else 0.0


def compare_retrieval_sources(retrieved_contexts, expected_source_filename):
    # ... (this function can remain the same as it already does .lower()) ...
    if not retrieved_contexts and not expected_source_filename: return True, 1.0, "N/A (None expected)"
    if not retrieved_contexts: return False, 0.0, "No contexts retrieved"
    if not expected_source_filename: return False, 0.0, "Expected source was empty"

    top_retrieved_source = retrieved_contexts[0].get("source")
    if top_retrieved_source:
        match = top_retrieved_source.strip().lower() == expected_source_filename.strip().lower()
        return match, 1.0 if match else 0.0, top_retrieved_source
    return False, 0.0, "N/A (Source missing in top retrieved context)"


def run_single_query_evaluation(user_query: str) -> AgentState:
    # ... (this function remains the same) ...
    initial_state: AgentState = {
        "original_query": user_query,
        "intent": None, "entities": None, "sql_query_generated": None,
        "sql_query_result": None, "retrieved_contexts": None, "rag_summary": None,
        "intermediate_response": None, "final_answer": None, "error_message": None,
        "history": [], "processing_steps_versions": {},
        "node_latencies": {}, "node_execution_order": []
    }
    total_latency_start = time.perf_counter()
    final_state = langgraph_app.invoke(initial_state)
    total_latency_end = time.perf_counter()
    final_state["_total_latency_"] = round(total_latency_end - total_latency_start, 4)
    return final_state


def generate_report_markdown(eval_name, description, results, output_path):
    # ... (this function remains the same) ...
    report_content = [f"# Evaluation Report: {eval_name}"]
    if description:
      report_content.append(f"_{description}_")
    report_content.append(f"\nDate: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if not results:
        report_content.append("\n\nNo results to report.")
        with open(output_path, 'w', encoding='utf-8') as f: f.write("\n".join(report_content))
        logger.info(f"Empty evaluation report saved to {output_path}")
        return
    report_content.append("\n## Summary Metrics")
    num_queries = len(results)
    avg_total_latency = sum(r['total_latency'] for r in results if 'total_latency' in r and isinstance(r['total_latency'], (int, float))) / num_queries if num_queries else 0
    sql_queries_results = [r for r in results if r['type'] == 'SQL']
    sql_queries_count = len(sql_queries_results)
    sql_accuracy_count = sum(1 for r in sql_queries_results if r.get('sql_query_correct'))
    sql_result_accuracy_count = sum(1 for r in sql_queries_results if r.get('sql_result_correct'))
    retrieval_queries_results = [r for r in results if r['type'] == 'Retrieval']
    retrieval_queries_count = len(retrieval_queries_results)
    retrieval_accuracy_count = sum(1 for r in retrieval_queries_results if r.get('retrieval_source_correct'))
    report_content.append(f"- **Total Queries Processed:** {num_queries}")
    report_content.append(f"- **Average Total Latency:** {avg_total_latency:.4f}s")
    if sql_queries_count > 0:
        report_content.append(f"- **SQL Query Accuracy:** {sql_accuracy_count}/{sql_queries_count} ({ (sql_accuracy_count/sql_queries_count)*100 if sql_queries_count else 0 :.2f}%)")
        report_content.append(f"- **SQL Result Accuracy:** {sql_result_accuracy_count}/{sql_queries_count} ({ (sql_result_accuracy_count/sql_queries_count)*100 if sql_queries_count else 0 :.2f}%)")
    if retrieval_queries_count > 0:
        report_content.append(f"- **Retrieval Source Accuracy (Top 1):** {retrieval_accuracy_count}/{retrieval_queries_count} ({ (retrieval_accuracy_count/retrieval_queries_count)*100 if retrieval_queries_count else 0 :.2f}%)")
    report_content.append("\n## Detailed Results\n")
    report_content.append("| Query (First 50 chars) | Type | Total Latency (s) | SQL Query Correct | SQL Result Correct | Retrieval Source Correct | Final Answer (Preview) | Node Latencies | Execution Order | Agent Versions |")
    report_content.append("|---|---|---|---|---|---|---|---|---|---|")
    for res in results:
        query_preview = (res.get('original_query', 'N/A')[:50] + '...') if res.get('original_query') else 'N/A'
        total_lat = f"{res.get('total_latency', 0.0):.4f}"
        node_lats_str = json.dumps(res.get('node_latencies', {}))
        exec_order_str = " -> ".join(res.get('node_execution_order', []))
        agent_vers_str = json.dumps(res.get('processing_steps_versions', {}))
        final_ans_preview = (str(res.get('final_answer') or res.get('error_message', 'N/A') or "None")).replace("\n", "<br>")[:100] + "..."
        row = f"| {query_preview} | {res.get('type', 'N/A')} | {total_lat} " \
              f"| {res.get('sql_query_correct', 'N/A')} | {res.get('sql_result_correct', 'N/A')} " \
              f"| {res.get('retrieval_source_correct', 'N/A')} | {final_ans_preview} " \
              f"| `{node_lats_str}` | `{exec_order_str}` | `{agent_vers_str}` |"
        report_content.append(row)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_content))
    logger.info(f"Evaluation report saved to {output_path}")


def main(eval_config_file=DEFAULT_EVAL_CONFIG_PATH):
    # ... (this function remains largely the same, ensure it calls the updated compare_sql_results) ...
    logger.info(f"Starting evaluation process using config: {eval_config_file}...")
    if not os.getenv("OPENAI_API_KEY"):
        logger.critical("OPENAI_API_KEY not set. Evaluation requires API access. Exiting.")
        return

    load_agent_registry(force_reload=True) 
    eval_config = load_eval_config(eval_config_file)
    if not eval_config:
        logger.error("Failed to load evaluation configuration. Exiting.")
        return
    
    all_run_results = []
    for eval_run_config in eval_config.get("evaluations", []):
        eval_name = eval_run_config.get("name", "Unnamed Evaluation")
        eval_description = eval_run_config.get("description", "")
        logger.info(f"\n--- Running Evaluation Set: {eval_name} ---")
        if eval_description: logger.info(f"Description: {eval_description}")
        current_set_results = []
        sql_queries_path = eval_run_config.get("sql_golden_queries_path")
        if sql_queries_path:
            logger.info(f"Loading SQL golden queries from JSON: {sql_queries_path}")
            sql_golden_queries = load_golden_queries_from_json(sql_queries_path)
            for i, gq in enumerate(sql_golden_queries):
                query_text = gq.get("query", f"SQL Query {i+1} (missing text)")
                logger.info(f"Processing SQL query ({i+1}/{len(sql_golden_queries)}): {query_text[:70]}...")
                if not all(k in gq for k in ["query", "expected_sql", "expected_result"]):
                    logger.warning(f"Skipping malformed SQL golden query: {gq}")
                    continue
                final_state = run_single_query_evaluation(gq["query"])
                sql_q_correct, sql_q_sim = compare_sql_queries(final_state.get("sql_query_generated", ""), gq["expected_sql"])
                # Pass the raw generated_results here
                sql_r_correct, sql_r_sim = compare_sql_results(final_state.get("sql_query_result"), gq["expected_result"])
                current_set_results.append({
                    "type": "SQL", "original_query": gq['query'],
                    "expected_sql": gq["expected_sql"], "generated_sql": final_state.get("sql_query_generated"),
                    "sql_query_correct": sql_q_correct, "sql_query_similarity": round(sql_q_sim, 4),
                    "expected_result_str": gq["expected_result"], "generated_result": final_state.get("sql_query_result"), # Log raw generated
                    "sql_result_correct": sql_r_correct, "sql_result_similarity": round(sql_r_sim, 4),
                    "final_answer": final_state.get("final_answer"), "error_message": final_state.get("error_message"),
                    "total_latency": final_state.get("_total_latency_"), "node_latencies": final_state.get("node_latencies"),
                    "node_execution_order": final_state.get("node_execution_order"),
                    "processing_steps_versions": final_state.get("processing_steps_versions")
                })
        retrieval_queries_path = eval_run_config.get("retrieval_golden_queries_path")
        if retrieval_queries_path:
            logger.info(f"Loading Retrieval golden queries from JSON: {retrieval_queries_path}")
            retrieval_golden_queries = load_golden_queries_from_json(retrieval_queries_path)
            for i, gq in enumerate(retrieval_golden_queries):
                query_text = gq.get("query", f"Retrieval Query {i+1} (missing text)")
                logger.info(f"Processing Retrieval query ({i+1}/{len(retrieval_golden_queries)}): {query_text[:70]}...")
                if not all(k in gq for k in ["query", "expected_answer_source"]):
                    logger.warning(f"Skipping malformed Retrieval golden query: {gq}")
                    continue
                final_state = run_single_query_evaluation(gq["query"])
                ret_s_correct, ret_s_sim, actual_src = compare_retrieval_sources(final_state.get("retrieved_contexts"), gq["expected_answer_source"])
                current_set_results.append({
                    "type": "Retrieval", "original_query": gq['query'],
                    "expected_source": gq["expected_answer_source"], "retrieved_contexts": final_state.get("retrieved_contexts"),
                    "actual_top_source": actual_src, "retrieval_source_correct": ret_s_correct,
                    "retrieval_source_similarity": round(ret_s_sim, 4),
                    "rag_summary": final_state.get("rag_summary"), "final_answer": final_state.get("final_answer"),
                    "error_message": final_state.get("error_message"),
                    "total_latency": final_state.get("_total_latency_"), "node_latencies": final_state.get("node_latencies"),
                    "node_execution_order": final_state.get("node_execution_order"),
                    "processing_steps_versions": final_state.get("processing_steps_versions")
                })
        all_run_results.extend(current_set_results)

    output_report_path = eval_config.get("output_report_path", "evaluation_report.md")
    if all_run_results:
        report_title = eval_config.get("evaluations", [{}])[0].get("name", "System Evaluation")
        report_desc = eval_config.get("evaluations", [{}])[0].get("description", "Combined results")
        generate_report_markdown(report_title, report_desc, all_run_results, output_report_path)
    else:
        logger.warning("No evaluation results were generated to report.")
    logger.info("Evaluation process finished.")

if __name__ == "__main__":
    main()
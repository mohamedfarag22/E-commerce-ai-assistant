# agents/retrieval_node.py
import faiss
import json
import numpy as np
from utils import get_embeddings, get_llm_response, load_prompt_from_path, get_node_config, logger
from graph_state import AgentState
import time
NODE_NAME = "retrieval_processor"
_faiss_index = None
_metadata = None

def _load_retrieval_assets(config):
    global _faiss_index, _metadata
    if _faiss_index is None or _metadata is None: # Basic caching
        try:
            logger.info(f"{NODE_NAME}: Loading FAISS index from {config['vector_store_path']}")
            _faiss_index = faiss.read_index(config['vector_store_path'])
            logger.info(f"{NODE_NAME}: Loading metadata from {config['metadata_store_path']}")
            with open(config['metadata_store_path'], 'r') as f:
                _metadata = json.load(f)
            logger.info(f"{NODE_NAME}: Retrieval assets loaded successfully.")
        except Exception as e:
            logger.error(f"{NODE_NAME}: Failed to load retrieval assets: {e}")
            _faiss_index = None # Reset on error
            _metadata = None
            return False
    return True


def retrieval_node(state: AgentState) -> dict:
    """
    Performs semantic search for relevant documents and synthesizes an answer using RAG.
    Updates state with retrieved_contexts, rag_summary.
    """
    node_start_time = time.perf_counter() # Start timer for this node

    # Initialize benchmarking fields in state if not present
    current_latencies = state.get("node_latencies", {})
    current_order = state.get("node_execution_order", [])
    if NODE_NAME not in current_order:
      current_order.append(NODE_NAME)

    logger.info(f"--- NODE: {NODE_NAME} ---")
    config = get_node_config(NODE_NAME)
    if not config:
        error_result = {"error_message": f"Configuration for node '{NODE_NAME}' not found."}
        node_end_time = time.perf_counter()
        current_latencies[NODE_NAME] = round(node_end_time - node_start_time, 4)
        error_result["node_latencies"] = current_latencies
        error_result["node_execution_order"] = current_order
        return error_result

    if not _load_retrieval_assets(config):
         return {"error_message": "Failed to load retrieval assets (FAISS index or metadata)."}
    
    if _faiss_index is None or _metadata is None: # Double check after load attempt
        return {"error_message": "Retrieval assets are not available."}

    user_query = state["original_query"]
    embedding_model = config["embedding_model"]
    top_k = config.get("top_k", 3)
    # similarity_threshold = config.get("similarity_threshold", 0.5) # FAISS L2 search returns distances

    query_embedding_list = get_embeddings([user_query], model=embedding_model)
    if not query_embedding_list or not query_embedding_list[0]:
        logger.error(f"{NODE_NAME}: Failed to generate embedding for query: {user_query}")
        return {"error_message": "Failed to generate query embedding."}
    
    query_embedding = np.array(query_embedding_list[0]).astype('float32').reshape(1, -1)

    # FAISS search: D = distances, I = indices
    distances, indices = _faiss_index.search(query_embedding, top_k)
    
    retrieved_contexts = []
    if indices.size > 0:
        for i in range(indices.shape[1]): # Iterate through top_k results
            doc_index = indices[0][i]
            if 0 <= doc_index < len(_metadata):
                # L2 distance is lower for more similar items. 
                # We might want to convert this to a similarity score or filter by a max distance.
                # For now, just retrieve top_k.
                # logger.info(f"Retrieved doc index: {doc_index}, distance: {distances[0][i]}")
                context = _metadata[doc_index]
                retrieved_contexts.append({
                    "source": context.get("source"),
                    "text": context.get("text"),
                    "distance": float(distances[0][i]) # Add distance for potential filtering
                })
            else:
                logger.warning(f"{NODE_NAME}: Retrieved invalid document index {doc_index}.")
    
    logger.info(f"{NODE_NAME}: Retrieved {len(retrieved_contexts)} contexts.")
    logger.info(f" Retrieved Context {retrieved_contexts}")


    if not retrieved_contexts:
        return {
            "retrieved_contexts": [], 
            "rag_summary": "I couldn't find specific information about that in my knowledge base.",
            "processing_steps_versions": {**state.get("processing_steps_versions", {}), NODE_NAME: config.get("version")}
        }

    # RAG: Synthesize answer from contexts
    rag_prompt_template = load_prompt_from_path(config["rag_prompt_path"])
    # Example RAG prompt (prompts/retrieval/v1_0_rag.txt):
    """
    You are an AI assistant. Use the following retrieved context to answer the user's question.
    If the context doesn't directly answer the question, say you couldn't find the information.
    Be concise and helpful.

    Context:
    {context_str}

    User Question: {user_query}
    Answer:
    """
    context_str = "\n\n---\n\n".join([ctx["text"] for ctx in retrieved_contexts])
    formatted_rag_prompt = rag_prompt_template.format(context_str=context_str, user_query=user_query)
    logger.info(f"RAG Prompt: {formatted_rag_prompt}")
    content = get_llm_response(
        prompt=formatted_rag_prompt,
        model=config.get("llm_model_for_rag")
    )
    partial_result = {}
    # Update processing steps versions
    node_end_time = time.perf_counter()
    current_latencies[NODE_NAME] = round(node_end_time - node_start_time, 4)
    partial_result["node_latencies"] = current_latencies
    partial_result["node_execution_order"] = current_order
    new_State = {"intermediate_response": content.strip() if content else "",
        "rag_summary": content.strip() if content else "",
        "error_message": None,
        "processing_steps_versions": {NODE_NAME: config.get("version")},
        "retrieved_contexts": retrieved_contexts,**partial_result
        }
    logger.info(f"{NODE_NAME}: RAG summary: {state['rag_summary']}")
    logger.info(f"{NODE_NAME}: Intermediate response: {state['intermediate_response']}")

    return new_State
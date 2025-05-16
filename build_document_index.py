# build_document_index.py
import os
import json
import faiss
import numpy as np
from utils import get_embeddings, logger # Use the centralized logger
import tiktoken
from typing import TypedDict, Optional, List, Dict, Any

# Configuration
DOCUMENTS_DIR = os.path.join("data", "documents")
INDEX_DIR = os.path.join("data", "doc_index")

# Updated for text-embedding-3-small
EMBEDDING_MODEL = "text-embedding-3-small" 
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "doc_index_v1_tes.faiss") 
METADATA_PATH = os.path.join(INDEX_DIR, "doc_metadata_v1_tes.json")

MAX_CHUNK_TOKENS = 100 
OVERLAP_TOKENS = 25

try:
    tokenizer = tiktoken.encoding_for_model("gpt-4o") # gpt-4o uses cl100k_base
except Exception:
    logger.warning("Falling back to cl100k_base tokenizer")
    tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_text_by_tokens(text, max_tokens=MAX_CHUNK_TOKENS, overlap_tokens=OVERLAP_TOKENS):
    # ... (same chunking logic as before)
    tokens = tokenizer.encode(text)
    chunks = []
    current_pos = 0
    while current_pos < len(tokens):
        end_pos = min(current_pos + max_tokens, len(tokens))
        chunk_tokens = tokens[current_pos:end_pos]
        # Ensure chunk is not empty before decoding
        if chunk_tokens:
            chunks.append(tokenizer.decode(chunk_tokens))
        
        if end_pos == len(tokens):
            break
        current_pos += (max_tokens - overlap_tokens)
        if current_pos >= end_pos : 
            current_pos = end_pos
    return chunks

def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    all_chunks_text = []
    all_chunks_metadata = [] 
    doc_id_counter = 0

    logger.info(f"Starting document processing from: {DOCUMENTS_DIR}")
    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCUMENTS_DIR, filename)
            logger.info(f"Processing file: {filename}")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = chunk_text_by_tokens(content)
            logger.info(f"Split '{filename}' into {len(chunks)} chunks.")

            for i, chunk_text in enumerate(chunks):
                if not chunk_text.strip(): # Skip empty chunks
                    continue
                all_chunks_text.append(chunk_text)
                all_chunks_metadata.append({
                    "id": doc_id_counter,
                    "source": filename,
                    "chunk_index_in_doc": i,
                    "text": chunk_text 
                })
                doc_id_counter += 1
    
    if not all_chunks_text:
        logger.warning("No text chunks found to process. Exiting.")
        return

    logger.info(f"Total chunks to embed: {len(all_chunks_text)}")
    
    embeddings_list = get_embeddings(all_chunks_text, model=EMBEDDING_MODEL)
    
    valid_embeddings_data = [] # To store tuples of (embedding, metadata_index)
    for i, emb in enumerate(embeddings_list):
        if emb and len(emb) > 0: # Check if embedding is valid
            valid_embeddings_data.append((emb, i))
        else:
            logger.warning(f"Failed to get embedding for chunk {i} from {all_chunks_metadata[i]['source'] if i < len(all_chunks_metadata) else 'unknown source'}. Skipping.")

    if not valid_embeddings_data:
        logger.error("No valid embeddings were generated. Cannot build FAISS index.")
        return

    # Prepare embeddings and filter metadata
    final_embeddings = np.array([data[0] for data in valid_embeddings_data]).astype('float32')
    final_metadata_indices = [data[1] for data in valid_embeddings_data]
    final_metadata = [all_chunks_metadata[i] for i in final_metadata_indices]

    if final_embeddings.shape[0] == 0:
        logger.error("No embeddings to add to FAISS index after processing.")
        return

    dimension = final_embeddings.shape[1]
    logger.info(f"Embedding dimension: {dimension}")

    index = faiss.IndexFlatL2(dimension)
    index.add(final_embeddings)
    
    logger.info(f"FAISS index built with {index.ntotal} vectors.")
    
    faiss.write_index(index, FAISS_INDEX_PATH)
    logger.info(f"FAISS index saved to {FAISS_INDEX_PATH}")
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(final_metadata, f, indent=4)
    logger.info(f"Document metadata saved to {METADATA_PATH}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"): # Check from utils
        print("OPENAI_API_KEY not set in .env file. Aborting index build.")
    else:
        build_index()
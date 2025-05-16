import faiss
import numpy as np
import json
from pprint import pprint

# Configuration
FAISS_INDEX_PATH = "data/doc_index/doc_index_v1_tes.faiss"
METADATA_PATH = "data/doc_index/doc_metadata_v1_tes.json"
DIMENSIONS_TO_SHOW = 10  # Number of embedding dimensions to display

def print_full_database():
    # Load the index
    index = faiss.read_index(FAISS_INDEX_PATH)
    total_vectors = index.ntotal
    
    # Load metadata
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE VECTOR DATABASE CONTENTS ({total_vectors} ENTRIES)")
    print(f"{'='*60}\n")
    
    # Reconstruct all vectors
    all_vectors = index.reconstruct_n(0, total_vectors)
    
    for idx in range(total_vectors):
        # Print entry header
        print(f"\n{'#'*20} ENTRY {idx+1}/{total_vectors} {'#'*20}")
        print(f"FAISS Index Position: {idx}")
        
        # Print full metadata
        print("\nMETADATA:")
        pprint(metadata[idx], width=100, indent=2)
        
        # Print embedding info
        vector = all_vectors[idx]
        print(f"\nEMBEDDING (DIMENSIONS: {len(vector)})")
        print("First 10 dimensions:")
        print(np.array2string(vector[:DIMENSIONS_TO_SHOW], 
                            precision=6, 
                            suppress_small=True,
                            floatmode='fixed'))
        
        # Print vector statistics
        print("\nVECTOR STATISTICS:")
        print(f"  Magnitude (L2 norm): {np.linalg.norm(vector):.6f}")
        print(f"  Min value: {np.min(vector):.6f}")
        print(f"  Max value: {np.max(vector):.6f}")
        print(f"  Mean: {np.mean(vector):.6f}")
        print(f"  Std Dev: {np.std(vector):.6f}")
        
        print(f"\n{'='*60}")

    # Database-wide statistics
    print("\nDATABASE SUMMARY STATISTICS:")
    print(f"Total vectors: {total_vectors}")
    print(f"Embedding dimension: {index.d}")
    print(f"Global min value: {np.min(all_vectors):.6f}")
    print(f"Global max value: {np.max(all_vectors):.6f}")
    print(f"Average vector magnitude: {np.mean([np.linalg.norm(v) for v in all_vectors]):.6f}")

if __name__ == "__main__":
    print_full_database()
"""
Diagnostics for embeddings and retrieval
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from prepare_data import prepare_dataset
from retrieval import Retriever
from config import EMBS_FILE

def check_embeddings():
    # Load embeddings
    embeddings = np.load(EMBS_FILE)
    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    # Norms
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"[INFO] Norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")

    # Cosine similarity between first two chunks
    if embeddings.shape[0] >= 2:
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] # type: ignore
        print(f"[INFO] Cosine similarity between chunk 0 and 1: {sim:.4f}")
    
    sim = cosine_similarity([embeddings[0]], [embeddings[100]])[0][0] #type: ignore
    print(f"[INFO] Cosine similarity between chunk 0 and 100: {sim:.4f}")
    print(sim)


def test_retrieval():
    retriever = Retriever()
    query = "bias metrics in LLMs"
    results = retriever.retrieve(query, top_k=3)
    print(f"\n[INFO] Retrieval test for query: {query}")
    for r in results:
        print(f"  Rank {r['rank']} | Score {r['score']:.3f} | Paper {r['meta']['paper_id']}")
        print(f"  Text: {r['text'][:200]}...\n")

if __name__ == "__main__":
    check_embeddings()
    test_retrieval()
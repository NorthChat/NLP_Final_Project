"""
Retrieval module: Load index and retrieve relevant chunks
"""
import json
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch

from config import *


class Retriever:
    """Handles document retrieval using FAISS"""
    
    def __init__(self, embed_model: str = EMBED_MODELS["minilm"]):
        """
        Initialize retriever
        
        Args:
            embed_model: Embedding model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Retriever] Loading on device: {self.device}")
        
        # Load embedding model
        self.embedder = SentenceTransformer(embed_model, device=self.device)
        
        # Load FAISS index
        self.index = faiss.read_index(str(INDEX_FILE))
        print(f"[Retriever] Loaded index with {self.index.ntotal} vectors")
        
        # Load chunks and metadata
        self.texts = []
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.texts.append(obj["text"])
        
        self.metas = []
        with open(METAS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                self.metas.append(json.loads(line))
        
        print(f"[Retriever] Loaded {len(self.texts)} chunks")
        
        # Cache for repeated queries
        self.cache = {}
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks for a query
        
        Args:
            query: Query string
            top_k: Number of results to return
            use_cache: Whether to use cached results
        
        Returns:
            List of dicts with 'score', 'meta', 'text', 'rank'
        """
        # Check cache
        cache_key = f"{query}_{top_k}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Encode query
        q_emb = self.embedder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32").reshape(1, -1)  # type: ignore
        
        # Search
        scores, indices = self.index.search(q_emb, top_k)
        
        # Format results
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            results.append({
                "rank": rank,
                "score": float(score),
                "meta": self.metas[idx],
                "text": self.texts[idx]
            })
        
        # Cache results
        if use_cache:
            self.cache[cache_key] = results
        
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = TOP_K
    ) -> List[List[Dict]]:
        """
        Retrieve for multiple queries (for evaluation)
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
        
        Returns:
            List of results for each query
        """
        all_results = []
        for query in queries:
            results = self.retrieve(query, top_k, use_cache=False)
            all_results.append(results)
        return all_results
    
    def get_retrieval_stats(self) -> Dict:
        """Get statistics about the retrieval system"""
        return {
            "total_chunks": len(self.texts),
            "index_size": self.index.ntotal,
            "embedding_dim": self.embedder.get_sentence_embedding_dimension(),
            "device": self.device,
            "cache_size": len(self.cache)
        }


def test_retrieval():
    """Test retrieval with sample queries"""
    retriever = Retriever()
    
    test_queries = [
        "What are common bias metrics in LLMs?",
        "How can we mitigate bias in language models?",
        "What datasets are used for fairness evaluation?"
    ]
    
    print("\n" + "="*60)
    print("TESTING RETRIEVAL")
    print("="*60 + "\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        results = retriever.retrieve(query, top_k=3)
        
        for r in results:
            print(f"  Rank {r['rank']}: {r['meta']['paper_id']} (score: {r['score']:.3f})")
            print(f"  Text: {r['text'][:200]}...")
            print()
        print("-" * 60)


if __name__ == "__main__":
    test_retrieval()
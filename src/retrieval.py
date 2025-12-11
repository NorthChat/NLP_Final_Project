"""
Retrieval module with automatic model detection
PREVENTS dimension mismatch errors
"""
import json
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
import re

from config import *


class Retriever:
    """Handles document retrieval with automatic model detection"""
    
    def __init__(self, embed_model: str = None): #type: ignore
        """
        Initialize retriever
        
        Args:
            embed_model: Embedding model to use (auto-detected if None)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Retriever] Loading on device: {self.device}")
        
        # Auto-detect model from version.json if not specified
        if embed_model is None:
            embed_model = self._detect_indexed_model()
            print(f"[Retriever] Auto-detected model: {embed_model}")
        else:
            print(f"[Retriever] Using specified model: {embed_model}")
        
        self.is_e5_model = "e5-" in embed_model.lower()
        self.embed_model_name = embed_model
        
        # Load embedding model
        print(f"[Retriever] Loading embedding model...")
        self.embedder = SentenceTransformer(embed_model, device=self.device)
        
        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_FILE}.\n"
                f"Please run: python src/main.py --prepare --model <model_name>"
            )
        
        # Load FAISS index
        self.index = faiss.read_index(str(INDEX_FILE))
        
        # Verify dimensions match
        model_dim = self.embedder.get_sentence_embedding_dimension()
        index_dim = self.index.d
        
        if model_dim != index_dim:
            raise ValueError(
                f"\n{'='*70}\n"
                f"DIMENSION MISMATCH ERROR\n"
                f"{'='*70}\n"
                f"Model dimension: {model_dim}D\n"
                f"Index dimension: {index_dim}D\n"
                f"\nThe embedding model ({embed_model}) doesn't match\n"
                f"the model used to build the index.\n"
                f"\nSOLUTION: Rebuild the index with the correct model:\n"
                f"  python src/main.py --prepare --force-rebuild --model <model>\n"
                f"\nAvailable models: {', '.join(EMBED_MODELS.keys())}\n"
                f"{'='*70}\n"
            )
        
        print(f"[Retriever] ✓ Dimension check passed ({model_dim}D)")
        print(f"[Retriever] Loaded index with {self.index.ntotal} vectors")
        
        # Load chunks
        self.chunks = []
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))
        
        self.metas = []
        with open(METAS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                self.metas.append(json.loads(line))
        
        # Build chunk organization
        self._build_chunk_index()
        
        print(f"[Retriever] Loaded {len(self.chunks)} chunks from {len(self.paper_chunks)} papers")
        if self.is_e5_model:
            print(f"[Retriever] E5 mode: Queries will be prefixed with 'query: '")
        
        self.cache = {}
    
    def _detect_indexed_model(self) -> str:
        """
        Auto-detect which model was used for indexing
        
        Returns:
            Model name from version.json or default
        """
        version_file = DATA_DIR / "version.json"
        
        if version_file.exists():
            try:
                with open(version_file, 'r') as f:
                    version_data = json.load(f)
                    indexed_model = version_data.get('config', {}).get('embed_model')
                    
                    if indexed_model:
                        print(f"[Retriever] Found indexed model in version.json: {indexed_model}")
                        return indexed_model
            except Exception as e:
                print(f"[Retriever] Warning: Could not read version.json: {e}")
        
        # Fallback: Try to detect from index dimensions
        if INDEX_FILE.exists():
            index = faiss.read_index(str(INDEX_FILE))
            index_dim = index.d
            
            print(f"[Retriever] Index dimension: {index_dim}D")
            
            # Map dimensions to likely models
            if index_dim == 384:
                print(f"[Retriever] 384D detected - defaulting to 'minilm'")
                print(f"[Retriever] (Could also be: bge, e5-small)")
                return EMBED_MODELS["minilm"]
            elif index_dim == 768:
                print(f"[Retriever] 768D detected - defaulting to 'e5-base'")
                print(f"[Retriever] (Could also be: bge-base)")
                return EMBED_MODELS["e5-base"]
        
        # Last resort
        print(f"[Retriever] Could not detect model - defaulting to 'minilm'")
        print(f"[Retriever] If you get errors, rebuild with: python src/main.py --prepare")
        return EMBED_MODELS["minilm"]
    
    def _build_chunk_index(self):
        """Organize chunks by paper"""
        self.paper_chunks = {}
        
        for idx, chunk in enumerate(self.chunks):
            paper_id = chunk['paper_id']
            if paper_id not in self.paper_chunks:
                self.paper_chunks[paper_id] = []
            self.paper_chunks[paper_id].append({
                'idx': idx,
                'chunk_id': chunk['chunk_id'],
                'text': chunk['text']
            })
        
        for paper_id in self.paper_chunks:
            self.paper_chunks[paper_id].sort(
                key=lambda x: self._get_chunk_num(x['chunk_id'])
            )
    
    def _get_chunk_num(self, chunk_id: str) -> int:
        """Extract chunk number"""
        match = re.search(r'chunk(\d+)', chunk_id)
        return int(match.group(1)) if match else 0
    
    def _get_neighbors(self, chunk_idx: int, before: int = 1, after: int = 1) -> Dict:
        """Get neighboring chunks"""
        if chunk_idx >= len(self.chunks):
            return {'before': [], 'after': []}
        
        chunk = self.chunks[chunk_idx]
        paper_id = chunk['paper_id']
        paper_list = self.paper_chunks.get(paper_id, [])
        
        pos = None
        for i, pc in enumerate(paper_list):
            if pc['idx'] == chunk_idx:
                pos = i
                break
        
        if pos is None:
            return {'before': [], 'after': []}
        
        neighbors_before = [paper_list[i] for i in range(max(0, pos - before), pos)]
        neighbors_after = [paper_list[i] for i in range(pos + 1, min(len(paper_list), pos + after + 1))]
        
        return {'before': neighbors_before, 'after': neighbors_after}
    
    def _preprocess_query(self, query: str) -> str:
        """Add E5 prefix if needed"""
        if self.is_e5_model and not query.startswith("query: "):
            return f"query: {query}"
        return query
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        use_cache: bool = True,
        include_neighbors: bool = True,
        neighbors_before: int = 1,
        neighbors_after: int = 1
    ) -> List[Dict]:
        """
        Retrieve top-k chunks with optional neighboring context
        
        Args:
            query: Query string
            top_k: Number of results
            use_cache: Use cache
            include_neighbors: Include neighboring chunks
            neighbors_before/after: Number of neighbor chunks
        
        Returns:
            List of results with optional neighbors
        """
        cache_key = f"{query}_{top_k}_{include_neighbors}_{neighbors_before}_{neighbors_after}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        processed_query = self._preprocess_query(query)
        
        q_emb = self.embedder.encode(
            processed_query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32").reshape(1, -1) #type: ignore
        
        scores, indices = self.index.search(q_emb, top_k)
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            result = {
                "rank": rank,
                "score": float(score),
                "meta": self.metas[idx],
                "text": self.chunks[idx]['text']
            }
            
            if include_neighbors:
                neighbors = self._get_neighbors(
                    idx, 
                    before=neighbors_before, 
                    after=neighbors_after
                )
                result['neighbors_before'] = neighbors['before']
                result['neighbors_after'] = neighbors['after']
            
            results.append(result)
        
        if use_cache:
            self.cache[cache_key] = results
        
        return results
    
    def retrieve_with_expanded_text(
        self,
        query: str,
        top_k: int = TOP_K,
        window_size: int = 3
    ) -> List[Dict]:
        """Retrieve with pre-combined expanded text"""
        neighbors_count = window_size // 2
        results = self.retrieve(
            query,
            top_k=top_k,
            include_neighbors=True,
            neighbors_before=neighbors_count,
            neighbors_after=neighbors_count
        )
        
        for result in results:
            parts = []
            for neighbor in result.get('neighbors_before', []):
                parts.append(neighbor['text'])
            parts.append(result['text'])
            for neighbor in result.get('neighbors_after', []):
                parts.append(neighbor['text'])
            
            result['expanded_text'] = " ".join(parts)
            result['expansion_info'] = {
                'before_count': len(result.get('neighbors_before', [])),
                'after_count': len(result.get('neighbors_after', [])),
                'total_chunks': len(parts)
            }
        
        return results
    
    def batch_retrieve(self, queries: List[str], top_k: int = TOP_K) -> List[List[Dict]]:
        """Retrieve for multiple queries"""
        return [self.retrieve(q, top_k, use_cache=False) for q in queries]
    
    def get_retrieval_stats(self) -> Dict:
        """Get retrieval statistics"""
        return {
            "total_chunks": len(self.chunks),
            "total_papers": len(self.paper_chunks),
            "index_size": self.index.ntotal,
            "embedding_dim": self.embedder.get_sentence_embedding_dimension(),
            "embedding_model": self.embed_model_name,
            "device": self.device,
            "is_e5_model": self.is_e5_model
        }


def test_retrieval():
    """Test retrieval"""
    print("\n" + "="*70)
    print("TESTING RETRIEVAL")
    print("="*70 + "\n")
    
    try:
        retriever = Retriever()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    
    query = "What metrics measure bias in LLMs?"
    print(f"Query: {query}\n")
    
    print("--- Standard Retrieval ---")
    results = retriever.retrieve(query, top_k=3, include_neighbors=False)
    for r in results:
        print(f"[{r['rank']}] {r['meta']['paper_id']} | Score: {r['score']:.3f}")
        print(f"    {r['text'][:120]}...\n")


if __name__ == "__main__":
    test_retrieval()
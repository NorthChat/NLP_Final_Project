"""
Enhanced retrieval module with contextual chunk expansion
Retrieves neighboring chunks to provide more complete context
"""
import json
from typing import List, Dict, Tuple, Set
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch

from config import *


class EnhancedRetriever:
    """Handles document retrieval with contextual expansion"""
    
    def __init__(self, embed_model: str = EMBED_MODELS["minilm"]):
        """Initialize retriever with context expansion capabilities"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[EnhancedRetriever] Loading on device: {self.device}")
        
        # Check if using E5 model
        self.is_e5_model = "e5-" in embed_model.lower()
        self.embed_model_name = embed_model
        
        # Load embedding model
        self.embedder = SentenceTransformer(embed_model, device=self.device)
        
        # Load FAISS index
        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_FILE}. "
                "Please run 'python src/main.py --prepare' first."
            )
        
        self.index = faiss.read_index(str(INDEX_FILE))
        print(f"[EnhancedRetriever] Loaded index with {self.index.ntotal} vectors")
        
        # Load chunks and metadata
        self.chunks = []
        if not CHUNKS_FILE.exists():
            raise FileNotFoundError(f"Chunks file not found at {CHUNKS_FILE}")
            
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))
        
        self.metas = []
        if not METAS_FILE.exists():
            raise FileNotFoundError(f"Metadata file not found at {METAS_FILE}")
            
        with open(METAS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                self.metas.append(json.loads(line))
        
        # Build chunk index for fast lookup
        self._build_chunk_index()
        
        print(f"[EnhancedRetriever] Loaded {len(self.chunks)} chunks")
        print(f"[EnhancedRetriever] Organized into {len(self.paper_chunks)} papers")
        if self.is_e5_model:
            print(f"[EnhancedRetriever] Using E5 model - queries will be prefixed")
        
        self.cache = {}
    
    def _build_chunk_index(self):
        """Build index to find neighboring chunks efficiently"""
        # Group chunks by paper_id and sort by chunk_id
        self.paper_chunks = {}
        self.chunk_to_idx = {}  # Map chunk_id to its index in self.chunks
        
        for idx, chunk in enumerate(self.chunks):
            paper_id = chunk['paper_id']
            chunk_id = chunk['chunk_id']
            
            if paper_id not in self.paper_chunks:
                self.paper_chunks[paper_id] = []
            
            self.paper_chunks[paper_id].append({
                'idx': idx,
                'chunk_id': chunk_id,
                'text': chunk['text']
            })
            
            self.chunk_to_idx[chunk_id] = idx
        
        # Sort chunks within each paper by chunk number
        for paper_id in self.paper_chunks:
            self.paper_chunks[paper_id].sort(
                key=lambda x: self._extract_chunk_number(x['chunk_id'])
            )
    
    def _extract_chunk_number(self, chunk_id: str) -> int:
        """Extract chunk number from chunk_id (e.g., 'paper_chunk5' -> 5)"""
        import re
        match = re.search(r'chunk(\d+)', chunk_id)
        return int(match.group(1)) if match else 0
    
    def _preprocess_query(self, query: str) -> str:
        """Add model-specific prefixes if needed"""
        if self.is_e5_model:
            if not query.startswith("query: "):
                return f"query: {query}"
        return query
    
    def _get_neighboring_chunks(
        self, 
        chunk_idx: int, 
        before: int = 1, 
        after: int = 1
    ) -> List[Dict]:
        """
        Get neighboring chunks from the same paper
        
        Args:
            chunk_idx: Index of the central chunk
            before: Number of chunks to retrieve before
            after: Number of chunks to retrieve after
        
        Returns:
            List of neighboring chunks with their indices
        """
        if chunk_idx >= len(self.chunks):
            return []
        
        chunk = self.chunks[chunk_idx]
        paper_id = chunk['paper_id']
        
        # Get all chunks from this paper
        paper_chunk_list = self.paper_chunks.get(paper_id, [])
        
        # Find position of current chunk
        current_pos = None
        for pos, pc in enumerate(paper_chunk_list):
            if pc['idx'] == chunk_idx:
                current_pos = pos
                break
        
        if current_pos is None:
            return []
        
        # Get neighboring chunks
        neighbors = []
        
        # Before chunks
        for i in range(max(0, current_pos - before), current_pos):
            pc = paper_chunk_list[i]
            neighbors.append({
                'idx': pc['idx'],
                'chunk_id': pc['chunk_id'],
                'text': pc['text'],
                'position': 'before',
                'distance': current_pos - i
            })
        
        # After chunks
        for i in range(current_pos + 1, min(len(paper_chunk_list), current_pos + after + 1)):
            pc = paper_chunk_list[i]
            neighbors.append({
                'idx': pc['idx'],
                'chunk_id': pc['chunk_id'],
                'text': pc['text'],
                'position': 'after',
                'distance': i - current_pos
            })
        
        return neighbors
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        use_cache: bool = True,
        expand_context: bool = True,
        context_before: int = 1,
        context_after: int = 1
    ) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks with optional context expansion
        
        Args:
            query: Query string
            top_k: Number of initial results to return
            use_cache: Whether to use cached results
            expand_context: Whether to include neighboring chunks
            context_before: Number of chunks before each result
            context_after: Number of chunks after each result
        
        Returns:
            List of dicts with 'score', 'meta', 'text', 'rank', optionally 'context_before', 'context_after'
        """
        # Check cache
        cache_key = f"{query}_{top_k}_{expand_context}_{context_before}_{context_after}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Preprocess query
        processed_query = self._preprocess_query(query)
        
        # Encode query
        q_emb = self.embedder.encode(
            processed_query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32").reshape(1, -1)  #type: ignore
        
        # Search
        scores, indices = self.index.search(q_emb, top_k)
        
        # Format results
        results = []
        seen_indices = set()
        
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            result = {
                "rank": rank,
                "score": float(score),
                "meta": self.metas[idx],
                "text": self.chunks[idx]['text'],
                "chunk_idx": idx
            }
            
            # Add context expansion
            if expand_context:
                neighbors = self._get_neighboring_chunks(
                    idx, 
                    before=context_before, 
                    after=context_after
                )
                
                result['context_before'] = [
                    n for n in neighbors if n['position'] == 'before'
                ]
                result['context_after'] = [
                    n for n in neighbors if n['position'] == 'after'
                ]
                
                # Mark all indices as seen to avoid duplicates
                seen_indices.add(idx)
                for n in neighbors:
                    seen_indices.add(n['idx'])
            
            results.append(result)
        
        # Cache results
        if use_cache:
            self.cache[cache_key] = results
        
        return results
    
    def retrieve_with_window(
        self,
        query: str,
        top_k: int = TOP_K,
        window_size: int = 3
    ) -> List[Dict]:
        """
        Retrieve chunks and return them with surrounding context as single text
        
        Args:
            query: Query string
            top_k: Number of results
            window_size: Total window size (center chunk + before + after)
        
        Returns:
            List of results with expanded text including context
        """
        context_size = window_size // 2
        results = self.retrieve(
            query, 
            top_k=top_k, 
            expand_context=True,
            context_before=context_size,
            context_after=context_size
        )
        
        # Combine context into single text
        for result in results:
            text_parts = []
            
            # Add before context
            for ctx in result.get('context_before', []):
                text_parts.append(ctx['text'])
            
            # Add main chunk
            text_parts.append(result['text'])
            
            # Add after context
            for ctx in result.get('context_after', []):
                text_parts.append(ctx['text'])
            
            # Create expanded text
            result['expanded_text'] = " ".join(text_parts)
            result['has_expansion'] = len(text_parts) > 1
        
        return results
    
    def retrieve_with_reranking(
        self,
        query: str,
        initial_k: int = 20,
        final_k: int = 5,
        expand_context: bool = True
    ) -> List[Dict]:
        """
        Two-stage retrieval: broad initial search + context expansion + reranking
        
        Args:
            query: Query string
            initial_k: Number of chunks in initial retrieval
            final_k: Number of chunks to return after reranking
            expand_context: Whether to expand with neighboring chunks
        
        Returns:
            Reranked results with context
        """
        # Stage 1: Retrieve more chunks initially
        initial_results = self.retrieve(
            query,
            top_k=initial_k,
            expand_context=expand_context,
            context_before=1,
            context_after=1
        )
        
        # Stage 2: Rerank using expanded text
        if expand_context:
            # Create expanded texts for reranking
            expanded_texts = []
            for result in initial_results:
                parts = [result['text']]
                
                for ctx in result.get('context_before', []):
                    parts.insert(0, ctx['text'])
                
                for ctx in result.get('context_after', []):
                    parts.append(ctx['text'])
                
                expanded_texts.append(" ".join(parts))
            
            # Re-encode with expanded context
            processed_query = self._preprocess_query(query)
            query_emb = self.embedder.encode(
                processed_query,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype("float32") #type: ignore
            
            expanded_embs = self.embedder.encode(
                expanded_texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype("float32") #type: ignore
            
            # Compute new scores
            new_scores = np.dot(expanded_embs, query_emb)
            
            # Update scores and rerank
            for i, result in enumerate(initial_results):
                result['original_score'] = result['score']
                result['reranked_score'] = float(new_scores[i])
                result['score'] = float(new_scores[i])
            
            # Sort by new scores
            initial_results.sort(key=lambda x: x['reranked_score'], reverse=True)
            
            # Update ranks
            for rank, result in enumerate(initial_results, 1):
                result['rank'] = rank
        
        # Return top final_k
        return initial_results[:final_k]
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = TOP_K,
        **kwargs
    ) -> List[List[Dict]]:
        """Retrieve for multiple queries"""
        all_results = []
        for query in queries:
            results = self.retrieve(query, top_k, use_cache=False, **kwargs)
            all_results.append(results)
        return all_results
    
    def get_retrieval_stats(self) -> Dict:
        """Get statistics about the retrieval system"""
        return {
            "total_chunks": len(self.chunks),
            "total_papers": len(self.paper_chunks),
            "index_size": self.index.ntotal,
            "embedding_dim": self.embedder.get_sentence_embedding_dimension(),
            "embedding_model": self.embed_model_name,
            "device": self.device,
            "cache_size": len(self.cache),
            "is_e5_model": self.is_e5_model
        }


def test_enhanced_retrieval():
    """Test enhanced retrieval features"""
    print("\n" + "="*70)
    print("TESTING ENHANCED RETRIEVAL")
    print("="*70 + "\n")
    
    retriever = EnhancedRetriever()
    
    query = "What metrics measure bias in LLMs?"
    print(f"Query: {query}\n")
    
    # Test 1: Standard retrieval
    print("--- STANDARD RETRIEVAL ---")
    results = retriever.retrieve(query, top_k=3, expand_context=False)
    for r in results:
        print(f"[{r['rank']}] Score: {r['score']:.3f} | {r['meta']['paper_id']}")
        print(f"    {r['text'][:150]}...")
    print()
    
    # Test 2: With context expansion
    print("--- WITH CONTEXT EXPANSION (±1 chunk) ---")
    results = retriever.retrieve(query, top_k=3, expand_context=True, 
                                 context_before=1, context_after=1)
    for r in results:
        print(f"[{r['rank']}] Score: {r['score']:.3f} | {r['meta']['paper_id']}")
        print(f"    Main: {r['text'][:100]}...")
        
        if r.get('context_before'):
            print(f"    Before ({len(r['context_before'])}): {r['context_before'][0]['text'][:100]}...")
        
        if r.get('context_after'):
            print(f"    After ({len(r['context_after'])}): {r['context_after'][0]['text'][:100]}...")
        print()
    
    # Test 3: Window-based retrieval
    print("--- WINDOW-BASED RETRIEVAL (window_size=5) ---")
    results = retriever.retrieve_with_window(query, top_k=2, window_size=5)
    for r in results:
        print(f"[{r['rank']}] Score: {r['score']:.3f} | {r['meta']['paper_id']}")
        print(f"    Expanded: {r['has_expansion']}")
        print(f"    Text: {r['expanded_text'][:200]}...")
        print()
    
    # Test 4: Reranking
    print("--- WITH RERANKING (initial=10, final=3) ---")
    results = retriever.retrieve_with_reranking(query, initial_k=10, final_k=3)
    for r in results:
        print(f"[{r['rank']}] Original: {r.get('original_score', 0):.3f} | "
              f"Reranked: {r['reranked_score']:.3f} | {r['meta']['paper_id']}")
        print(f"    {r['text'][:150]}...")
        print()


if __name__ == "__main__":
    test_enhanced_retrieval()
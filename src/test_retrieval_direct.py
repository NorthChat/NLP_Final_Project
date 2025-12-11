# test_retrieval_direct.py
import sys
sys.path.insert(0, 'src')

from retrieval import Retriever

retriever = Retriever()

queries = [
    "AI-AI bias manifest",
    "three application domains consumer products papers movies",
    "downstream effects gate tax digital divide"
]

for query in queries:
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print('='*70)
    
    results = retriever.retrieve(query, top_k=3, include_neighbors=False)
    
    for r in results:
        print(f"\n[{r['rank']}] {r['meta']['paper_id']} | Score: {r['score']:.3f}")
        print(f"Text: {r['text'][:300]}...")
        
        # Check if it mentions key terms
        text_lower = r['text'].lower()
        if 'consumer product' in text_lower or 'movie' in text_lower or 'paper' in text_lower:
            print("✓ MENTIONS KEY TERMS!")
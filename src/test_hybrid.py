import sys
sys.path.insert(0, 'src')

# You need to use the advanced retriever with hybrid search
# For now, let's test if exact term matching helps

from retrieval import Retriever

retriever = Retriever()

# Test with very specific keywords
query = "consumer products marketplace classified ads scientific papers abstracts movies plot summaries"

results = retriever.retrieve(query, top_k=5)

print("Results for specific keywords:")
for r in results:
    print(f"\n{r['meta']['paper_id']} | {r['score']:.3f}")
    print(r['text'][:200])
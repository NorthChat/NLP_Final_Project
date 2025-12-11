"""
Quick test script to compare embedding models and prompts
"""
import sys
sys.path.insert(0, 'src')

from pathlib import Path
from retrieval import Retriever
from generation import Generator
from config import EMBED_MODELS
import json

def test_embedding_model(model_name: str, test_queries: list):
    """Test a specific embedding model"""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}\n")
    
    try:
        retriever = Retriever(embed_model=EMBED_MODELS[model_name])
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = retriever.retrieve(query, top_k=5)
            
            print(f"Top 3 Results:")
            for r in results[:3]:
                print(f"  [{r['rank']}] Score: {r['score']:.3f} | {r['meta']['paper_id']}")
                print(f"      {r['text'][:150]}...")
            print()
        
        return True
    except Exception as e:
        print(f"ERROR with {model_name}: {e}")
        return False

def test_prompt_versions(retriever, generator, question: str):
    """Test different prompt versions"""
    print(f"\n{'='*70}")
    print(f"Testing Prompt Versions")
    print(f"{'='*70}\n")
    
    print(f"Question: {question}\n")
    
    # Retrieve once
    chunks = retriever.retrieve(question, top_k=5)
    print(f"Retrieved from: {', '.join(set(c['meta']['paper_id'] for c in chunks[:3]))}")
    scores = [c['score'] for c in chunks[:3]]
    print(f"Top scores: {[f'{s:.3f}' for s in scores]}\n")  

    # Test each prompt version 
    for version in [1, 2, 3]:
        print(f"\n--- PROMPT VERSION {version} ---")
        try:
            # If you've added prompt_version parameter
            result = generator.generate(question, chunks, prompt_version=version)
            print(f"Answer: {result['answer']}")
        except TypeError:
            # If not yet implemented, just use default
            result = generator.generate(question, chunks)
            print(f"Answer: {result['answer']}")
            print("\n(Note: prompt_version not yet implemented)")
            break
        print()

def main():
    """Run comparison tests"""
    
    # Test queries
    test_queries = [
        "What metrics measure bias in LLMs?",
        "How can we mitigate bias in language models?",
        "What datasets are used for fairness evaluation?"
    ]
    
    print("\n" + "="*70)
    print("RETRIEVAL QUALITY TESTING")
    print("="*70)
    
    # Test different embedding models
    models_to_test = ["minilm", "bge"]
    
    # Add new models if you've updated config.py
    if "e5-base" in EMBED_MODELS:
        models_to_test.append("e5-base")
    if "bge-base" in EMBED_MODELS:
        models_to_test.append("bge-base")
    
    print("\n[1] Testing Embedding Models")
    print("-" * 70)
    
    working_models = []
    for model_name in models_to_test:
        if test_embedding_model(model_name, test_queries[:2]):  # Test with 2 queries
            working_models.append(model_name)
    
    if working_models:
        print(f"\n✓ Working models: {', '.join(working_models)}")
        
        # Test prompts with best model
        print("\n[2] Testing Prompt Variations")
        print("-" * 70)
        
        best_model = working_models[-1]  # Use last working model
        retriever = Retriever(embed_model=EMBED_MODELS[best_model])
        generator = Generator()
        
        test_prompt_versions(retriever, generator, test_queries[0])
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print("\nRecommendations:")
    print("1. Best embedding model based on scores above")
    print("2. Best prompt version based on answer quality")
    print("3. Consider increasing Top-K if scores are low")
    print()

if __name__ == "__main__":
    main()
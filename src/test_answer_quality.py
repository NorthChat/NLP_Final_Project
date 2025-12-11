"""
Test different prompts with OPTIMIZED retrieval settings
"""
import sys
sys.path.insert(0, 'src')

from retrieval import Retriever
from generation import Generator
from pathlib import Path
import json

def test_prompts():
    print("\n" + "="*70)
    print("TESTING PROMPTS (OPTIMIZED)")
    print("="*70 + "\n")
    
    # Load retriever and generator
    print("[INFO] Initializing retriever and generator...")
    try:
        retriever = Retriever()
        generator = Generator()
        print("[INFO] Successfully loaded models\n")
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        print("[INFO] Make sure you've run: python src/main.py --prepare")
        return
    
    # Load Q/A pairs
    qa_file = Path(__file__).parent.parent / 'qa_pairs.json'
    
    if not qa_file.exists():
        print(f"[ERROR] Cannot find qa_pairs.json at {qa_file}")
        return
    
    with open(qa_file, 'r') as f:
        qa_data = json.load(f)
        test_questions = qa_data[:3]
        print(f"[INFO] Loaded {len(test_questions)} test questions\n")
    
    # Test different prompt versions
    prompt_versions_to_test = [2, 5]
    
    for i, qa in enumerate(test_questions, 1):
        question = qa['question']
        reference = qa['answer']
        
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_questions)}")
        print(f"{'='*70}")
        print(f"\nQuestion: {question}")
        print(f"\nReference Answer:\n{reference}")
        
        # Retrieve MORE chunks so generator can filter junk
        chunks = retriever.retrieve(
            question, 
            top_k=15,  # Get more candidates for filtering
            include_neighbors=False
        )
        
        print(f"\nRetrieved {len(chunks)} chunks (expanded)")
        print(f"Top papers: {', '.join(set(c['meta']['paper_id'] for c in chunks[:3]))}")
        print(f"Top score: {chunks[0]['score']:.3f}")
        
        # Test each prompt version
        for pv in prompt_versions_to_test:
            print(f"\n{'-'*70}")
            print(f"PROMPT VERSION {pv}")
            print(f"{'-'*70}")
            
            try:
                # Generate with specific prompt version
                result = generator.generate(
                    question, 
                    chunks, 
                    prompt_version=pv
                )
                
                print(f"\nGenerated Answer:\n{result['answer']}")
                print(f"\nMethod: {result['method']}")
                print(f"Chunks used: {result['chunks_used']}/{result['chunks_available']}")
                
                if result['method'] == 'multipass-with-memory':
                    print(f"Windows: {result.get('windows_used', 'N/A')}")
                    print(f"Findings extracted: {result.get('findings_extracted', 'N/A')}")
                
            except Exception as e:
                print(f"[ERROR] Generation failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Quality assessment
        print(f"\n{'─'*70}")
        print("QUALITY ASSESSMENT:")
        print(f"{'─'*70}")
        print("✓ Specific? (uses concrete terms)")
        print("✓ Complete? (fully addresses question)")
        print("✓ Grounded? (based on papers)")
        print("✓ Accurate? (matches reference)")
        print()
        
        if i < len(test_questions):
            input("Press Enter for next question...")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print("\nRecommendations:")
    print("─" * 70)
    print("✓ Good answers → System is working!")
    print("✗ Still vague → Fine-tune FLAN-T5 on QA pairs")
    print("✗ Repetitive → Check deduplication logs")
    print("✗ Wrong facts → Try better embedding (e5-base)")
    print()


def compare_retrieval_settings():
    """Compare different retrieval configurations"""
    print("\n" + "="*70)
    print("COMPARING RETRIEVAL SETTINGS")
    print("="*70 + "\n")
    
    retriever = Retriever()
    generator = Generator()
    
    question = "What is Auto-Debias?"
    
    configs = [
        {"top_k": 5, "window": 3, "name": "Conservative (5 chunks, window=3)"},
        {"top_k": 7, "window": 3, "name": "Balanced (7 chunks, window=3)"},
        {"top_k": 10, "window": 5, "name": "Aggressive (10 chunks, window=5)"},
    ]
    
    print(f"Question: {question}\n")
    
    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config['name']}")
        print(f"{'='*70}")
        
        chunks = retriever.retrieve_with_expanded_text(
            question,
            top_k=config['top_k'],
            window_size=config['window']
        )
        
        # Calculate total tokens
        total_tokens = sum(
            len(c.get('expanded_text', c['text']).split()) * 1.3  # Rough token estimate
            for c in chunks
        )
        
        print(f"Retrieved: {len(chunks)} chunks")
        print(f"Estimated tokens: {int(total_tokens)}")
        
        result = generator.generate(question, chunks, prompt_version=2)
        
        print(f"\nMethod: {result['method']}")
        if result['method'] == 'multipass-with-memory':
            print(f"Windows: {result.get('windows_used', 'N/A')}")
            print(f"Findings: {result.get('findings_extracted', 'N/A')}")
        
        print(f"\nAnswer:\n{result['answer'][:200]}...")
        print()
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("\nBest setting: Balanced (7 chunks, window=3)")
    print("  - Enough context for good answers")
    print("  - Not too many windows (stays under 6)")
    print("  - Reduces repetition risk")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test prompt quality")
    parser.add_argument(
        "--compare-retrieval",
        action="store_true",
        help="Compare different retrieval settings"
    )
    
    args = parser.parse_args()
    
    if args.compare_retrieval:
        compare_retrieval_settings()
    else:
        test_prompts()
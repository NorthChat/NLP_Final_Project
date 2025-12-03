"""
Test different prompts to see which produces better answers
"""
import sys
sys.path.insert(0, 'src')

from retrieval import Retriever
from generation import Generator
import json

def test_prompts():
    print("\n" + "="*70)
    print("TESTING DIFFERENT PROMPTS FOR ANSWER QUALITY")
    print("="*70 + "\n")
    
    # Load retriever and generator
    retriever = Retriever()
    generator = Generator()
    
    # Load Q/A pairs from correct path
    from pathlib import Path
    qa_file = Path(__file__).parent.parent / 'qa_pairs.json'
    
    if not qa_file.exists():
        print(f"[ERROR] Cannot find qa_pairs.json")
        print(f"[INFO] Looked in: {qa_file}")
        print("[INFO] Using sample questions instead...\n")
        
        # Use sample questions if file not found
        test_questions = [
            {
                "question": "What are common metrics used to measure bias in large language models?",
                "answer": "Common bias metrics include demographic parity, equalized odds, and counterfactual fairness."
            },
            {
                "question": "What datasets are used for fairness evaluation?",
                "answer": "Common datasets include WinoBias, BOLD, and StereoSet."
            },
            {
                "question": "How can we mitigate bias in language models?",
                "answer": "Bias mitigation techniques include data augmentation, adversarial debiasing, and regularization."
            }
        ]
    else:
        with open(qa_file, 'r') as f:
            qa_data = json.load(f)
            test_questions = qa_data[:3]
    
    for i, qa in enumerate(test_questions, 1):
        question = qa['question']
        reference = qa['answer']
        
        print(f"\n{'='*70}")
        print(f"TEST {i}")
        print(f"{'='*70}")
        print(f"\nQuestion: {question}")
        print(f"\nReference Answer:\n{reference}")
        
        # Retrieve
        chunks = retriever.retrieve(question, top_k=5)
        
        print(f"\nRetrieved from: {', '.join(set(c['meta']['paper_id'] for c in chunks[:3]))}")
        print(f"Top score: {chunks[0]['score']:.3f}")
        
        # Generate
        result = generator.generate(question, chunks)
        
        print(f"\nGenerated Answer:\n{result['answer']}")
        
        # Manual quality check prompts
        print(f"\n{'─'*70}")
        print("QUALITY CHECK:")
        print(f"{'─'*70}")
        print("Is the answer:")
        print("  [ ] Specific (uses concrete terms from context)?")
        print("  [ ] Complete (fully addresses question)?")
        print("  [ ] Grounded (based on retrieved papers)?")
        print("  [ ] Accurate (matches reference meaning)?")
        print()
        
        input("Press Enter to continue to next question...")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print("\nBased on your manual checks:")
    print("- If answers are good: Prompt is working!")
    print("- If still vague: Try FLAN-T5-XL or Mistral models")
    print("- If wrong facts: Check retrieval quality")
    print()

if __name__ == "__main__":
    test_prompts()
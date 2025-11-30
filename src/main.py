"""
Main entry point for the RAG system

Usage:
    python src/main.py --prepare                    # Prepare dataset (first time)
    python src/main.py --prepare --ablation         # Run ablation study
    python src/main.py --ui                         # Launch UI
    python src/main.py --evaluate                   # Run evaluation
    python src/main.py --test                       # Quick test
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from prepare_data import prepare_dataset
from ui import launch_ui
from evaluation import run_full_evaluation
from retrieval import test_retrieval
from generation import test_generation
from config import *


def run_ablation_study():
    """
    Run ablation study comparing different configurations
    Required for A grade
    """
    print("\n" + "="*80)
    print("RUNNING ABLATION STUDY")
    print("="*80 + "\n")
    
    from retrieval import Retriever
    from generation import Generator
    from evaluation import RAGEvaluator
    
    evaluator = RAGEvaluator()
    
    # Load Q/A pairs
    qa_pairs = evaluator.load_qa_pairs()
    print(f"[INFO] Loaded {len(qa_pairs)} Q/A pairs for ablation study\n")
    
    # Configurations to compare
    configs = [
        {"embed": "minilm", "chunk_size": 250, "name": "MiniLM-250"},
        {"embed": "minilm", "chunk_size": 450, "name": "MiniLM-450"},
        {"embed": "bge", "chunk_size": 250, "name": "BGE-250"},
        {"embed": "bge", "chunk_size": 450, "name": "BGE-450"},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}\n")
        
        # Prepare data with this config
        embed_model = EMBED_MODELS[config['embed']]
        
        print(f"[1] Preparing data with {embed_model}, chunk_size={config['chunk_size']}...")
        prepare_dataset(
            embed_model=embed_model,
            chunk_size=config['chunk_size']
        )
        
        # Evaluate
        print(f"\n[2] Evaluating {config['name']}...")
        retriever = Retriever(embed_model=embed_model)
        generator = Generator()
        
        # Retrieval evaluation
        retrieval_metrics = evaluator.evaluate_retrieval(
            qa_pairs, retriever, k_values=[5]
        )
        
        # Generation evaluation (sample)
        sample_qa = qa_pairs[:10]  # Sample for speed
        retrieval_results = [retriever.retrieve(qa["question"], top_k=5) for qa in sample_qa]
        generated = [generator.generate(qa["question"], retr)["answer"] 
                    for qa, retr in zip(sample_qa, retrieval_results)]
        
        generation_metrics = evaluator.evaluate_generation(sample_qa, generated)
        
        # Store results
        results[config['name']] = {
            "retrieval": retrieval_metrics[5],
            "generation": generation_metrics
        }
        
        print(f"\n{config['name']} Results:")
        print(f"  Precision@5: {retrieval_metrics[5]['precision@k']:.3f}")
        print(f"  Recall@5: {retrieval_metrics[5]['recall@k']:.3f}")
        print(f"  ROUGE-L: {generation_metrics['rougeL']:.3f}")
    
    # Save ablation results
    import json
    ablation_file = EVAL_DIR / "ablation_study.json"
    with open(ablation_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETE")
    print(f"Results saved to: {ablation_file}")
    print("="*80 + "\n")
    
    # Create comparison visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    names = list(results.keys())
    precision = [results[n]['retrieval']['precision@k'] for n in names]
    recall = [results[n]['retrieval']['recall@k'] for n in names]
    rouge = [results[n]['generation']['rougeL'] for n in names]
    
    # Retrieval comparison
    x = range(len(names))
    width = 0.35
    axes[0].bar([i - width/2 for i in x], precision, width, label='Precision@5', alpha=0.8)
    axes[0].bar([i + width/2 for i in x], recall, width, label='Recall@5', alpha=0.8)
    axes[0].set_xlabel('Configuration')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Retrieval Performance Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Generation comparison
    axes[1].bar(names, rouge, color='green', alpha=0.7)
    axes[1].set_xlabel('Configuration')
    axes[1].set_ylabel('ROUGE-L Score')
    axes[1].set_title('Generation Performance Comparison')
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    viz_file = EVAL_DIR / "ablation_comparison.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"[INFO] Ablation visualization saved to: {viz_file}\n")


def quick_test():
    """Quick end-to-end test"""
    print("\n" + "="*60)
    print("RUNNING QUICK TEST")
    print("="*60 + "\n")
    
    print("[1/2] Testing retrieval...")
    test_retrieval()
    
    print("\n[2/2] Testing generation...")
    test_generation()
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="RAG System for Fairness & Bias in LLMs"
    )
    
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare dataset (PDF → chunks → embeddings → index)"
    )
    
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study (use with --prepare)"
    )
    
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch Gradio UI"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run full evaluation"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link for UI"
    )
    
    args = parser.parse_args()
    
    # If no args, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    # Execute commands
    if args.prepare:
        if args.ablation:
            run_ablation_study()
        else:
            prepare_dataset()
    
    if args.test:
        quick_test()
    
    if args.evaluate:
        run_full_evaluation()
    
    if args.ui:
        launch_ui(share=args.share)


if __name__ == "__main__":
    main()
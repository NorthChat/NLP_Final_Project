"""
Main entry point for the RAG system - WITH E5 MODEL SUPPORT

Usage:
    python src/main.py --prepare                    # Prepare dataset
    python src/main.py --prepare --model e5-base    # Prepare with E5 model
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
    
    - Compare embedding models (MiniLM vs BGE vs E5)
    - Compare chunk sizes (250 vs 450)
    - Generate comprehensive comparison
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
    print(f"[INFO] Loaded {len(qa_pairs)} Q/A pairs for ablation study")
    
    # Configurations to compare
    configs = [
        {"embed": "bge", "chunk_size": 250, "name": "BGE-250"},
        {"embed": "bge", "chunk_size": 450, "name": "BGE-450"},
        {"embed": "minilm", "chunk_size": 250, "name": "MiniLM-250"},
        {"embed": "minilm", "chunk_size": 450, "name": "MiniLM-450"},
    ]
    
    # Add E5 if available
    if "e5-base" in EMBED_MODELS:
        configs.extend([
            {"embed": "e5-base", "chunk_size": 450, "name": "E5-Base-450"},
            {"embed": "e5-base", "chunk_size": 250, "name": "E5-Base-250"},
        ])
    
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
            chunk_size=config['chunk_size'],
            force_rebuild=True,
            use_sentence_chunking=True
        )
        
        # Evaluate
        print(f"\n[2] Evaluating {config['name']}...")
        retriever = Retriever(embed_model=embed_model)
        generator = Generator()
        
        # Retrieval evaluation
        retrieval_metrics = evaluator.evaluate_retrieval(
            qa_pairs, retriever, k_values=[5]
        )
        
        # Generation evaluation (sample for speed)
        sample_size = min(10, len(qa_pairs))
        sample_qa = qa_pairs[:sample_size]
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
    EVAL_DIR.mkdir(exist_ok=True)
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
    axes[0].set_title('Ablation Study: Retrieval Performance')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Generation comparison
    axes[1].bar(names, rouge, color='green', alpha=0.7)
    axes[1].set_xlabel('Configuration')
    axes[1].set_ylabel('ROUGE-L Score')
    axes[1].set_title('Ablation Study: Generation Performance')
    axes[1].set_xticks(x)
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
        description="RAG System for Fairness & Bias in LLMs - WITH E5 SUPPORT"
    )
    
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare dataset (PDF → chunks → embeddings → index)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="minilm",
        choices=list(EMBED_MODELS.keys()),
        help="Embedding model to use (default: minilm)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Chunk size in words (default: 250)"
    )
    
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force complete rebuild (ignore incremental updates)"
    )
    
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Disable incremental indexing"
    )
    
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation study"
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
        print("\n" + "="*60)
        print("QUICK START GUIDE")
        print("="*60)
        print("\n1. Prepare with MiniLM (fast):")
        print("   python src/main.py --prepare --model minilm")
        print("\n2. Prepare with E5-Base (better quality):")
        print("   python src/main.py --prepare --model e5-base")
        print("\n3. Prepare with BGE:")
        print("   python src/main.py --prepare --model bge")
        print("\n4. Run ablation study:")
        print("   python src/main.py --prepare --ablation")
        print("\n5. Test the system:")
        print("   python src/main.py --test")
        print("\n6. Launch UI:")
        print("   python src/main.py --ui")
        print("\n7. Run evaluation:")
        print("   python src/main.py --evaluate")
        print("="*60 + "\n")
        
        # Show available models
        print("Available embedding models:")
        for key, model in EMBED_MODELS.items():
            is_e5 = "e5-" in model.lower()
            e5_mark = " (E5 - needs query/passage prefixes)" if is_e5 else ""
            print(f"  - {key}: {model}{e5_mark}")
        print()
        
        sys.exit(0)
    
    # Execute commands
    if args.prepare:
        embed_model = EMBED_MODELS[args.model]
        
        print("\n" + "="*60)
        print(f"Selected model: {embed_model}")
        print(f"Chunk size: {args.chunk_size}")
        print("="*60 + "\n")
        
        if args.ablation:
            run_ablation_study()
        else:
            prepare_dataset(
                embed_model=embed_model,
                chunk_size=args.chunk_size,
                force_rebuild=args.force_rebuild,
                incremental=not args.no_incremental
            )
    
    if args.test:
        quick_test()
    
    if args.evaluate:
        run_full_evaluation()
    
    if args.ui:
        launch_ui(share=args.share)


if __name__ == "__main__":
    main()
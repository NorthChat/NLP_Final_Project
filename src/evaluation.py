"""
Comprehensive evaluation module for the RAG system
Implements: Precision@K, Recall@K, MRR, ROUGE, Human Evaluation
"""
import json
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer
import pandas as pd
from pathlib import Path

from config import *


class RAGEvaluator:
    """Comprehensive evaluator for RAG system"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        EVAL_DIR.mkdir(exist_ok=True)
    
    def load_qa_pairs(self, qa_file: Path = QA_PAIRS_FILE) -> List[Dict]:
        """
        Load annotated Q/A pairs
        
        Expected format:
        [
          {
            "question": "...",
            "answer": "...",
            "relevant_papers": ["paper1", "paper2"]  # for retrieval eval
          },
          ...
        ]
        """
        with open(qa_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def evaluate_retrieval(
        self,
        qa_pairs: List[Dict],
        retriever,
        k_values: List[int] = [3, 5, 10]
    ) -> Dict:
        """
        Evaluate retrieval quality using Precision@K, Recall@K, MRR
        
        Args:
            qa_pairs: List of Q/A pairs with relevant_papers
            retriever: Retriever instance
            k_values: List of K values to evaluate
        
        Returns:
            Dict with metrics for each K
        """
        print("\n" + "="*60)
        print("EVALUATING RETRIEVAL")
        print("="*60 + "\n")
        
        results = {k: {"precision": [], "recall": [], "mrr": []} for k in k_values}
        
        for qa in qa_pairs:
            question = qa["question"]
            relevant_papers = set(qa.get("relevant_papers", []))
            
            if not relevant_papers:
                continue
            
            # Retrieve for max K
            max_k = max(k_values)
            retrieved = retriever.retrieve(question, top_k=max_k)
            
            # Get retrieved paper IDs
            retrieved_papers = [r["meta"]["paper_id"] for r in retrieved]
            
            # Calculate metrics for each K
            for k in k_values:
                retrieved_k = retrieved_papers[:k]
                relevant_retrieved = set(retrieved_k) & relevant_papers
                
                # Precision@K
                precision = len(relevant_retrieved) / k if k > 0 else 0
                results[k]["precision"].append(precision)
                
                # Recall@K
                recall = len(relevant_retrieved) / len(relevant_papers) if relevant_papers else 0
                results[k]["recall"].append(recall)
                
                # MRR
                mrr = 0
                for rank, paper_id in enumerate(retrieved_k, 1):
                    if paper_id in relevant_papers:
                        mrr = 1.0 / rank
                        break
                results[k]["mrr"].append(mrr)
        
        # Compute averages
        summary = {}
        for k in k_values:
            summary[k] = {
                "precision@k": np.mean(results[k]["precision"]),
                "recall@k": np.mean(results[k]["recall"]),
                "mrr@k": np.mean(results[k]["mrr"]),
                "num_queries": len(results[k]["precision"])
            }
        
        return summary
    
    def evaluate_generation(
        self,
        qa_pairs: List[Dict],
        generated_answers: List[str]
    ) -> Dict:
        """
        Evaluate generated answers using ROUGE scores
        
        Args:
            qa_pairs: List of Q/A pairs with reference answers
            generated_answers: List of generated answers
        
        Returns:
            Dict with ROUGE scores
        """
        print("\n" + "="*60)
        print("EVALUATING GENERATION (ROUGE)")
        print("="*60 + "\n")
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for qa, generated in zip(qa_pairs, generated_answers):
            reference = qa["answer"]
            
            # Compute ROUGE
            scores = self.rouge_scorer.score(reference, generated)
            
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            "rouge1": np.mean(rouge1_scores),
            "rouge2": np.mean(rouge2_scores),
            "rougeL": np.mean(rougeL_scores),
            "num_examples": len(rouge1_scores)
        }
    
    def human_evaluation_template(self, qa_pairs: List[Dict], generated_answers: List[str]) -> pd.DataFrame:
        """
        Create template for human evaluation
        
        Returns DataFrame to be filled manually with scores (1-5):
        - Correctness: Is the answer factually correct?
        - Groundedness: Is the answer based on retrieved context?
        - Completeness: Does the answer fully address the question?
        """
        data = []
        for i, (qa, gen_answer) in enumerate(zip(qa_pairs, generated_answers)):
            data.append({
                "id": i,
                "question": qa["question"],
                "reference_answer": qa["answer"],
                "generated_answer": gen_answer,
                "correctness": None,  # Fill manually: 1-5
                "groundedness": None,  # Fill manually: 1-5
                "completeness": None   # Fill manually: 1-5
            })
        
        df = pd.DataFrame(data)
        
        # Save template
        template_file = EVAL_DIR / "human_eval_template.csv"
        df.to_csv(template_file, index=False)
        print(f"[INFO] Human evaluation template saved to: {template_file}")
        print("[INFO] Please fill in the scores (1-5) and save as 'human_eval_completed.csv'")
        
        return df
    
    def analyze_human_evaluation(self, csv_file: Path) -> Dict:
        """
        Analyze completed human evaluation
        
        Args:
            csv_file: Path to completed CSV with scores
        
        Returns:
            Dict with average scores and statistics
        """
        df = pd.read_csv(csv_file)
        
        metrics = {}
        for col in ["correctness", "groundedness", "completeness"]:
            if col in df.columns:
                scores = df[col].dropna()
                metrics[col] = {
                    "mean": scores.mean(),
                    "std": scores.std(),
                    "min": scores.min(),
                    "max": scores.max()
                }
        
        return metrics
    
    def error_analysis(
        self,
        qa_pairs: List[Dict],
        retrieval_results: List[List[Dict]],
        generated_answers: List[str]
    ) -> Dict:
        """
        Perform error analysis
        
        Identifies:
        - Questions with low retrieval quality
        - Questions with short/incomplete answers
        - Common failure patterns
        """
        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60 + "\n")
        
        errors = {
            "low_retrieval_score": [],
            "short_answers": [],
            "no_relevant_retrieved": []
        }
        
        for i, (qa, retrieved, answer) in enumerate(zip(qa_pairs, retrieval_results, generated_answers)):
            question = qa["question"]
            relevant_papers = set(qa.get("relevant_papers", []))
            
            # Check retrieval quality
            if retrieved:
                top_score = retrieved[0]["score"]
                if top_score < 0.3:  # Low similarity threshold
                    errors["low_retrieval_score"].append({
                        "id": i,
                        "question": question,
                        "top_score": top_score
                    })
                
                # Check if any relevant papers retrieved
                retrieved_papers = {r["meta"]["paper_id"] for r in retrieved}
                if relevant_papers and not (retrieved_papers & relevant_papers):
                    errors["no_relevant_retrieved"].append({
                        "id": i,
                        "question": question,
                        "expected": list(relevant_papers),
                        "got": list(retrieved_papers)[:3]
                    })
            
            # Check answer length
            if len(answer.split()) < 10:
                errors["short_answers"].append({
                    "id": i,
                    "question": question,
                    "answer": answer,
                    "length": len(answer.split())
                })
        
        # Print summary
        print(f"Low retrieval scores: {len(errors['low_retrieval_score'])}")
        print(f"Short answers: {len(errors['short_answers'])}")
        print(f"No relevant retrieved: {len(errors['no_relevant_retrieved'])}")
        
        # Save detailed errors
        error_file = EVAL_DIR / "error_analysis.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2)
        print(f"\n[INFO] Detailed errors saved to: {error_file}")
        
        return errors
    
    def visualize_results(self, retrieval_metrics: Dict, generation_metrics: Dict):
        """Create visualizations for the report"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Retrieval Metrics by K
        k_values = sorted(retrieval_metrics.keys())
        precision_vals = [retrieval_metrics[k]["precision@k"] for k in k_values]
        recall_vals = [retrieval_metrics[k]["recall@k"] for k in k_values]
        mrr_vals = [retrieval_metrics[k]["mrr@k"] for k in k_values]
        
        axes[0, 0].plot(k_values, precision_vals, marker='o', label='Precision@K')
        axes[0, 0].plot(k_values, recall_vals, marker='s', label='Recall@K')
        axes[0, 0].plot(k_values, mrr_vals, marker='^', label='MRR@K')
        axes[0, 0].set_xlabel('K')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Retrieval Metrics vs K')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: ROUGE Scores
        rouge_metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        rouge_scores = [
            generation_metrics['rouge1'],
            generation_metrics['rouge2'],
            generation_metrics['rougeL']
        ]
        
        axes[0, 1].bar(rouge_metrics, rouge_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('ROUGE Scores')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Placeholder for human eval (will be filled after human scoring)
        axes[1, 0].text(0.5, 0.5, 'Human Evaluation\n(Complete CSV and re-run)',
                       ha='center', va='center', fontsize=12)
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')
        
        # Plot 4: Overall Performance Summary
        summary_data = f"""
        Retrieval Performance:
        • Precision@5: {retrieval_metrics[5]['precision@k']:.3f}
        • Recall@5: {retrieval_metrics[5]['recall@k']:.3f}
        • MRR@5: {retrieval_metrics[5]['mrr@k']:.3f}
        
        Generation Performance:
        • ROUGE-1: {generation_metrics['rouge1']:.3f}
        • ROUGE-2: {generation_metrics['rouge2']:.3f}
        • ROUGE-L: {generation_metrics['rougeL']:.3f}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_data, ha='left', va='center',
                       fontsize=10, family='monospace')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        viz_file = EVAL_DIR / "evaluation_results.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"[INFO] Visualization saved to: {viz_file}")
        plt.close()


def run_full_evaluation():
    """Run complete evaluation pipeline"""
    from retrieval import Retriever
    from generation import Generator
    
    print("\n" + "="*60)
    print("RUNNING FULL EVALUATION")
    print("="*60 + "\n")
    
    # Initialize
    evaluator = RAGEvaluator()
    retriever = Retriever()
    generator = Generator()
    
    # Load Q/A pairs
    qa_pairs = evaluator.load_qa_pairs()
    print(f"[INFO] Loaded {len(qa_pairs)} Q/A pairs")
    
    # Run retrieval
    print("\n[1/4] Retrieving for all questions...")
    retrieval_results = []
    for qa in qa_pairs:
        retrieved = retriever.retrieve(qa["question"], top_k=10)
        retrieval_results.append(retrieved)
    
    # Evaluate retrieval
    print("\n[2/4] Evaluating retrieval...")
    retrieval_metrics = evaluator.evaluate_retrieval(qa_pairs, retriever, k_values=[3, 5, 10])
    
    # Generate answers
    print("\n[3/4] Generating answers...")
    generated_answers = []
    for qa, retrieved in zip(qa_pairs, retrieval_results):
        result = generator.generate(qa["question"], retrieved[:5])
        generated_answers.append(result["answer"])
    
    # Evaluate generation
    print("\n[4/4] Evaluating generation...")
    generation_metrics = evaluator.evaluate_generation(qa_pairs, generated_answers)
    
    # Error analysis
    errors = evaluator.error_analysis(qa_pairs, retrieval_results, generated_answers)
    
    # Visualize
    evaluator.visualize_results(retrieval_metrics, generation_metrics)
    
    # Human evaluation template
    evaluator.human_evaluation_template(qa_pairs, generated_answers)
    
    # Save all results
    results = {
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
        "error_counts": {k: len(v) for k, v in errors.items()}
    }
    
    results_file = EVAL_DIR / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[INFO] Results saved to: {results_file}")
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    run_full_evaluation()
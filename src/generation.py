"""
Generation module: Generate answers using retrieved context
"""
from typing import List, Dict
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

from config import *


class Generator:
    """Handles answer generation using retrieved context"""
    
    def __init__(self, model_name: str = GEN_MODELS["flan-t5-large"]):
        """
        Initialize generator
        
        Args:
            model_name: Name of generation model
        """
        print(f"[Generator] Loading model: {model_name}")
        
        # Check CUDA availability
        use_cuda = torch.cuda.is_available()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with appropriate settings
        if use_cuda:
            # Use device_map="auto" for efficient GPU loading
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            # Don't specify device when using device_map
            self.pipe = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            print(f"[Generator] Model loaded on GPU with device_map='auto'")
        else:
            # CPU mode - load normally
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            self.pipe = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1
            )
            print(f"[Generator] Model loaded on CPU")
    
    def generate(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        max_length: int = MAX_GEN_LENGTH,
        include_citations: bool = True
    ) -> Dict:
        """
        Generate answer from retrieved context
        
        Args:
            question: User question
            retrieved_chunks: List of retrieved chunks from Retriever
            max_length: Maximum generation length
            include_citations: Whether to include source citations
        
        Returns:
            Dict with 'answer', 'sources', 'prompt'
        """
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[{i}] {chunk['text']}")
            sources.append({
                "rank": chunk["rank"],
                "paper_id": chunk["meta"]["paper_id"],
                "score": chunk["score"]
            })
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = self._build_prompt(question, context)
        
        # Generate
        output = self.pipe(
            prompt,
            max_length=max_length,
            do_sample=False,  # Deterministic generation
            num_beams=4,  # Beam search for better quality
            early_stopping=True
        )
        
        # Extract generated text
        generated_text = output[0]["generated_text"]  # type: ignore[index]
        
        # Clean output
        answer = str(generated_text).strip()
        
        return {
            "answer": answer,
            "sources": sources if include_citations else [],
            "prompt": prompt,
            "question": question
        }
    
    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build prompt for the model
        
        Template optimized for FLAN-T5
        """
        prompt = f"""Context:
{context}

Question: {question}

Based on the context above, provide a detailed and accurate answer. Use information directly from the context.

Answer:"""
        return prompt
    
    def batch_generate(
        self,
        questions: List[str],
        retrieved_results: List[List[Dict]],
        **kwargs
    ) -> List[Dict]:
        """
        Generate answers for multiple questions (for evaluation)
        
        Args:
            questions: List of questions
            retrieved_results: List of retrieval results for each question
            **kwargs: Additional arguments for generate()
        
        Returns:
            List of generation results
        """
        results = []
        for question, chunks in zip(questions, retrieved_results):
            result = self.generate(question, chunks, **kwargs)
            results.append(result)
        return results


def test_generation():
    """Test generation with sample query"""
    from retrieval import Retriever
    
    print("\n" + "="*60)
    print("TESTING GENERATION")
    print("="*60 + "\n")
    
    # Initialize
    retriever = Retriever()
    generator = Generator()
    
    # Test query
    question = "What are common metrics used to measure bias in large language models?"
    
    print(f"Question: {question}\n")
    
    # Retrieve
    print("Retrieving relevant chunks...")
    chunks = retriever.retrieve(question, top_k=5)
    
    print(f"Retrieved {len(chunks)} chunks\n")
    
    # Generate
    print("Generating answer...")
    result = generator.generate(question, chunks)
    
    print(f"\nAnswer:\n{result['answer']}\n")
    
    print("Sources:")
    for src in result['sources']:
        print(f"  [{src['rank']}] {src['paper_id']} (score: {src['score']:.3f})")


if __name__ == "__main__":
    test_generation()
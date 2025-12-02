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
        
        # CRITICAL: Check token count and truncate if needed
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        input_length = inputs['input_ids'].shape[1]
        
        # FLAN-T5 max length is 512 tokens
        max_input_tokens = 512
        
        if input_length > max_input_tokens:
            # Silently truncate context (no warning spam)
            prompt = self._build_prompt_with_truncation(question, context, max_input_tokens)
        
        # Generate
        try:
            output = self.pipe(
                prompt,
                max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
                do_sample=False,  # Deterministic generation
                num_beams=4,  # Beam search for better quality
                early_stopping=True,
                truncation=True  # Enable truncation as safety
            )
            
            # Extract generated text
            generated_text = output[0]["generated_text"]  # type: ignore[index]
            
            # Clean output
            answer = str(generated_text).strip()
            
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            answer = "Error: Context too long. Please reduce Top-K or try a shorter query."
        
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
    
    def _build_prompt_with_truncation(self, question: str, context: str, max_tokens: int) -> str:
        """
        Build prompt with truncated context to fit token limit
        
        Args:
            question: User question
            context: Full context (may be too long)
            max_tokens: Maximum tokens allowed (512 for FLAN-T5)
        
        Returns:
            Truncated prompt that fits within max_tokens
        """
        # Reserve tokens for prompt template and question
        template_overhead = 50  # "Context:", "Question:", "Answer:", etc.
        question_tokens = len(self.tokenizer.encode(question))
        available_for_context = max_tokens - template_overhead - question_tokens - 50  # Safety margin
        
        # Tokenize context and truncate
        context_tokens = self.tokenizer.encode(context, truncation=False)
        
        if len(context_tokens) > available_for_context:
            # Truncate context tokens
            truncated_tokens = context_tokens[:available_for_context]
            truncated_context = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            # Only print in verbose mode
            # print(f"[INFO] Context truncated from {len(context_tokens)} to {available_for_context} tokens")
        else:
            truncated_context = context
        
        # Build prompt with truncated context
        prompt = f"""Context:
{truncated_context}

Question: {question}

Based on the context above, provide a detailed and accurate answer.

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
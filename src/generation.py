"""
Generation module: Generate answers using retrieved context
"""
from typing import List, Dict, Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Pipeline,
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
            self.Pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1
            )
            print(f"[Generator] Model loaded on CPU")
    

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text
        
        Args:
            text: Input text
        
        Returns:
            Estimated token count
        """
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def _select_best_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        question: str,
        max_context_tokens: int = 400
    ) -> List[Dict[str, Any]]:
        """
        Intelligently select chunks that fit within token budget
        
        Prioritizes:
        1. Highest scoring chunks (most relevant)
        2. Diversity (different papers)
        3. Token budget compliance
        
        Args:
            chunks: List of retrieved chunks with scores
            question: User question (for token budget calculation)
            max_context_tokens: Maximum tokens for context
        
        Returns:
            Selected chunks that fit within budget
        """
        # Calculate tokens for question and template
        template_tokens = 50  # Approximate for "Context:", "Question:", etc.
        question_tokens = self._estimate_tokens(question)
        available_tokens = max_context_tokens - template_tokens - question_tokens
        
        # Sort by score (already sorted, but ensure)
        sorted_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)
        
        selected = []
        used_tokens = 0
        seen_papers = set()
        
        for chunk in sorted_chunks:
            chunk_tokens = self._estimate_tokens(chunk['text'])
            
            # Check if adding this chunk would exceed budget
            if used_tokens + chunk_tokens > available_tokens:
                # Try to fit a smaller portion of the chunk
                if len(selected) == 0:
                    # Must include at least one chunk
                    # Truncate this chunk to fit
                    selected.append(chunk)
                break
            
            # Add chunk
            selected.append(chunk)
            used_tokens += chunk_tokens
            seen_papers.add(chunk['meta']['paper_id'])
            
            # Stop if we have good coverage (3-4 chunks or 2+ papers)
            if len(selected) >= 3 and len(seen_papers) >= 2:
                break
        
        return selected

    def generate(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_length: int = MAX_GEN_LENGTH,
        include_citations: bool = True,
        smart_selection: bool = True
    ) -> Dict[str, Any]:
        """
        Generate answer from retrieved context
        
        Args:
            question: User question
            retrieved_chunks: List of retrieved chunks from Retriever
            max_length: Maximum generation length
            include_citations: Whether to include source citations
            smart_selection: Whether to use intelligent chunk selection
        
        Returns:
            Dict with 'answer', 'sources', 'prompt', 'chunks_used'
        """

        # Smart chunk selection to fit token budget
        if smart_selection:
            selected_chunks = self._select_best_chunks(
                retrieved_chunks, 
                question,
                max_context_tokens=400
            )
        else:
            selected_chunks = retrieved_chunks
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(selected_chunks, 1):
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
            # Emergency truncation (should rarely happen with smart selection)
            prompt = self._build_prompt_with_truncation(question, context, max_input_tokens)
        
        # Generate
        try:
            output = self.pipe(
                prompt,
                max_new_tokens=max_length,
                do_sample=False,  # Deterministic generation
                num_beams=4,  # Beam search for better quality
                early_stopping=True,
                truncation=True  # Enable truncation as safety
            )
            
            # Extract generated text safely
            if isinstance(output, list) and len(output) > 0:
                generated_text = output[0].get("generated_text", "")
            else:
                generated_text = ""
            # Clean output
            answer = str(generated_text).strip()
            
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            answer = "Error: Context too long. Please reduce Top-K or try a shorter query."
        
        return {
            "answer": answer,
            "sources": sources if include_citations else [],
            "prompt": prompt,
            "question": question,
            "chunks_used": len(selected_chunks),
            "chunks_available": len(retrieved_chunks)
        }
    
    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build prompt for the model
        
        Template optimized for FLAN-T5
        """
        prompt = f"""You are an expert on fairness and bias in large language models. 
Answer the question using SPECIFIC, CONCRETE information from the research papers below.
The answer should be DETAILED and TECHNICAL.
It should not be vague or general.

IMPORTANT: 
- Include EXACT NAMES (e.g., "demographic parity", "WinoBias dataset")
- Provide SPECIFIC FINDINGS from the papers
- DO NOT make vague or general statements
- Use terminology from the context

Research Papers:
{context}

Question: {question}

Detailed answer with specific information from the papers:
"""
        return prompt
    
    def _build_prompt_with_truncation(self, question: str, context: str, max_tokens: int) -> str:
        """
        Build prompt with truncated context to fit token limit
        
        This is a fallback for edge cases where smart selection isn't enough.
        
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
        retrieved_results: List[List[Dict[str, Any]]],
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
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
    
    def get_context_stats(self, question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about context size and token usage
        
        Useful for debugging and optimization
        
        Args:
            question: User question
            chunks: Retrieved chunks
        
        Returns:
            Dict with token statistics
        """
        selected = self._select_best_chunks(chunks, question)
        
        total_tokens = 0
        for chunk in selected:
            total_tokens += self._estimate_tokens(chunk['text'])
        
        return {
            "question_tokens": self._estimate_tokens(question),
            "context_tokens": total_tokens,
            "chunks_selected": len(selected),
            "chunks_available": len(chunks),
            "within_budget": total_tokens < 400,
            "estimated_total": self._estimate_tokens(question) + total_tokens + 50
        }


def test_generation() -> None:
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
    
    # Check context stats
    stats = generator.get_context_stats(question, chunks)
    print("Context Statistics:")
    print(f"  Question tokens: {stats['question_tokens']}")
    print(f"  Context tokens: {stats['context_tokens']}")
    print(f"  Chunks selected: {stats['chunks_selected']}/{stats['chunks_available']}")
    print(f"  Within budget: {stats['within_budget']}")
    print(f"  Estimated total: {stats['estimated_total']}/512")
    print()

    # Generate
    print("Generating answer...")
    result = generator.generate(question, chunks)
    
    print(f"\nAnswer:\n{result['answer']}\n")
    
    print(f"Chunks used: {result['chunks_used']}/{result['chunks_available']}")
    print("\nSources:")
    for src in result['sources']:
        print(f"  [{src['rank']}] {src['paper_id']} (score: {src['score']:.3f})")

if __name__ == "__main__":
    test_generation()
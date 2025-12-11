"""
Simple, robust direct generation for FLAN-T5-XL
No multi-pass complexity - just works!
"""
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import logging

from config import *

logging.getLogger("transformers.pipelines.base").setLevel(logging.ERROR)


class Generator:
    """Simple direct generator - no multi-pass complexity"""
    
    def __init__(
        self, 
        model_name: str = "google/flan-t5-xl",  # XL for better quality
        default_prompt_version: int = 5
    ):
        self.default_prompt_version = default_prompt_version
        self.model_name = model_name
        
        print(f"[Generator] Loading model: {model_name}")
        print(f"[Generator] Default prompt version: {default_prompt_version}")
        
        use_cuda = torch.cuda.is_available()
        
        if use_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[Generator] GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if use_cuda:
            print(f"[Generator] Loading model to GPU...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.pipe = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            
            allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"[Generator] ✓ Model loaded on GPU")
            print(f"[Generator] VRAM: {allocated:.2f} GB / {gpu_memory:.1f} GB used")
        else:
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
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough but fast)"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def _build_prompt_v1(self, question: str, context: str) -> str:
        """Prompt version 1 - Simple and direct"""
        return f"""Answer this question based on the research papers below.

Papers:
{context}

Question: {question}

Answer:"""
    
    def _build_prompt_v2(self, question: str, context: str) -> str:
        """Prompt version 2 - Structured (DEFAULT)"""
        return f"""Based on research papers about fairness and bias in LLMs, answer this question.

Use specific details from the papers: technical terms, method names, datasets, metrics, and concrete findings.

Research Papers:
{context}

Question: {question}

Detailed Answer:"""
    
    def _build_prompt_v3(self, question: str, context: str) -> str:
        """Prompt version 3 - Very detailed"""
        return f"""You are an expert on fairness and bias in large language models. Answer this question using the research papers below.

Requirements:
- Use technical terminology from the papers
- Mention specific methods, datasets, or metrics
- Provide 2-3 sentences with concrete details
- Be precise and accurate

Research Context:
{context}

Question: {question}

Expert Answer:"""
    
    def _build_prompt_v5(self, question: str, context: str) -> str:
        """Prompt version 5 - Constrained format"""
        return f"""Answer with specific information from these papers.

Papers:
{context}

Question: {question}

Requirements:
- Include at least 2 technical terms or specific names
- Mention concrete findings, numbers, or methods
- Write 2-4 sentences

Answer:"""
    
    def _build_prompt(self, question: str, context: str, version: int) -> str:
        """Build prompt based on version"""
        if version == 1:
            return self._build_prompt_v1(question, context)
        elif version == 3:
            return self._build_prompt_v3(question, context)
        elif version == 5:
            return self._build_prompt_v5(question, context)
        else:  # version 2 (default)
            return self._build_prompt_v2(question, context)
    
    def _is_good_chunk(self, text: str) -> bool:
        """
        Filter out low-quality chunks (tables, captions, citations)
        
        Returns True if chunk is substantive content
        """
        text_lower = text.lower().strip()
        
        # Too short
        if len(text.split()) < 15:
            return False
        
        # Mostly numbers/symbols (likely a table)
        non_alpha = sum(1 for c in text if not c.isalpha() and not c.isspace())
        if non_alpha / max(len(text), 1) > 0.3:
            return False
        
        # Starts with common table/caption markers
        bad_starts = [
            'table', 'figure', '[1]', '[2]', '[3]', '[4]', '[5]',
            'absolute values', 'note that', 'acknowledgement'
        ]
        for bad in bad_starts:
            if text_lower.startswith(bad):
                return False
        
        # Too many numbers (likely table data or citations)
        words = text.split()
        number_words = sum(1 for w in words if any(c.isdigit() for c in w))
        if number_words / len(words) > 0.3:
            return False
        
        return True
    
    def generate(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_length: int = MAX_GEN_LENGTH,
        include_citations: bool = True,
        prompt_version: int = None #type: ignore 
    ) -> Dict[str, Any]:
        """
        Simple direct generation from top chunks
        
        Args:
            question: User question
            retrieved_chunks: Retrieved chunks from retriever
            max_length: Max generation length
            include_citations: Include source information
            prompt_version: Prompt version (1, 2, 3, or 5)
        
        Returns:
            Dict with answer and metadata
        """
        if prompt_version is None:
            prompt_version = self.default_prompt_version
        
        print(f"[Generator] Generating answer from {len(retrieved_chunks)} chunks")
        
        # Sort by score (best first)
        sorted_chunks = sorted(
            retrieved_chunks, 
            key=lambda x: x.get('score', 0), 
            reverse=True
        )
        
        # Filter out junk chunks
        good_chunks = [c for c in sorted_chunks if self._is_good_chunk(c.get('text', ''))]
        
        #if len(good_chunks) < len(sorted_chunks):
        #    print(f"[Generator] Filtered out {len(sorted_chunks) - len(good_chunks)} low-quality chunks")
        
        if len(good_chunks) == 0:
            print(f"[Generator] Warning: No good chunks found, using top 3 anyway")
            good_chunks = sorted_chunks[:3]
        
        # Build context from top chunks that fit in budget
        context_parts = []
        sources = []
        total_tokens = 0
        max_context_tokens = 380  # Leave room for prompt template and question
        
        for i, chunk in enumerate(good_chunks, 1):
            # Use raw text (not expanded_text to avoid token explosion)
            text = chunk.get('text', '')
            if not text:
                continue
            
            chunk_tokens = self._estimate_tokens(text)
            
            # Check if adding this chunk would exceed budget
            if total_tokens + chunk_tokens > max_context_tokens:
                # If we haven't added any chunks yet, truncate this one
                if len(context_parts) == 0:
                    words = text.split()
                    # Take proportion of words that fit
                    proportion = max_context_tokens / chunk_tokens
                    truncated_words = words[:int(len(words) * proportion)]
                    text = " ".join(truncated_words)
                    chunk_tokens = self._estimate_tokens(text)
                    
                    context_parts.append(f"[{i}] {text}")
                    sources.append({
                        "rank": chunk.get("rank", i),
                        "paper_id": chunk["meta"]["paper_id"],
                        "score": chunk.get("score", 0.0)
                    })
                    total_tokens += chunk_tokens
                break
            
            # Add this chunk
            context_parts.append(f"[{i}] {text}")
            sources.append({
                "rank": chunk.get("rank", i),
                "paper_id": chunk["meta"]["paper_id"],
                "score": chunk.get("score", 0.0)
            })
            total_tokens += chunk_tokens
        
        # Handle edge case: no chunks fit
        if len(context_parts) == 0:
            return {
                "answer": "No relevant information found in the retrieved documents.",
                "sources": [],
                "method": "direct",
                "chunks_used": 0,
                "chunks_available": len(retrieved_chunks),
                "model": self.model_name
            }
        
        # Build full prompt
        context = "\n\n".join(context_parts)
        prompt = self._build_prompt(question, context, prompt_version)
        
        # Log token usage
        prompt_tokens = self._estimate_tokens(prompt)
        # print(f"[Generator] Using {len(context_parts)} chunks ({total_tokens} context tokens)")
        # print(f"[Generator] Total prompt: {prompt_tokens} tokens")
        
        if prompt_tokens > 500:
            print(f"[Generator] Warning: Prompt exceeds 512 tokens, will be truncated")
        
        # Generate answer
        output = self.pipe(
            prompt,
            max_new_tokens=max_length,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
            truncation=True,
            repetition_penalty=1.2,
            length_penalty=1.0
        )
        
        answer = output[0]["generated_text"].strip() if output else "" #type: ignore
        
        # Clean up answer (remove any leftover prompt artifacts)
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        
        print(f"[Generator] Generated {len(answer.split())} words")
        
        return {
            "answer": answer,
            "sources": sources if include_citations else [],
            "method": "direct",
            "chunks_used": len(context_parts),
            "chunks_available": len(retrieved_chunks),
            "context_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "model": self.model_name
        }


def test_generation():
    """Test simple direct generation"""
    from retrieval import Retriever
    
    print("\n" + "="*70)
    print("TESTING SIMPLE DIRECT GENERATION (FLAN-T5-XL)")
    print("="*70 + "\n")
    
    retriever = Retriever()
    generator = Generator()
    
    test_questions = [
        "What is Auto-Debias?",
        "How does Auto-Debias differ from previous debiasing approaches?",
        "How does the Auto-Debias approach search for biased prompts?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_questions)}")
        print(f"{'='*70}")
        print(f"\nQuestion: {question}\n")
        
        # Retrieve (no expansion - keep it simple)
        chunks = retriever.retrieve(question, top_k=7, include_neighbors=False)
        
        print(f"Retrieved {len(chunks)} chunks")
        print(f"Top papers: {', '.join(set(c['meta']['paper_id'] for c in chunks[:3]))}")
        print(f"Top score: {chunks[0]['score']:.3f}\n")
        
        # Test different prompt versions
        for pv in [2, 5]:
            print(f"{'-'*70}")
            print(f"PROMPT VERSION {pv}")
            print(f"{'-'*70}")
            
            result = generator.generate(question, chunks, prompt_version=pv)
            
            print(f"\nAnswer:\n{result['answer']}\n")
            print(f"Chunks used: {result['chunks_used']}/{result['chunks_available']}")
            print(f"Tokens: {result['context_tokens']} context + {result['prompt_tokens']} total")
            print()
        
        if i < len(test_questions):
            input("Press Enter for next question...")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)


def compare_prompts():
    """Compare all prompt versions on one question"""
    from retrieval import Retriever
    
    print("\n" + "="*70)
    print("COMPARING PROMPT VERSIONS")
    print("="*70 + "\n")
    
    retriever = Retriever()
    generator = Generator()
    
    question = "What is Auto-Debias?"
    
    print(f"Question: {question}\n")
    
    chunks = retriever.retrieve(question, top_k=7, include_neighbors=False)
    
    print(f"Retrieved: {len(chunks)} chunks\n")
    
    for version in [1, 2, 3, 5]:
        print(f"\n{'='*70}")
        print(f"PROMPT VERSION {version}")
        print(f"{'='*70}")
        
        result = generator.generate(question, chunks, prompt_version=version)
        
        print(f"\nAnswer:\n{result['answer']}\n")
        print(f"Length: {len(result['answer'].split())} words")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_prompts()
    else:
        test_generation()
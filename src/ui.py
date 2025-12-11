"""
Gradio UI for the RAG system - WITH MODEL CONSISTENCY CHECK
"""
from unittest import result
import gradio as gr
from typing import Tuple
import torch
import json

from retrieval import Retriever
from generation import Generator
from config import *


class RAGUI:
    """Interactive UI for the RAG system"""
    
    def __init__(self, embed_model: str = None): #type: ignore
        """
        Initialize RAG UI
        
        Args:
            embed_model: Embedding model to use (must match indexed model)
        """
        print("\n[UI] Initializing RAG system...")
        
        # Load version info to check which model was used for indexing
        if embed_model is None:
            embed_model = self._get_indexed_model()
        
        print(f"[UI] Using embedding model: {embed_model}")
        
        try:
            self.retriever = Retriever(embed_model=embed_model)
            self.generator = Generator(default_prompt_version=5)
            print("[UI] System ready!\n")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            print("\n⚠️  Please run data preparation first:")
            print(f"   python src/main.py --prepare --model {embed_model.split('/')[-1]}")
            raise
        except AssertionError as e:
            print(f"[ERROR] Dimension mismatch detected!")
            print(f"\n⚠️  The FAISS index was built with a different embedding model.")
            print(f"   Current model: {embed_model}")
            print(f"   Please rebuild the index with the correct model:")
            print(f"   python src/main.py --prepare --model <model_name> --force-rebuild")
            raise
    
    def _get_indexed_model(self) -> str:
        """
        Get the embedding model that was used for indexing
        
        Returns:
            Model name from config
        """
        version_file = DATA_DIR / "version.json"
        
        if version_file.exists():
            with open(version_file, 'r') as f:
                version_data = json.load(f)
                indexed_model = version_data.get('config', {}).get('embed_model')
                
                if indexed_model:
                    print(f"[UI] Detected indexed model: {indexed_model}")
                    return indexed_model
        
        # Default to minilm if can't detect
        print(f"[UI] Could not detect indexed model, defaulting to minilm")
        print(f"[UI] If you get errors, rebuild with: python src/main.py --prepare --model <model>")
        return EMBED_MODELS["minilm"]
    
    def query(
        self,
        question: str,
        top_k: int,
        show_context: bool,
    ) -> Tuple[str, str, str]:
        """
        Handle user query
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            show_context: Whether to show retrieved context
        
        Returns:
            answer, sources, context (if requested)
        """
        if not question.strip():
            return "Please enter a question.", "", ""
        
        try:
            retrieved = self.retriever.retrieve(
                question, 
                top_k=top_k,
                include_neighbors=False  
            )
            
            # Generate
            result = self.generator.generate(
                question, 
                retrieved,
            )
            
            # Format sources
            sources_text = "**Retrieved Sources:**\n\n"
            for src in result['sources']:
                sources_text += f"- **{src['paper_id']}** (score: {src['score']:.3f})\n"
            # Format context (if requested)
            context_text = ""
            if show_context:
                context_text = "**Retrieved Context:**\n\n"
                for i, chunk in enumerate(retrieved, 1):
                    context_text += f"**[{i}]** {chunk['meta']['paper_id']}\n"
                    context_text += f"{chunk['text']}\n\n"
                    context_text += "-" * 80 + "\n\n"
            
            return result['answer'], sources_text, context_text
        
        except AssertionError as e:
            error_msg = (
                "⚠️  **Dimension Mismatch Error**\n\n"
                "The embedding model doesn't match the indexed model.\n\n"
                "**Solution:**\n"
                "1. Close this UI\n"
                "2. Run: `python src/main.py --prepare --model <model_name> --force-rebuild`\n"
                "3. Restart the UI\n\n"
                "Available models: minilm, bge, bge-base, e5-small, e5-base"
            )
            return error_msg, "", ""
        
        except Exception as e:
            return f"Error: {str(e)}", "", ""
    
    def launch(self, share: bool = False):
        """Launch the Gradio interface"""
        
        # Get retriever stats
        stats = self.retriever.get_retrieval_stats()
        
        # Custom CSS
        custom_css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .gr-button-primary {
            background: linear-gradient(90deg, #4CAF50, #45a049) !important;
        }
        """
        
        with gr.Blocks(css=custom_css, title="RAG System: Fairness & Bias in LLMs") as demo:
            gr.Markdown(
                f"""
                # 🤖 Retrieval-Augmented Generation System
                ## Fairness & Bias in Large Language Models
                
                Ask questions about fairness and bias research in LLMs. 
                The system retrieves relevant passages from academic papers and generates grounded answers.
                
                **Current Model:** `{stats['embedding_model']}`  
                **Indexed Chunks:** {stats['total_chunks']}  
                **Device:** {stats['device'].upper()}
                """
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    question_box = gr.Textbox(
                        label="Your Research Question",
                        placeholder="e.g., What metrics are used to measure bias in LLMs?",
                        lines=3
                    )
                    
                    with gr.Row():
                        top_k_slider = gr.Slider(
                            minimum=5,
                            maximum=20,
                            value=15,
                            step=1,
                            label="Number of chunks to retrieve (Top-K)"
                        )
                        
                        show_context_check = gr.Checkbox(
                            label="Show retrieved context",
                            value=False
                        )
                    
                    submit_btn = gr.Button("Get Answer", variant="primary", size="lg")
                    
                    gr.Markdown("### Example Questions:")
                    gr.Examples(
                        examples=[
                            ["How does Auto-Debias differ from previous debiasing approaches?"],
                            ["What is bias?"],
                            ["What are the three primary sources contributing to bias in Large Language Models?"],
                            ["What are the main types of bias in LLMs?"],
                            ["What techniques exist for debiasing language models?"]
                        ],
                        inputs=question_box
                    )
                
                with gr.Column(scale=3):
                    answer_box = gr.Textbox(
                        label="Generated Answer",
                        lines=8,
                        interactive=False
                    )
                    
                    sources_box = gr.Markdown(
                        label="Sources"
                    )
                    
                    with gr.Accordion("Retrieved Context", open=False):
                        context_box = gr.Markdown()
            
            # Stats footer
            gr.Markdown(
                f"""
                ---
                **System Statistics:**
                - Total chunks indexed: {stats['total_chunks']}
                - Total papers: {stats.get('total_papers', 'N/A')}
                - Embedding model: {stats['embedding_model']}
                - Embedding dimension: {stats['embedding_dim']}
                - Generation model: {GEN_MODELS['flan-t5-large']}
                - Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}
                - E5 Model: {'Yes ✓' if stats.get('is_e5_model', False) else 'No'}
                """
            )
            
            # Event handler
            submit_btn.click(
                fn=self.query,
                inputs=[question_box, top_k_slider, show_context_check],
                outputs=[answer_box, sources_box, context_box]
            )
        
        # Launch
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=share,
            show_error=True
        )


def launch_ui(share: bool = False, embed_model: str = None): #type: ignore
    """
    Launch the UI
    
    Args:
        share: Create public share link
        embed_model: Embedding model to use (auto-detected if None)
    """
    try:
        ui = RAGUI(embed_model=embed_model)
        ui.launch(share=share)
    except Exception as e:
        print(f"\n❌ Failed to launch UI: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've prepared the dataset:")
        print("   python src/main.py --prepare --model <model_name>")
        print("\n2. Available models: minilm, bge, bge-base, e5-small, e5-base")
        print("\n3. If you get dimension mismatch, rebuild with --force-rebuild")


if __name__ == "__main__":
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print('\n\n[UI] Shutting down gracefully...')
        print('[UI] You can close this window now.')
        sys.exit(0)
    
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\n" + "="*70)
    print("RAG SYSTEM UI")
    print("="*70)
    print("\n💡 TIP: Press Ctrl+C to stop the server\n")
    
    try:
        launch_ui()
    except KeyboardInterrupt:
        print('\n\n[UI] Shutting down gracefully...')
        print('[UI] Goodbye!')
        sys.exit(0)
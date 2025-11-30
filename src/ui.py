"""
Gradio UI for the RAG system
"""
import gradio as gr
from typing import Tuple
import torch

from retrieval import Retriever
from generation import Generator
from config import *


class RAGUI:
    """Interactive UI for the RAG system"""
    
    def __init__(self):
        print("\n[UI] Initializing RAG system...")
        self.retriever = Retriever()
        self.generator = Generator()
        print("[UI] System ready!\n")
    
    def query(
        self,
        question: str,
        top_k: int,
        show_context: bool
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
        
        # Retrieve
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        
        # Generate
        result = self.generator.generate(question, retrieved)
        
        # Format sources
        sources_text = "**Retrieved Sources:**\n\n"
        for src in result['sources']:
            sources_text += f"- **[{src['rank']}]** {src['paper_id']} (relevance: {src['score']:.3f})\n"
        
        # Format context (if requested)
        context_text = ""
        if show_context:
            context_text = "**Retrieved Context:**\n\n"
            for i, chunk in enumerate(retrieved, 1):
                context_text += f"**[{i}]** {chunk['meta']['paper_id']}\n"
                context_text += f"{chunk['text']}\n\n"
                context_text += "-" * 80 + "\n\n"
        
        return result['answer'], sources_text, context_text
    
    def launch(self, share: bool = False):
        """Launch the Gradio interface"""
        
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
                """
                # 🤖 Retrieval-Augmented Generation System
                ## Fairness & Bias in Large Language Models
                
                Ask questions about fairness and bias research in LLMs. 
                The system retrieves relevant passages from academic papers and generates grounded answers.
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
                            minimum=1,
                            maximum=6,
                            value=3,
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
                            ["What are common metrics for measuring bias in large language models?"],
                            ["How can we mitigate bias in language models?"],
                            ["What datasets are used for fairness evaluation in NLP?"],
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
                - Total chunks indexed: {self.retriever.get_retrieval_stats()['total_chunks']}
                - Embedding model: {EMBED_MODELS['minilm']}
                - Generation model: {GEN_MODELS['flan-t5-large']}
                - Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}
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
            server_name="127.0.0.1",  # Use 127.0.0.1 instead of 0.0.0.0 for Windows
            server_port=7860,
            share=share,
            show_error=True
        )


def launch_ui(share: bool = False):
    """Launch the UI"""
    ui = RAGUI()
    ui.launch(share=share)


if __name__ == "__main__":
    launch_ui()
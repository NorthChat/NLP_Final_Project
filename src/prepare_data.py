"""
Data preparation: PDF extraction, chunking, embedding, and indexing
"""
import json
import re
from pathlib import Path
from typing import List, Dict
import numpy as np
from numpy.typing import NDArray
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import torch

from config import *


def clean_text(text: str) -> str:
    """Clean extracted text from PDFs"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers and common artifacts
    text = re.sub(r'\n\d+\n', '\n', text)
    # Remove email artifacts
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()


def pdf_to_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF with cleaning"""
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            text = page.get_text("text")  # type: ignore[attr-defined]
            pages.append(text)
        full_text = "\n".join(pages)
        return clean_text(full_text)
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {pdf_path}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 450, overlap: int = 100) -> List[str]:
    """
    Chunk text using sliding window on words.
    
    Args:
        text: Input text
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    i = 0
    
    while i < len(words):
        # Get chunk
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)
        
        # Move window
        i += chunk_size - overlap
        
        # Avoid infinite loop on small texts
        if i >= len(words):
            break
    
    return chunks


def sentence_level_chunking(text: str, max_sentences: int = 5, overlap_sentences: int = 1) -> List[str]:
    """
    Alternative chunking: sentence-based (for A grade improvement)
    
    Args:
        text: Input text
        max_sentences: Maximum sentences per chunk
        overlap_sentences: Number of overlapping sentences
    
    Returns:
        List of text chunks
    """
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    i = 0
    
    while i < len(sentences):
        chunk_sents = sentences[i:i + max_sentences]
        chunk = ". ".join(chunk_sents) + "."
        chunks.append(chunk)
        i += max_sentences - overlap_sentences
    
    return chunks


def prepare_dataset(
    embed_model: str = EMBED_MODELS["minilm"],
    chunk_size: int = 450,
    use_sentence_chunking: bool = False
):
    """
    Main data preparation pipeline
    
    Steps:
    1. Extract text from PDFs
    2. Chunk text
    3. Generate embeddings
    4. Build FAISS index
    
    Args:
        embed_model: Name of embedding model
        chunk_size: Size of chunks (if word-based)
        use_sentence_chunking: Use sentence-based chunking instead
    """
    print(f"\n{'='*60}")
    print(f"PREPARING DATASET")
    print(f"{'='*60}\n")
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Find PDFs
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    print(f"[INFO] Found {len(pdf_paths)} PDFs\n")
    
    if len(pdf_paths) == 0:
        print("[ERROR] No PDFs found! Please add papers to papers_pdf/")
        return
    
    # ===========================
    # STEP 1: Extract and Chunk
    # ===========================
    all_chunks = []
    all_metas = []
    
    for pdf in tqdm(pdf_paths, desc="Processing PDFs"):
        paper_id = pdf.stem
        
        # Extract text
        text = pdf_to_text(pdf)
        
        if len(text.strip()) < 100:
            print(f"\n[WARNING] Skipping {pdf.name} - insufficient text extracted")
            continue
        
        # Chunk text
        if use_sentence_chunking:
            chunks = sentence_level_chunking(text)
        else:
            chunks = chunk_text(text, chunk_size, CHUNK_OVERLAP)
        
        # Store chunks with metadata
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{paper_id}_chunk{idx}"
            all_chunks.append({
                "paper_id": paper_id,
                "chunk_id": chunk_id,
                "text": chunk
            })
            all_metas.append({
                "paper_id": paper_id,
                "chunk_id": chunk_id
            })
    
    print(f"\n[INFO] Total chunks created: {len(all_chunks)}")
    
    # Save chunks
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for obj in all_chunks:
            f.write(json.dumps(obj) + "\n")
    
    with open(METAS_FILE, "w", encoding="utf-8") as f:
        for m in all_metas:
            f.write(json.dumps(m) + "\n")
    
    print(f"[INFO] Saved chunks to {CHUNKS_FILE}")
    
    # ===========================
    # STEP 2: Generate Embeddings
    # ===========================
    print(f"\n[INFO] Loading embedding model: {embed_model}")
    embedder = SentenceTransformer(embed_model, device=device)
    
    raw_chunks = [c["text"] for c in all_chunks]
    
    print(f"[INFO] Generating embeddings for {len(raw_chunks)} chunks...")
    embeddings: NDArray[np.float32] = embedder.encode(
        raw_chunks,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Important for cosine similarity
    ).astype("float32")  # type: ignore
    
    np.save(EMBS_FILE, embeddings)
    print(f"[INFO] Saved embeddings to {EMBS_FILE}")
    print(f"[INFO] Embedding shape: {embeddings.shape}")
    
    # ===========================
    # STEP 3: Build FAISS Index
    # ===========================
    print(f"\n[INFO] Building FAISS index...")
    
    dim = embeddings.shape[1]
    
    # Use Inner Product for normalized embeddings (equivalent to cosine similarity)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)  # type: ignore[arg-type]
    
    faiss.write_index(index, str(INDEX_FILE))
    print(f"[INFO] Saved FAISS index to {INDEX_FILE}")
    print(f"[INFO] Index contains {index.ntotal} vectors")
    
    print(f"\n{'='*60}")
    print("PREPARATION COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Default preparation
    prepare_dataset()
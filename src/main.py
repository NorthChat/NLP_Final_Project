"""
End-to-end Retrieval-Augmented Generation System (RAG)
for summarizing and answering questions about
Fairness & Bias research in LLMs.

Run these commands:

(1) Prepare dataset (extract → chunk → embed → index)
    python src/main.py --prepare

(2) Launch the UI
    python src/main.py
"""

import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from numpy.typing import NDArray
import numpy as np


# PDF extraction
import fitz  # PyMuPDF

# Embeddings
from sentence_transformers import SentenceTransformer

# FAISS index
import faiss

# Generation
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

# UI
import gradio as gr

# -----------------------
# CONFIG
# -----------------------

ROOT = Path(__file__).resolve().parent.parent

PDF_DIR = ROOT / "papers_pdf"
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"

CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
METAS_FILE = DATA_DIR / "metas.jsonl"
EMBS_FILE = DATA_DIR / "embeddings.npy"
INDEX_FILE = INDEX_DIR / "faiss_index.idx"

EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-large"

CHUNK_SIZE = 450
CHUNK_OVERLAP = 100
TOP_K = 5
BATCH = 32


# -----------------------
# UTILITIES
# -----------------------

def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def pdf_to_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc] # type: ignore[attr-defined]
    return "\n".join(pages)


def chunk_text(text: str, chunk_size=450, overlap=100):
    """Chunk long text using word-based sliding window."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# -----------------------
# PREPARE DATA
# -----------------------

def prepare():
    DATA_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)

    all_chunks = []
    all_metas = []

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))

    print(f"\n[INFO] Found {len(pdf_paths)} PDFs.\n")

    # -----------------------
    # 1. PDF → Text → Chunks
    # -----------------------
    for pdf in pdf_paths:
        paper_id = pdf.stem
        print(f"[+] Processing {pdf.name}")

        text = pdf_to_text(pdf)

        if len(text.strip()) < 20:
            print(f"[WARNING] PDF might be scanned or empty: {pdf}")
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

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

    # Save chunks
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for obj in all_chunks:
            f.write(json.dumps(obj) + "\n")

    with open(METAS_FILE, "w", encoding="utf-8") as f:
        for m in all_metas:
            f.write(json.dumps(m) + "\n")

    print(f"[INFO] Saved {len(all_chunks)} chunks.")

    # -----------------------
    # 2. Embeddings
    # -----------------------

    device = "cuda" if cuda_available() else "cpu"
    print(f"[INFO] Embedding on device: {device}")

    embedder = SentenceTransformer(EMBED_MODEL, device=device)

    raw_chunks = [c["text"] for c in all_chunks]

    print("\n[INFO] Generating embeddings...")
    embeddings: NDArray[np.float32] = embedder.encode(
        raw_chunks,
        batch_size=BATCH,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")  # type: ignore

    np.save(EMBS_FILE, embeddings)

    # -----------------------
    # 3. FAISS Index
    # -----------------------
    print("\n[INFO] Building FAISS index...")

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)  # type: ignore

    faiss.write_index(index, str(INDEX_FILE))

    print(f"\n[✓] PREPARATION DONE.")
    print(f"    Chunks:     {len(all_chunks)}")
    print(f"    Embeddings: {EMBS_FILE}")
    print(f"    Index:      {INDEX_FILE}\n")


# -----------------------
# RETRIEVAL
# -----------------------

def load_index_and_data():
    index = faiss.read_index(str(INDEX_FILE))
    texts = [json.loads(line)["text"] for line in open(CHUNKS_FILE, "r", encoding="utf-8")]
    metas = [json.loads(line) for line in open(METAS_FILE, "r", encoding="utf-8")]
    return index, metas, texts


def retrieve(query, index, embedder, metas, texts, top_k=TOP_K):
    q_emb = embedder.encode(query, convert_to_numpy=True).astype("float32").reshape(1, -1)
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": float(score),
            "meta": metas[idx],
            "text": texts[idx]
        })
    return results


# -----------------------
# GENERATION
# -----------------------

def load_generator():
    device = 0 if cuda_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    return pipe


def generate_answer(generator, retrieved, question):
    context = "\n\n".join([r["text"] for r in retrieved])
    prompt = f"""
Answer ONLY using the context below.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    out = generator(prompt, max_length=300, do_sample=False)[0]["generated_text"]
    return out


# -----------------------
# UI
# -----------------------

def launch_ui():
    index, metas, texts = load_index_and_data()
    embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if cuda_available() else "cpu")
    generator = load_generator()

    def qa_fn(query, top_k):
        retrieved = retrieve(query, index, embedder, metas, texts, top_k)
        answer = generate_answer(generator, retrieved, query)
        cites = "\n".join([f"{r['meta']['paper_id']} (score {r['score']:.3f})" for r in retrieved])
        return answer, cites

    ui = gr.Interface(
        fn=qa_fn,
        inputs=[
            gr.Textbox(label="Enter your research question"),
            gr.Slider(1, 10, value=5, step=1, label="Top-K")
        ],
        outputs=[
            gr.Textbox(label="Answer"),
            gr.Textbox(label="Retrieved Chunks")
        ],
        title="Fairness & Bias in LLMs — RAG System",
        description="Ask any research question about fairness or bias in LLMs."
    )

    ui.launch(server_name="0.0.0.0", server_port=7860)


# -----------------------
# MAIN
# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        prepare()
    else:
        launch_ui()

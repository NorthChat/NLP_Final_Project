"""
Configuration file for the RAG system
"""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "papers_pdf"
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
EVAL_DIR = ROOT / "evaluation"

# Data files
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
METAS_FILE = DATA_DIR / "metas.jsonl"
EMBS_FILE = DATA_DIR / "embeddings.npy"
INDEX_FILE = INDEX_DIR / "faiss_index.idx"
QA_PAIRS_FILE = ROOT / "qa_pairs.json"

# Model configurations
EMBED_MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5"
}

GEN_MODELS = {
    "flan-t5-large": "google/flan-t5-large",
    "flan-t5-xl": "google/flan-t5-xl"
}

# Hyperparameters
CHUNK_SIZES = [250, 450]  # For ablation study
CHUNK_OVERLAP = 100
TOP_K = 5
BATCH_SIZE = 32
MAX_GEN_LENGTH = 300

# Evaluation
EVAL_METRICS = ["precision", "recall", "mrr", "rouge", "human"]

# Device settings
DEVICE = "cuda"  # Your RTX 4060 will be utilized
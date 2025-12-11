"""
Data preparation: PDF extraction, chunking, embedding, and indexing
WITH E5 MODEL SUPPORT
"""
import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Tuple
import numpy as np
from numpy.typing import NDArray
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import torch
import unicodedata
import ftfy
import spacy
from datetime import datetime

print("CUDA available:", torch.cuda.is_available())
spacy.prefer_gpu()  # type: ignore
nlp = spacy.load("en_core_web_trf")

from config import *


# ===========================
# CONFIGURATION
# ===========================

MIN_WORDS = 20
MIN_CHARS = 100
MAX_CHUNK_RATIO = 0.95


# ===========================
# E5 MODEL DETECTION AND PREPROCESSING
# ===========================

def is_e5_model(model_name: str) -> bool:
    """Check if the model is an E5 model"""
    return "e5-" in model_name.lower() or "e5_" in model_name.lower()


def prepare_texts_for_embedding(texts: List[str], embed_model: str) -> List[str]:
    """
    Prepare texts for embedding based on model type
    
    E5 models require:
    - "passage: " prefix for documents during indexing
    - "query: " prefix for queries during retrieval
    
    Args:
        texts: List of text chunks
        embed_model: Name of embedding model
    
    Returns:
        Processed texts ready for embedding
    """
    if is_e5_model(embed_model):
        print(f"[INFO] E5 model detected: Adding 'passage: ' prefix to documents")
        return [f"passage: {text}" for text in texts]
    return texts


# ===========================
# TEXT CLEANING
# ===========================

def clean_text(text: str) -> str:
    """Clean extracted text from PDFs"""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200b", "")
    text = text.replace("\u00ad", "")
    text = text.replace("\ufeff", "")
    text = text.replace("\xa0", " ")
    text = re.sub(r'[\uf000-\uf8ff]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\d+\n', '\n', text)
    text = text.replace("-\n", "")
    text = text.replace("- ", "")
    text = text.replace("\n", " ")
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r"/abs/\d{4}\.\d{5}", "", text)
    text = re.sub(r"arXiv:\d{4}\.\d{5}", "", text)
    text = re.sub(r"doi:\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r'\bhttps?://\S+\b', '', text)
    text = re.sub(r'\bwww\.\S+\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = ftfy.fix_text(text)
    return text.strip()


def pdf_to_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF with cleaning"""
    try:
        doc = fitz.open(pdf_path)  # type: ignore[attr-defined]
        pages = []
        for page in doc:  # type: ignore[attr-defined]
            text = page.get_text("text")  # type: ignore[attr-defined]
            pages.append(text)
        full_text = "\n".join(pages)
        return clean_text(full_text)
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {pdf_path}: {e}")
        return ""


# ===========================
# CHUNKING STRATEGIES
# ===========================

def chunk_text(text: str, chunk_size: int = 250, overlap: int = 100) -> List[str]:
    """Chunk text using sliding window on words"""
    words = text.split()
    chunks = []
    i = 0
    
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        
        if chunk.strip():
            chunks.append(chunk)
        
        i += chunk_size - overlap
        
        if i >= len(words):
            break
    
    return chunks


def sentence_level_chunking(text: str, max_sentences: int = 5, overlap_sentences: int = 1) -> List[str]:
    """Alternative chunking: sentence-based"""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sents = sentences[i:i + max_sentences]
        chunk = " ".join(chunk_sents)
        chunks.append(chunk)
        i += max_sentences - overlap_sentences
    
    return chunks


# ===========================
# CHUNK FILTERING & DEDUPLICATION
# ===========================

def compute_chunk_hash(text: str) -> str:
    """Compute hash for chunk deduplication"""
    normalized = text.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def filter_chunks(chunks: List[str]) -> List[str]:
    """Remove duplicates and very short chunks"""
    seen_hashes: Set[str] = set()
    filtered = []
    
    for chunk in chunks:
        text = chunk.strip()
        
        if len(text.split()) < MIN_WORDS:
            continue
        
        if len(text) < MIN_CHARS:
            continue
        
        chunk_hash = compute_chunk_hash(text)
        if chunk_hash in seen_hashes:
            continue
        
        seen_hashes.add(chunk_hash)
        filtered.append(text)
    
    return filtered


# ===========================
# VERSIONING SYSTEM
# ===========================

class DatasetVersion:
    """Manages dataset versioning and incremental updates"""
    
    def __init__(self, version_file: Path = DATA_DIR / "version.json"):
        self.version_file = version_file
        self.version_data = self._load_version()
    
    def _load_version(self) -> Dict:
        """Load existing version data"""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {
            "version": 0,
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "processed_files": {},
            "total_chunks": 0,
            "config": {}
        }
    
    def _save_version(self):
        """Save version data"""
        self.version_data["last_updated"] = datetime.now().isoformat()
        with open(self.version_file, 'w') as f:
            json.dump(self.version_data, f, indent=2)
    
    def get_file_hash(self, pdf_path: Path) -> str:
        """Compute hash of PDF file"""
        with open(pdf_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def is_file_processed(self, pdf_path: Path) -> bool:
        """Check if file has been processed with same hash"""
        filename = pdf_path.name
        if filename not in self.version_data["processed_files"]:
            return False
        
        current_hash = self.get_file_hash(pdf_path)
        stored_hash = self.version_data["processed_files"][filename].get("hash")
        
        return current_hash == stored_hash
    
    def mark_file_processed(self, pdf_path: Path, chunk_count: int):
        """Mark file as processed"""
        filename = pdf_path.name
        self.version_data["processed_files"][filename] = {
            "hash": self.get_file_hash(pdf_path),
            "chunk_count": chunk_count,
            "timestamp": datetime.now().isoformat()
        }
    
    def increment_version(self):
        """Increment version number"""
        self.version_data["version"] += 1
        self._save_version()
    
    def get_processed_files(self) -> Set[str]:
        """Get set of processed filenames"""
        return set(self.version_data["processed_files"].keys())
    
    def remove_file(self, filename: str):
        """Remove file from processed list"""
        if filename in self.version_data["processed_files"]:
            del self.version_data["processed_files"][filename]
    
    def update_config(self, config: Dict):
        """Update configuration"""
        self.version_data["config"] = config
        self._save_version()


# ===========================
# INCREMENTAL INDEXING
# ===========================

def load_existing_data() -> Tuple[List[Dict], List[Dict], NDArray[np.float32] | None]:
    """Load existing chunks, metadata, and embeddings"""
    chunks: List[Dict] = []
    metas: List[Dict] = []
    embeddings: NDArray[np.float32] | None = None
    
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            chunks = [json.loads(line) for line in f]
    
    if METAS_FILE.exists():
        with open(METAS_FILE, 'r', encoding='utf-8') as f:
            metas = [json.loads(line) for line in f]
    
    if EMBS_FILE.exists():
        embeddings = np.load(EMBS_FILE).astype(np.float32)
    
    return chunks, metas, embeddings


def prepare_dataset(
    embed_model: str = EMBED_MODELS["e5-base"],
    chunk_size: int = 250,
    use_sentence_chunking: bool = True,
    force_rebuild: bool = False,
    incremental: bool = True
):
    """
    Main data preparation pipeline with E5 support
    
    Args:
        embed_model: Name of embedding model (can be E5)
        chunk_size: Size of chunks (if word-based)
        use_sentence_chunking: Use sentence-based chunking instead
        force_rebuild: Force complete rebuild
        incremental: Use incremental indexing
    """
    print(f"\n{'='*60}")
    print(f"PREPARING DATASET {'(INCREMENTAL)' if incremental and not force_rebuild else '(FULL REBUILD)'}")
    print(f"{'='*60}\n")
    
    # Check if E5 model
    is_e5 = is_e5_model(embed_model)
    if is_e5:
        print(f"⚠️  E5 MODEL DETECTED: {embed_model}")
        print(f"    Documents will be prefixed with 'passage: '")
        print(f"    Queries must be prefixed with 'query: ' (handled by Retriever)")
        print()
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)
    
    # Initialize versioning
    version_manager = DatasetVersion()
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Find PDFs
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    print(f"[INFO] Found {len(pdf_paths)} PDFs")
    
    if len(pdf_paths) == 0:
        print("[ERROR] No PDFs found! Please add papers to papers_pdf/")
        return
    
    # ===========================
    # INCREMENTAL vs FULL REBUILD
    # ===========================
    
    if incremental and not force_rebuild:
        print("\n[INFO] Checking for changes...")
        existing_chunks, existing_metas, existing_embeddings = load_existing_data()
        print(f"[INFO] Loaded {len(existing_chunks)} existing chunks")
        
        current_files = {p.name for p in pdf_paths}
        processed_files = version_manager.get_processed_files()
        
        removed_files = processed_files - current_files
        for removed in removed_files:
            print(f"[INFO] Removed file detected: {removed}")
            version_manager.remove_file(removed)
            existing_chunks = [c for c in existing_chunks if c["paper_id"] != Path(removed).stem]
            existing_metas = [m for m in existing_metas if m["paper_id"] != Path(removed).stem]
        
        files_to_process = []
        for pdf_path in pdf_paths:
            if not version_manager.is_file_processed(pdf_path):
                files_to_process.append(pdf_path)
                if pdf_path.name in processed_files:
                    print(f"[INFO] Modified file detected: {pdf_path.name}")
                else:
                    print(f"[INFO] New file detected: {pdf_path.name}")
        
        if not files_to_process and not removed_files:
            print("\n[INFO] No changes detected. Dataset is up-to-date!")
            print(f"[INFO] Current version: {version_manager.version_data['version']}")
            print(f"[INFO] Total chunks: {len(existing_chunks)}")
            return
        
        print(f"\n[INFO] Processing {len(files_to_process)} new/modified files...")
        all_chunks = existing_chunks
        all_metas = existing_metas
        
    else:
        print("\n[INFO] Performing full rebuild...")
        files_to_process = pdf_paths
        all_chunks = []
        all_metas = []
        existing_embeddings: NDArray[np.float32] | None = None
    
    # ===========================
    # STEP 1: Extract and Chunk New Files
    # ===========================
    
    new_chunks = []
    new_metas = []
    
    for pdf in tqdm(files_to_process, desc="Processing PDFs"):
        paper_id = pdf.stem
        
        if incremental:
            all_chunks = [c for c in all_chunks if c["paper_id"] != paper_id]
            all_metas = [m for m in all_metas if m["paper_id"] != paper_id]
        
        text = pdf_to_text(pdf)
        
        if len(text.strip()) < 100:
            print(f"\n[WARNING] Skipping {pdf.name} - insufficient text extracted")
            continue
        
        if use_sentence_chunking:
            chunks = sentence_level_chunking(text)
        else:
            chunks = chunk_text(text, chunk_size, CHUNK_OVERLAP)
        
        chunks = filter_chunks(chunks)
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{paper_id}_chunk{idx}"
            new_chunks.append({
                "paper_id": paper_id,
                "chunk_id": chunk_id,
                "text": chunk
            })
            new_metas.append({
                "paper_id": paper_id,
                "chunk_id": chunk_id
            })
        
        version_manager.mark_file_processed(pdf, len(chunks))
    
    all_chunks.extend(new_chunks)
    all_metas.extend(new_metas)
    
    print(f"\n[INFO] Total chunks: {len(all_chunks)} ({len(new_chunks)} new)")
    
    # Global deduplication
    print("[INFO] Performing global deduplication...")
    chunk_texts = [c["text"] for c in all_chunks]
    filtered_indices = []
    seen_hashes = set()
    
    for i, text in enumerate(chunk_texts):
        chunk_hash = compute_chunk_hash(text)
        if chunk_hash not in seen_hashes:
            seen_hashes.add(chunk_hash)
            filtered_indices.append(i)
    
    all_chunks = [all_chunks[i] for i in filtered_indices]
    all_metas = [all_metas[i] for i in filtered_indices]
    
    duplicates_removed = len(chunk_texts) - len(all_chunks)
    if duplicates_removed > 0:
        print(f"[INFO] Removed {duplicates_removed} duplicate chunks")
    print(f"[INFO] Final chunk count: {len(all_chunks)}")
    
    # Save chunks
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for obj in all_chunks:
            f.write(json.dumps(obj) + "\n")
    
    with open(METAS_FILE, "w", encoding="utf-8") as f:
        for m in all_metas:
            f.write(json.dumps(m) + "\n")
    
    print(f"[INFO] Saved chunks to {CHUNKS_FILE}")
    
    # ===========================
    # STEP 2: Generate Embeddings WITH E5 SUPPORT
    # ===========================
    
    print(f"\n[INFO] Loading embedding model: {embed_model}")
    embedder = SentenceTransformer(embed_model, device=device)
    
    if not incremental or existing_embeddings is None:
        raw_chunks = [c["text"] for c in all_chunks]
        
        # 🔥 CRITICAL: Prepare texts for E5 models
        prepared_texts = prepare_texts_for_embedding(raw_chunks, embed_model)
        
        if is_e5:
            print(f"[INFO] ✓ E5 preprocessing applied: Added 'passage: ' to {len(prepared_texts)} chunks")
            print(f"[INFO] Sample: '{prepared_texts[0][:80]}...'")
        
        print(f"[INFO] Generating embeddings for {len(prepared_texts)} chunks...")
        
        embeddings: NDArray[np.float32] = embedder.encode(
            prepared_texts,  # 🔥 Use prepared texts, not raw_chunks!
            batch_size=64 if device == "cuda" else 32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")  # type: ignore
    
    np.save(EMBS_FILE, embeddings)
    print(f"[INFO] Saved embeddings to {EMBS_FILE}")
    print(f"[INFO] Embedding shape: {embeddings.shape}")
    
    # ===========================
    # STEP 3: Build FAISS Index
    # ===========================
    
    print(f"\n[INFO] Building FAISS index...")
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
    index.add(embeddings)  # type: ignore[arg-type]
    
    faiss.write_index(index, str(INDEX_FILE))  # type: ignore[attr-defined]
    print(f"[INFO] Saved FAISS index to {INDEX_FILE}")
    print(f"[INFO] Index contains {index.ntotal} vectors")  # type: ignore[attr-defined]
    
    # ===========================
    # Update Version
    # ===========================
    
    version_manager.version_data["total_chunks"] = len(all_chunks)
    version_manager.update_config({
        "embed_model": embed_model,
        "is_e5_model": is_e5,
        "chunk_size": chunk_size,
        "use_sentence_chunking": use_sentence_chunking,
        "min_words": MIN_WORDS,
        "min_chars": MIN_CHARS
    })
    version_manager.increment_version()
    
    print(f"\n{'='*60}")
    print("PREPARATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Version: {version_manager.version_data['version']}")
    print(f"Model: {embed_model}")
    if is_e5:
        print(f"E5 Model: YES ✓")
    print(f"Files processed: {len(version_manager.version_data['processed_files'])}")
    print(f"Total chunks: {len(all_chunks)}")
    print()


if __name__ == "__main__":
    prepare_dataset()
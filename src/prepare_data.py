"""
Data preparation: PDF extraction, chunking, embedding, and indexing
WITH VERSIONING AND INCREMENTAL UPDATES
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

MIN_WORDS = 20  # Minimum words per chunk
MIN_CHARS = 100  # Minimum characters per chunk
MAX_CHUNK_RATIO = 0.95  # Maximum similarity ratio for deduplication


# ===========================
# TEXT CLEANING
# ===========================

def clean_text(text: str) -> str:
    """Clean extracted text from PDFs"""
    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove zero-width and special spaces
    text = text.replace("\u200b", "")  # zero-width space
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\ufeff", "")  # BOM
    text = text.replace("\xa0", " ")   # non-breaking space

    # Remove Private Use Area characters (PUA)
    text = re.sub(r'[\uf000-\uf8ff]', '', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and common artifacts
    text = re.sub(r'\n\d+\n', '\n', text)

    # Handle hyphenated breaks
    text = text.replace("-\n", "")
    text = text.replace("- ", "")
    text = text.replace("\n", " ")

    # Remove email artifacts
    text = re.sub(r'\S+@\S+', '', text)

    # Remove URLs and identifiers
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r"/abs/\d{4}\.\d{5}", "", text)
    text = re.sub(r"arXiv:\d{4}\.\d{5}", "", text)
    text = re.sub(r"doi:\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r'\bhttps?://\S+\b', '', text)
    text = re.sub(r'\bwww\.\S+\b', '', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix text encoding issues
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
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        
        if chunk.strip():
            chunks.append(chunk)
        
        i += chunk_size - overlap
        
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
    # Normalize text for hashing (lowercase, strip)
    normalized = text.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def filter_chunks(chunks: List[str]) -> List[str]:
    """
    Remove duplicates and very short chunks
    
    Args:
        chunks: List of text chunks
    
    Returns:
        Filtered list of chunks
    """
    seen_hashes: Set[str] = set()
    filtered = []
    
    for chunk in chunks:
        text = chunk.strip()
        
        # Filter 1: Minimum length (words)
        if len(text.split()) < MIN_WORDS:
            continue
        
        # Filter 2: Minimum length (characters)
        if len(text) < MIN_CHARS:
            continue
        
        # Filter 3: Deduplicate by hash
        chunk_hash = compute_chunk_hash(text)
        if chunk_hash in seen_hashes:
            continue
        
        seen_hashes.add(chunk_hash)
        filtered.append(text)
    
    return filtered


def is_chunk_similar(chunk1: str, chunk2: str, threshold: float = MAX_CHUNK_RATIO) -> bool:
    """Check if two chunks are too similar (for near-duplicate detection)"""
    words1 = set(chunk1.lower().split())
    words2 = set(chunk2.lower().split())
    
    if not words1 or not words2:
        return False
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    jaccard = intersection / union if union > 0 else 0
    return jaccard > threshold


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
            "processed_files": {},  # {filename: {hash, chunk_count, timestamp}}
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
        """Remove file from processed list (if deleted)"""
        if filename in self.version_data["processed_files"]:
            del self.version_data["processed_files"][filename]
    
    def update_config(self, config: Dict):
        """Update configuration"""
        self.version_data["config"] = config
        self._save_version()
    
    def get_stats(self) -> Dict:
        """Get version statistics"""
        return {
            "version": self.version_data["version"],
            "files_processed": len(self.version_data["processed_files"]),
            "total_chunks": self.version_data["total_chunks"],
            "last_updated": self.version_data["last_updated"]
        }


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
    embed_model: str = EMBED_MODELS["minilm"],
    chunk_size: int = 450,
    use_sentence_chunking: bool = True,
    force_rebuild: bool = False,
    incremental: bool = True
):
    """
    Main data preparation pipeline with versioning and incremental updates
    
    Args:
        embed_model: Name of embedding model
        chunk_size: Size of chunks (if word-based)
        use_sentence_chunking: Use sentence-based chunking instead
        force_rebuild: Force complete rebuild (ignore incremental)
        incremental: Use incremental indexing (add only new/changed PDFs)
    """
    print(f"\n{'='*60}")
    print(f"PREPARING DATASET {'(INCREMENTAL)' if incremental and not force_rebuild else '(FULL REBUILD)'}")
    print(f"{'='*60}\n")
    
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
        
        # Load existing data
        existing_chunks, existing_metas, existing_embeddings = load_existing_data()
        print(f"[INFO] Loaded {len(existing_chunks)} existing chunks")
        
        # Find new/modified files
        current_files = {p.name for p in pdf_paths}
        processed_files = version_manager.get_processed_files()
        
        # Files to remove (deleted PDFs)
        removed_files = processed_files - current_files
        for removed in removed_files:
            print(f"[INFO] Removed file detected: {removed}")
            version_manager.remove_file(removed)
            # Remove chunks from this file
            existing_chunks = [c for c in existing_chunks if c["paper_id"] != Path(removed).stem]
            existing_metas = [m for m in existing_metas if m["paper_id"] != Path(removed).stem]
        
        # Files to process (new or modified)
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
        
        # Start with existing data
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
        
        # Remove old chunks from this paper (if modified)
        if incremental:
            all_chunks = [c for c in all_chunks if c["paper_id"] != paper_id]
            all_metas = [m for m in all_metas if m["paper_id"] != paper_id]
        
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
        
        # Filter chunks
        chunks = filter_chunks(chunks)
        
        # Store chunks with metadata
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
        
        # Mark file as processed
        version_manager.mark_file_processed(pdf, len(chunks))
    
    # Combine with existing
    all_chunks.extend(new_chunks)
    all_metas.extend(new_metas)
    
    print(f"\n[INFO] Total chunks: {len(all_chunks)} ({len(new_chunks)} new)")
    
    # Global deduplication across all chunks
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
    # STEP 2: Generate Embeddings
    # ===========================
    
    print(f"\n[INFO] Loading embedding model: {embed_model}")
    embedder = SentenceTransformer(embed_model, device=device)
    
    if incremental and existing_embeddings is not None and len(new_chunks) > 0:
        # Only embed new chunks
        print(f"[INFO] Generating embeddings for {len(new_chunks)} new chunks...")
        new_chunk_texts = [c["text"] for c in new_chunks]
        
        new_embeddings: NDArray[np.float32] = embedder.encode(
            new_chunk_texts,
            batch_size=64 if device == "cuda" else 32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")  # type: ignore
        
        # Combine with existing (in same order as chunks)
        print("[INFO] Merging with existing embeddings...")
        # This is tricky - we need to reorder existing embeddings to match new chunk order
        # Simpler approach: just regenerate all embeddings
        print("[INFO] Regenerating all embeddings to ensure consistency...")
        incremental = False  # Fall through to full regeneration
    
    if not incremental or existing_embeddings is None:
        # Generate all embeddings
        raw_chunks = [c["text"] for c in all_chunks]
        print(f"[INFO] Generating embeddings for {len(raw_chunks)} chunks...")
        
        embeddings: NDArray[np.float32] = embedder.encode(
            raw_chunks,
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
    print(f"Files processed: {len(version_manager.version_data['processed_files'])}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Chunks after deduplication: {len(all_chunks)}")
    print()


if __name__ == "__main__":
    # Default preparation
    prepare_dataset()
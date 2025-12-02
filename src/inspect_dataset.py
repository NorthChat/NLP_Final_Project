"""
Dataset inspection and statistics tool
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from datetime import datetime

from config import *


def load_version_info():
    """Load version information"""
    version_file = DATA_DIR / "version.json"
    if not version_file.exists():
        return None
    
    with open(version_file, 'r') as f:
        return json.load(f)


def load_chunks():
    """Load all chunks"""
    if not CHUNKS_FILE.exists():
        return []
    
    chunks = []
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def analyze_dataset():
    """Analyze dataset and print statistics"""
    print("\n" + "="*70)
    print("DATASET INSPECTION")
    print("="*70 + "\n")
    
    # Version info
    version_info = load_version_info()
    if version_info:
        print("📦 VERSION INFORMATION")
        print("-" * 70)
        print(f"Version:          {version_info['version']}")
        print(f"Created:          {version_info['created']}")
        print(f"Last Updated:     {version_info['last_updated']}")
        print(f"Total Chunks:     {version_info['total_chunks']}")
        print()
        
        print("⚙️  CONFIGURATION")
        print("-" * 70)
        config = version_info.get('config', {})
        for key, value in config.items():
            print(f"{key:20s}: {value}")
        print()
        
        print("📄 PROCESSED FILES")
        print("-" * 70)
        processed = version_info.get('processed_files', {})
        for filename, info in sorted(processed.items()):
            chunks = info.get('chunk_count', 0)
            timestamp = info.get('timestamp', 'N/A')
            print(f"{filename:40s} | {chunks:4d} chunks | {timestamp}")
        print()
    else:
        print("[WARNING] No version information found. Run prepare_dataset first.")
        print()
    
    # Chunk statistics
    chunks = load_chunks()
    if not chunks:
        print("[WARNING] No chunks found.")
        return
    
    print("📊 CHUNK STATISTICS")
    print("-" * 70)
    print(f"Total Chunks:     {len(chunks)}")
    
    # Chunks per paper
    paper_counts = Counter(c['paper_id'] for c in chunks)
    print(f"Papers:           {len(paper_counts)}")
    print(f"Avg chunks/paper: {len(chunks) / len(paper_counts):.1f}")
    print(f"Min chunks/paper: {min(paper_counts.values())}")
    print(f"Max chunks/paper: {max(paper_counts.values())}")
    print()
    
    print("📑 CHUNKS PER PAPER")
    print("-" * 70)
    for paper_id, count in sorted(paper_counts.items(), key=lambda x: -x[1]):
        print(f"{paper_id:40s}: {count:4d} chunks")
    print()
    
    # Chunk length statistics
    chunk_lengths = [len(c['text']) for c in chunks]
    word_counts = [len(c['text'].split()) for c in chunks]
    
    print("📏 CHUNK LENGTH STATISTICS")
    print("-" * 70)
    print(f"Characters - Min:  {min(chunk_lengths)}")
    print(f"Characters - Max:  {max(chunk_lengths)}")
    print(f"Characters - Mean: {np.mean(chunk_lengths):.1f}")
    print(f"Characters - Med:  {np.median(chunk_lengths):.1f}")
    print()
    print(f"Words - Min:       {min(word_counts)}")
    print(f"Words - Max:       {max(word_counts)}")
    print(f"Words - Mean:      {np.mean(word_counts):.1f}")
    print(f"Words - Med:       {np.median(word_counts):.1f}")
    print()
    
    # Embeddings info
    if EMBS_FILE.exists():
        embeddings = np.load(EMBS_FILE)
        print("🔢 EMBEDDINGS")
        print("-" * 70)
        print(f"Shape:            {embeddings.shape}")
        print(f"Dimension:        {embeddings.shape[1]}")
        print(f"Size on disk:     {EMBS_FILE.stat().st_size / 1e6:.2f} MB")
        print()
    
    # Index info
    if INDEX_FILE.exists():
        import faiss
        index = faiss.read_index(str(INDEX_FILE))  # type: ignore
        print("🔍 FAISS INDEX")
        print("-" * 70)
        print(f"Vectors:          {index.ntotal}")  # type: ignore
        print(f"Size on disk:     {INDEX_FILE.stat().st_size / 1e6:.2f} MB")
        print()
    
    # Sample chunks
    print("📝 SAMPLE CHUNKS")
    print("-" * 70)
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"Sample {i}:")
        print(f"  Paper ID:  {chunk['paper_id']}")
        print(f"  Chunk ID:  {chunk['chunk_id']}")
        print(f"  Length:    {len(chunk['text'])} chars, {len(chunk['text'].split())} words")
        print(f"  Text:      {chunk['text'][:150]}...")
        print()
    
    print("="*70)
    print()


def check_duplicates():
    """Check for duplicate chunks"""
    print("\n" + "="*70)
    print("DUPLICATE CHECK")
    print("="*70 + "\n")
    
    chunks = load_chunks()
    if not chunks:
        print("[WARNING] No chunks found.")
        return
    
    # Check exact duplicates
    texts = [c['text'] for c in chunks]
    text_counts = Counter(texts)
    duplicates = {text: count for text, count in text_counts.items() if count > 1}
    
    print(f"Total chunks:        {len(chunks)}")
    print(f"Unique chunks:       {len(text_counts)}")
    print(f"Duplicate chunks:    {len(duplicates)}")
    print()
    
    if duplicates:
        print("⚠️  DUPLICATES FOUND:")
        print("-" * 70)
        for text, count in sorted(duplicates.items(), key=lambda x: -x[1])[:5]:
            print(f"Count: {count}")
            print(f"Text:  {text[:100]}...")
            print()
    else:
        print("✅ No exact duplicates found!")
    
    print("="*70)
    print()


def check_quality():
    """Check chunk quality"""
    print("\n" + "="*70)
    print("QUALITY CHECK")
    print("="*70 + "\n")
    
    chunks = load_chunks()
    if not chunks:
        print("[WARNING] No chunks found.")
        return
    
    # Check for very short chunks
    short_chunks = [c for c in chunks if len(c['text'].split()) < 20]
    
    # Check for very long chunks
    long_chunks = [c for c in chunks if len(c['text'].split()) > 500]
    
    # Check for chunks with many numbers (might be tables)
    def count_numbers(text):
        return sum(1 for word in text.split() if any(char.isdigit() for char in word))
    
    number_heavy = [c for c in chunks if count_numbers(c['text']) / len(c['text'].split()) > 0.3]
    
    print(f"Total chunks:           {len(chunks)}")
    print(f"Short chunks (<20w):    {len(short_chunks)}")
    print(f"Long chunks (>500w):    {len(long_chunks)}")
    print(f"Number-heavy (>30%):    {len(number_heavy)}")
    print()
    
    if short_chunks:
        print("⚠️  SHORT CHUNKS:")
        print("-" * 70)
        for c in short_chunks[:3]:
            print(f"Paper: {c['paper_id']}")
            print(f"Words: {len(c['text'].split())}")
            print(f"Text:  {c['text']}")
            print()
    
    if long_chunks:
        print("⚠️  LONG CHUNKS:")
        print("-" * 70)
        for c in long_chunks[:3]:
            print(f"Paper: {c['paper_id']}")
            print(f"Words: {len(c['text'].split())}")
            print(f"Text:  {c['text'][:200]}...")
            print()
    
    print("="*70)
    print()


def main():
    """Run all inspections"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "stats":
            analyze_dataset()
        elif command == "duplicates":
            check_duplicates()
        elif command == "quality":
            check_quality()
        else:
            print("Usage: python inspect_dataset.py [stats|duplicates|quality]")
    else:
        # Run all
        analyze_dataset()
        check_duplicates()
        check_quality()


if __name__ == "__main__":
    main()
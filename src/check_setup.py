"""
Check system setup and diagnose issues
"""
import sys
sys.path.insert(0, 'src')

from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import *


def check_setup():
    """Check if everything is set up correctly"""
    
    print("\n" + "="*70)
    print("SYSTEM SETUP CHECK")
    print("="*70 + "\n")
    
    # Check 1: Files exist
    print("📁 File Check:")
    files_to_check = {
        "Chunks": CHUNKS_FILE,
        "Metadata": METAS_FILE,
        "Embeddings": EMBS_FILE,
        "FAISS Index": INDEX_FILE,
        "Version Info": DATA_DIR / "version.json"
    }
    
    all_exist = True
    for name, path in files_to_check.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_exist = False
    print()
    
    if not all_exist:
        print("❌ Missing files! Please run:")
        print("   python src/main.py --prepare --model <model_name>\n")
        return
    
    # Check 2: Version info
    print("📋 Version Info:")
    version_file = DATA_DIR / "version.json"
    with open(version_file, 'r') as f:
        version_data = json.load(f)
    
    config = version_data.get('config', {})
    indexed_model = config.get('embed_model', 'Unknown')
    is_e5 = config.get('is_e5_model', False)
    chunk_size = config.get('chunk_size', 'Unknown')
    total_chunks = version_data.get('total_chunks', 0)
    
    print(f"  Model: {indexed_model}")
    print(f"  E5 Model: {'Yes' if is_e5 else 'No'}")
    print(f"  Chunk Size: {chunk_size}")
    print(f"  Total Chunks: {total_chunks}")
    print(f"  Version: {version_data.get('version', 'Unknown')}")
    print()
    
    # Check 3: Embeddings dimension
    print("🔢 Embeddings Check:")
    embeddings = np.load(EMBS_FILE)
    emb_dim = embeddings.shape[1]
    emb_count = embeddings.shape[0]
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dimension: {emb_dim}")
    print(f"  Count: {emb_count}")
    print()
    
    # Check 4: FAISS index
    print("🔍 FAISS Index Check:")
    index = faiss.read_index(str(INDEX_FILE))
    index_dim = index.d
    index_count = index.ntotal
    print(f"  Dimension: {index_dim}")
    print(f"  Vectors: {index_count}")
    print()
    
    # Check 5: Dimension consistency
    print("🔧 Consistency Check:")
    
    if emb_dim != index_dim:
        print(f"  ❌ MISMATCH: Embeddings ({emb_dim}D) != Index ({index_dim}D)")
        print(f"     This will cause retrieval errors!")
        print()
        print("  🔨 Fix: Rebuild the index")
        print(f"     python src/main.py --prepare --force-rebuild")
        print()
        return
    else:
        print(f"  ✓ Embeddings and Index dimensions match ({emb_dim}D)")
    
    if emb_count != index_count:
        print(f"  ⚠️  WARNING: Embedding count ({emb_count}) != Index count ({index_count})")
    else:
        print(f"  ✓ Embedding and Index counts match ({emb_count})")
    
    if emb_count != total_chunks:
        print(f"  ⚠️  WARNING: Embeddings ({emb_count}) != Chunks ({total_chunks})")
    else:
        print(f"  ✓ Embeddings and Chunks counts match ({emb_count})")
    print()
    
    # Check 6: Model dimension mapping
    print("📊 Expected Model Dimensions:")
    model_dims = {
        "all-MiniLM-L6-v2": 384,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "intfloat/e5-small-v2": 384,
        "intfloat/e5-base-v2": 768,
        "thenlper/gte-small": 384,
    }
    
    if indexed_model in model_dims:
        expected_dim = model_dims[indexed_model]
        print(f"  {indexed_model}: {expected_dim}D")
        
        if emb_dim == expected_dim:
            print(f"  ✓ Current dimension ({emb_dim}D) matches expected ({expected_dim}D)")
        else:
            print(f"  ❌ MISMATCH: Current ({emb_dim}D) != Expected ({expected_dim}D)")
            print(f"     The index might have been built with a different model!")
    else:
        print(f"  Unknown model: {indexed_model}")
    print()
    
    # Check 7: Test loading model
    print("🧪 Testing Model Load:")
    try:
        print(f"  Loading {indexed_model}...")
        test_embedder = SentenceTransformer(indexed_model)
        test_dim = test_embedder.get_sentence_embedding_dimension()
        print(f"  ✓ Model loaded successfully")
        print(f"  Model dimension: {test_dim}D")
        
        if test_dim != emb_dim:
            print(f"  ❌ CRITICAL: Model dimension ({test_dim}D) != Index dimension ({emb_dim}D)")
            print(f"     You MUST rebuild with --force-rebuild!")
        else:
            print(f"  ✓ Model and Index dimensions match!")
        print()
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        print()
    
    # Final verdict
    print("="*70)
    if emb_dim == index_dim:
        print("✅ SETUP LOOKS GOOD! You can run the UI:")
        print("   python src/main.py --ui")
    else:
        print("❌ SETUP HAS ISSUES! Please fix before running UI:")
        print(f"   python src/main.py --prepare --force-rebuild --model <model>")
        print("\n   Available models: minilm, bge, bge-base, e5-small, e5-base")
    print("="*70 + "\n")


if __name__ == "__main__":
    check_setup()
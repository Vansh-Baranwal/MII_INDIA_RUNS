import importlib
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_imports():
    logging.info("--- 1. Checking Module Imports ---")
    required = ['polars', 'numpy', 'faiss', 'sentence_transformers']
    for req in required:
        try:
            importlib.import_module(req)
            logging.info(f"[OK] {req} is installed.")
        except ImportError:
            logging.error(f"[FAIL] {req} is missing! Run: pip install -r requirements.txt")
            sys.exit(1)

def check_directories():
    logging.info("--- 2. Checking Directories ---")
    base_dir = Path(__file__).resolve().parent.parent
    src_dir = base_dir / 'src'
    artifacts_dir = base_dir / 'artifacts'
    
    if not src_dir.exists():
        logging.error(f"[FAIL] src/ directory missing at {src_dir}")
        sys.exit(1)
        
    if not artifacts_dir.exists():
        logging.warning(f"[WARN] artifacts/ directory missing. Creating at {artifacts_dir}")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    else:
        logging.info(f"[OK] artifacts/ directory exists.")
        
    return base_dir, artifacts_dir

def check_inputs(base_dir: Path):
    logging.info("--- 3. Checking Input Files ---")
    candidates_file = base_dir / 'candidates.jsonl'
    if not candidates_file.exists():
        logging.error(f"[FAIL] candidates.jsonl missing at {candidates_file}")
        sys.exit(1)
    logging.info(f"[OK] candidates.jsonl exists. Size: {candidates_file.stat().st_size / (1024*1024):.2f} MB")

def test_model_download():
    logging.info("--- 4. Checking SentenceTransformer Model Cache ---")
    try:
        from sentence_transformers import SentenceTransformer
        _ = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("[OK] Model loaded from cache successfully.")
    except Exception as e:
        logging.error(f"[FAIL] Model download failed: {e}")
        sys.exit(1)

def main():
    check_imports()
    base_dir, _ = check_directories()
    check_inputs(base_dir)
    test_model_download()
    logging.info("\n>>> PREFLIGHT CHECKS PASSED. SYSTEM READY FOR EXECUTION. <<<")

if __name__ == "__main__":
    main()

import os
import shutil
from pathlib import Path

def rollback():
    base_dir = Path(__file__).resolve().parent
    backup_dir = base_dir / "backups" / "state_pre_expansion"
    if not backup_dir.exists():
        print(f"Error: Backup directory not found at {backup_dir}")
        return False

    print("Restoring project to pre-expansion snapshot...")

    # 1. Restore checkpoints
    ckpt_src = backup_dir / "checkpoints"
    ckpt_dst = base_dir / "models" / "checkpoints"
    if (ckpt_src / "multilingual_transformer.pt").exists():
        shutil.copy2(ckpt_src / "multilingual_transformer.pt", ckpt_dst / "multilingual_transformer.pt")
        print("Restored: multilingual_transformer.pt")
    if (ckpt_src / "classical_baseline.joblib").exists():
        shutil.copy2(ckpt_src / "classical_baseline.joblib", ckpt_dst / "classical_baseline.joblib")
        print("Restored: classical_baseline.joblib")

    # 2. Restore database
    db_src = backup_dir / "hate_speech_detector.db"
    db_dst = base_dir / "data" / "hate_speech_detector.db"
    if db_src.exists():
        shutil.copy2(db_src, db_dst)
        print("Restored: data/hate_speech_detector.db")

    # 3. Restore source files
    src_src = backup_dir / "src"
    if (src_src / "transformer_classifier.py").exists():
        shutil.copy2(src_src / "transformer_classifier.py", base_dir / "src" / "models" / "transformer_classifier.py")
        print("Restored: src/models/transformer_classifier.py")
    if (src_src / "text_attribution.py").exists():
        shutil.copy2(src_src / "text_attribution.py", base_dir / "src" / "explainability" / "text_attribution.py")
        print("Restored: src/explainability/text_attribution.py")

    # 4. Restore data_raw
    raw_src = backup_dir / "data_raw"
    raw_dst = base_dir / "data" / "raw"
    if raw_src.exists():
        shutil.rmtree(raw_dst, ignore_errors=True)
        shutil.copytree(raw_src, raw_dst)
        print("Restored: data/raw directory")

    print("\nSUCCESS: Entire project state successfully rolled back to pre-expansion baseline!")
    return True

if __name__ == "__main__":
    rollback()

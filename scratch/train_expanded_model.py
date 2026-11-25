import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.adapter import UniversalDatasetAdapter
from src.models.transformer_classifier import MultilingualTransformerEngine
from scratch.benchmark_1150_suite import run_1150_benchmark

def main():
    print("=" * 70)
    print("TRAINING MULTILINGUAL TRANSFORMER ON EXPANDED CORPUS (6,250 SAMPLES)")
    print("=" * 70)

    start_time = time.time()

    # 1. Load all 4 datasets
    print("Loading datasets...")
    adapter_hasoc = UniversalDatasetAdapter(
        file_path="data/raw/hasoc_hi_en.jsonl",
        text_column="text",
        label_column="label",
        language_column="language"
    )
    records_hasoc = adapter_hasoc.load()

    adapter_subtle = UniversalDatasetAdapter(
        file_path="data/raw/subtle_structural_benchmark.jsonl",
        text_column="text",
        label_column="label",
        language_column="language"
    )
    records_subtle = adapter_subtle.load()

    adapter_davidson = UniversalDatasetAdapter(
        file_path="data/raw/davidson_hatespeech.jsonl",
        text_column="text",
        label_column="label",
        language_column="language"
    )
    records_davidson = adapter_davidson.load()

    adapter_hatexplain = UniversalDatasetAdapter(
        file_path="data/raw/hatexplain_expanded.jsonl",
        text_column="text",
        label_column="label",
        language_column="language"
    )
    records_hatexplain = adapter_hatexplain.load()

    all_records = records_hasoc + records_subtle + records_davidson + records_hatexplain
    print(f"Loaded records:")
    print(f"  - HASOC (Hindi/English code-mixed): {len(records_hasoc)}")
    print(f"  - Subtle & Structural Hate Speech:  {len(records_subtle)}")
    print(f"  - Davidson et al. (Social Media):   {len(records_davidson)}")
    print(f"  - HateXplain Expanded:              {len(records_hatexplain)}")
    print(f"  Total Training Records:             {len(all_records)}")

    # 2. Train transformer engine
    engine = MultilingualTransformerEngine()
    print("\nBeginning training (6 epochs, lr=1.5e-3, batch_size=32)...")
    train_stats = engine.train(all_records, epochs=6, lr=1.5e-3, batch_size=32)
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds!")
    print(f"Final Epoch Loss: {train_stats['final_loss']:.4f}")
    print(f"Loss History:     {train_stats['loss_history']}")
    print(f"Vocabulary Size:  {train_stats['vocab_size']} unique tokens")

    # 3. Save checkpoint
    print("\nSaving checkpoint to models/checkpoints/multilingual_transformer.pt...")
    engine.save_checkpoint()
    print("Checkpoint saved successfully.")

    # 4. Run 1,150-sample benchmark suite
    print("\n" + "=" * 70)
    print("EVALUATING RETRAINED MODEL ON 1,150-SAMPLE BENCHMARK")
    print("=" * 70)
    benchmark_res = run_1150_benchmark()

    print("\nEvaluation Complete!")

if __name__ == "__main__":
    main()

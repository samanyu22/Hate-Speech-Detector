import random
from typing import List, Tuple, Dict
from src.data.adapter import DatasetRecord

class DatasetSplitter:
    @staticmethod
    def split(
        records: List[DatasetRecord],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[DatasetRecord], List[DatasetRecord], List[DatasetRecord]]:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Split ratios must sum to 1.0"

        # Stratify by label to guarantee identical class distribution
        hateful = [r for r in records if r.label == 1]
        non_hateful = [r for r in records if r.label == 0]

        rng = random.Random(seed)
        rng.shuffle(hateful)
        rng.shuffle(non_hateful)

        def _partition(items: List[DatasetRecord]):
            n = len(items)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train = items[:n_train]
            val = items[n_train:n_train + n_val]
            test = items[n_train + n_val:]
            return train, val, test

        h_train, h_val, h_test = _partition(hateful)
        nh_train, nh_val, nh_test = _partition(non_hateful)

        train = h_train + nh_train
        val = h_val + nh_val
        test = h_test + nh_test

        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

        return train, val, test


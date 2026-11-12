"""
Sanity Checks Validation Suite (Mandatory per Project Specification Part 14)
Validates system behavior on:
1. Clearly genuine multimodal pairs
2. Clearly manipulated samples
3. Deliberately mismatched multimodal pairs (testing semantic inconsistency vs media manipulation)
4. Incomplete samples with missing modalities
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
from src.classification.detector import MultimodalVerificationSystem
from src.data.dataset_manager import DatasetManager


class SanityCheckSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = MultimodalVerificationSystem()
        cls.dm = DatasetManager()
        train, val, test = cls.dm.get_splits("benchmark_300")
        # Ensure models are trained
        if not cls.detector.text_baseline.is_trained:
            cls.detector.text_baseline.fit(train, val, epochs=2)
        if not cls.detector.vision_baseline.is_trained:
            cls.detector.vision_baseline.fit(train, val, epochs=2)
        if not cls.detector.audio_baseline.is_trained:
            cls.detector.audio_baseline.fit(train, val, epochs=2)
        if not cls.detector.cross_modal_analyzer.is_calibrated:
            cls.detector.cross_modal_analyzer.fit_alignment(train, cls.detector.text_baseline, cls.detector.vision_baseline, cls.detector.audio_baseline, epochs=3)

    def test_clearly_genuine_pair(self):
        """Test Case 1: Clearly genuine multimodal pair (authentic image + factual caption)."""
        text = "Official press briefing at the summit discussing bilateral trade agreements."
        img_path = "data/benchmark_300/images/real_0000.jpg"

        res = self.detector.verify_sample(text=text, image_source=img_path, audio_source=None)
        print("\n[Sanity Check 1 - Clearly Genuine]")
        print("Input:", text)
        print("Prediction:", res["final_prediction"], "Confidence:", res["confidence"])
        print("Explanation:", res["explanation"])
        self.assertIn(res["final_prediction"], ["REAL", "UNCERTAIN"])
        self.assertFalse(res["cross_modal_consistency"]["cross_modal_contradiction_detected"])

    def test_clearly_manipulated_sample(self):
        """Test Case 2: Clearly manipulated sample (face manipulation / synthetic audio)."""
        text = "Deepfake video circulating of political leader endorsing unauthorized financial crypto token."
        img_path = "data/benchmark_300/images/fake_0000.jpg"
        aud_path = "data/benchmark_300/audio/fake_0000.wav"

        res = self.detector.verify_sample(text=text, image_source=img_path, audio_source=aud_path)
        print("\n[Sanity Check 2 - Clearly Manipulated]")
        print("Input:", text)
        print("Prediction:", res["final_prediction"], "Confidence:", res["confidence"])
        print("Detected manipulation:", res["detected_manipulation_type"])
        self.assertIn(res["final_prediction"], ["FAKE", "UNCERTAIN"])

    def test_deliberately_mismatched_pair(self):
        """
        Test Case 3: Deliberately mismatched multimodal pair.
        Genuine image paired with a completely unrelated sensationalist caption.
        Must detect cross-modal semantic contradiction!
        """
        text = "Shocking footage shows downtown skyscraper collapsing into dust today!"
        img_path = "data/benchmark_300/images/real_0000.jpg"  # Authentic government summit image
        aud_path = "data/benchmark_300/audio/real_0000.wav"

        res = self.detector.verify_sample(text=text, image_source=img_path, audio_source=aud_path)
        print("\n[Sanity Check 3 - Deliberately Mismatched]")
        print("Input:", text)
        print("Prediction:", res["final_prediction"], "Confidence:", res["confidence"])
        print("Contradiction detected:", res["cross_modal_consistency"]["cross_modal_contradiction_detected"])
        print("Explanation:", res["explanation"])
        # Should flag cross-modal semantic contradiction
        self.assertTrue(res["cross_modal_consistency"]["cross_modal_contradiction_detected"])

    def test_missing_modality_handling(self):
        """Test Case 4: Missing modality (Audio omitted, Text + Image only)."""
        text = "Scientists at the international laboratory announce new solar energy efficiency milestone."
        img_path = "data/benchmark_300/images/real_0001.jpg"

        res = self.detector.verify_sample(text=text, image_source=img_path, audio_source=None)
        print("\n[Sanity Check 4 - Missing Modality]")
        print("Available modalities:", res["available_modalities"])
        print("Warnings:", res["warnings"])
        self.assertIn("audio", res["warnings"][0].lower())
        self.assertEqual(res["available_modalities"], ["text", "image"])


if __name__ == "__main__":
    unittest.main()

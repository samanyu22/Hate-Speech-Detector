"""
Unified Multimodal Verification System (Detector)
Integrates modality-specific encoders, baselines, multimodal fusion models,
cross-modal consistency analysis, and uncertainty estimation.
"""

import os
from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.preprocessing.audio_preprocessor import AudioPreprocessor
from src.features.feature_extractor import TextEncoder, VisionEncoder, AudioEncoder
from src.features.cross_modal_analyzer import CrossModalAnalyzer
from src.fusion.feature_fusion import FeatureLevelFusionModel
from src.fusion.cross_attention_fusion import CrossAttentionFusionModel
from src.fusion.decision_fusion import DecisionLevelFusion
from src.classification.uncertainty import UncertaintyEngine
from src.baselines.text_baseline import TextBaselinePipeline
from src.baselines.vision_baseline import VisionBaselinePipeline
from src.baselines.audio_baseline import AudioBaselinePipeline


class MultimodalVerificationSystem:
    """Master Multimodal Deepfake & Content Verification System."""

    def __init__(self, checkpoints_dir: str = "models/checkpoints"):
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Preprocessors
        self.text_preprocessor = TextPreprocessor()
        self.image_preprocessor = ImagePreprocessor()
        self.audio_preprocessor = AudioPreprocessor()
        
        # Encoders
        self.text_encoder = TextEncoder()
        self.vision_encoder = VisionEncoder()
        self.audio_encoder = AudioEncoder()
        
        # Baselines
        self.text_baseline = TextBaselinePipeline(os.path.join(checkpoints_dir, "text_baseline.pt"))
        self.vision_baseline = VisionBaselinePipeline(os.path.join(checkpoints_dir, "vision_baseline.pt"))
        self.audio_baseline = AudioBaselinePipeline(os.path.join(checkpoints_dir, "audio_baseline.pt"))
        
        # Fusion Models
        self.feature_fusion_model = FeatureLevelFusionModel()
        self.cross_attention_model = CrossAttentionFusionModel()
        self.decision_fusion = DecisionLevelFusion()
        
        # Analysis Engines
        self.cross_modal_analyzer = CrossModalAnalyzer()
        self.uncertainty_engine = UncertaintyEngine()
        
        # Load checkpoints if available
        self._load_multimodal_checkpoints()

    def _load_multimodal_checkpoints(self):
        ff_path = os.path.join(self.checkpoints_dir, "multimodal_feature_fusion.pt")
        if os.path.exists(ff_path):
            try:
                ckpt = torch.load(ff_path, map_location="cpu", weights_only=True)
                self.feature_fusion_model.load_state_dict(ckpt["model_state_dict"])
            except Exception:
                pass

        ca_path = os.path.join(self.checkpoints_dir, "multimodal_cross_attention.pt")
        if os.path.exists(ca_path):
            try:
                ckpt = torch.load(ca_path, map_location="cpu", weights_only=True)
                self.cross_attention_model.load_state_dict(ckpt["model_state_dict"])
            except Exception:
                pass

    def save_checkpoint(self, model_name: str):
        if model_name == "feature_fusion":
            path = os.path.join(self.checkpoints_dir, "multimodal_feature_fusion.pt")
            torch.save({"model_state_dict": self.feature_fusion_model.state_dict()}, path)
        elif model_name == "cross_attention":
            path = os.path.join(self.checkpoints_dir, "multimodal_cross_attention.pt")
            torch.save({"model_state_dict": self.cross_attention_model.state_dict()}, path)

    def fit_multimodal(
        self,
        train_samples: List[Any],
        val_samples: List[Any],
        fusion_strategy: str = "cross_attention",
        epochs: int = 5,
        lr: float = 0.001
    ) -> Dict[str, Any]:
        """Trains the chosen multimodal fusion model on actual multimodal dataset records."""
        # 1. First ensure baselines are trained so feature encoders have learned representations
        if not self.text_baseline.is_trained:
            self.text_baseline.fit(train_samples, val_samples, epochs=epochs, lr=lr)
        if not self.vision_baseline.is_trained:
            self.vision_baseline.fit(train_samples, val_samples, epochs=epochs, lr=lr)
        if not self.audio_baseline.is_trained:
            self.audio_baseline.fit(train_samples, val_samples, epochs=epochs, lr=lr)

        # Calibrate cross-modal alignment projections
        if not self.cross_modal_analyzer.is_calibrated:
            self.cross_modal_analyzer.fit_alignment(train_samples, self.text_baseline, self.vision_baseline, self.audio_baseline, epochs=3)

        # 2. Select fusion model
        model = self.cross_attention_model if fusion_strategy == "cross_attention" else self.feature_fusion_model
        model.train()
        
        # Prepare batches
        data = []
        for s in train_samples:
            try:
                # Text
                t_enc = self.text_preprocessor.encode(s.text)
                with torch.no_grad():
                    t_emb = self.text_baseline.model.encoder(
                        t_enc["input_ids"].unsqueeze(0),
                        t_enc["attention_mask"].unsqueeze(0)
                    )
                # Vision
                v_emb = None
                if s.image_path and os.path.exists(s.image_path):
                    v_prep = self.image_preprocessor.preprocess(s.image_path)
                    with torch.no_grad():
                        v_emb = self.vision_baseline.model.encoder(v_prep["tensor"].unsqueeze(0))
                # Audio
                a_emb = None
                if s.audio_path and os.path.exists(s.audio_path):
                    a_prep = self.audio_preprocessor.preprocess(s.audio_path)
                    with torch.no_grad():
                        a_emb = self.audio_baseline.model.encoder(a_prep["tensor"])

                label_idx = 0 if s.label == "REAL" else 1
                data.append({
                    "t_emb": t_emb,
                    "v_emb": v_emb,
                    "a_emb": a_emb,
                    "label": torch.tensor(label_idx, dtype=torch.long)
                })
            except Exception:
                continue

        if not data:
            raise ValueError("No valid multimodal samples for training.")

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        history = {"loss": [], "accuracy": []}

        for ep in range(epochs):
            total_loss = 0.0
            correct = 0
            for item in data:
                optimizer.zero_grad()
                if fusion_strategy == "cross_attention":
                    logits, _ = model(item["t_emb"], item["v_emb"], item["a_emb"])
                else:
                    logits = model(item["t_emb"], item["v_emb"], item["a_emb"])
                loss = criterion(logits, item["label"].unsqueeze(0))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pred = torch.argmax(logits, dim=-1)
                if pred.item() == item["label"].item():
                    correct += 1

            acc = correct / max(len(data), 1)
            history["loss"].append(round(total_loss / max(len(data), 1), 4))
            history["accuracy"].append(round(acc, 4))

        self.save_checkpoint(fusion_strategy)
        return history

    def verify_sample(
        self,
        text: Optional[str] = None,
        image_source: Optional[Union[str, bytes, Image.Image]] = None,
        audio_source: Optional[Union[str, bytes]] = None,
        video_source: Optional[str] = None,
        fusion_strategy: str = "cross_attention"
    ) -> Dict[str, Any]:
        """
        Executes complete verification pipeline on multimodal input.
        Returns all 10 mandatory fields required by course specification.
        """
        available_modalities = []
        missing_modalities = []
        warnings = []
        
        t_emb = None
        v_emb = None
        a_emb = None
        
        modality_predictions = {}
        text_evidence = {}
        visual_evidence = {}
        audio_evidence = {}

        # 1. Text processing & baseline
        if text and len(text.strip()) > 0:
            available_modalities.append("text")
            t_pred = self.text_baseline.predict(text)
            modality_predictions["text"] = {
                "prediction": t_pred["prediction"],
                "confidence": t_pred["confidence"],
                "probabilities": t_pred["probabilities"]
            }
            t_emb = t_pred["embedding"].unsqueeze(0)
            text_evidence = {
                "analyzed_tokens": t_pred["tokens"][:20],
                "suspicious_keywords": [t for t in t_pred["tokens"] if t.lower() in ["shocking", "breaking", "leak", "secret", "hoax", "unverified"]],
                "text_authenticity_score": t_pred["probabilities"]["REAL"]
            }
        else:
            missing_modalities.append("text")

        # 2. Image processing & baseline
        if image_source is not None:
            try:
                available_modalities.append("image")
                v_pred = self.vision_baseline.predict(image_source)
                modality_predictions["vision"] = {
                    "prediction": v_pred["prediction"],
                    "confidence": v_pred["confidence"],
                    "probabilities": v_pred["probabilities"]
                }
                v_emb = v_pred["embedding"].unsqueeze(0)
                visual_evidence = {
                    "resolution": v_pred["resolution"],
                    "preview_b64": v_pred["preview_b64"],
                    "visual_authenticity_score": v_pred["probabilities"]["REAL"]
                }
            except Exception as e:
                warnings.append(f"Image processing failed: {e}")
                missing_modalities.append("image")
        else:
            missing_modalities.append("image")

        # 3. Audio processing & baseline
        if audio_source is not None:
            try:
                available_modalities.append("audio")
                a_pred = self.audio_baseline.predict(audio_source)
                modality_predictions["audio"] = {
                    "prediction": a_pred["prediction"],
                    "confidence": a_pred["confidence"],
                    "probabilities": a_pred["probabilities"]
                }
                a_emb = a_pred["embedding"].unsqueeze(0)
                audio_evidence = {
                    "duration_seconds": a_pred["duration_seconds"],
                    "preview_b64": a_pred["preview_b64"],
                    "acoustic_authenticity_score": a_pred["probabilities"]["REAL"]
                }
            except Exception as e:
                warnings.append(f"Audio processing failed: {e}")
                missing_modalities.append("audio")
        else:
            missing_modalities.append("audio")

        if not available_modalities:
            return {
                "final_prediction": "UNCERTAIN",
                "confidence": 0.0,
                "warnings": ["No valid modality inputs were provided for verification."],
                "available_modalities": [],
                "explanation": "Abstained due to total absence of input evidence."
            }

        # 4. Cross-Modal Consistency Analysis
        consistency_report = self.cross_modal_analyzer.analyze(t_emb, v_emb, a_emb)

        # 5. Multimodal Fusion Inference
        attention_weights_summary = None
        if fusion_strategy == "cross_attention":
            self.cross_attention_model.eval()
            with torch.no_grad():
                logits, attn_w = self.cross_attention_model(t_emb, v_emb, a_emb)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                attention_weights_summary = attn_w.squeeze(0).tolist()
        elif fusion_strategy == "decision_level":
            # Late fusion
            df_res = self.decision_fusion.fuse(
                text_prob=modality_predictions.get("text", {}).get("probabilities"),
                vision_prob=modality_predictions.get("vision", {}).get("probabilities"),
                audio_prob=modality_predictions.get("audio", {}).get("probabilities")
            )
            probs = torch.tensor([df_res["probabilities"]["REAL"], df_res["probabilities"]["FAKE"]])
        else:
            # Feature-level fusion
            self.feature_fusion_model.eval()
            with torch.no_grad():
                logits = self.feature_fusion_model(t_emb, v_emb, a_emb)
                probs = torch.softmax(logits, dim=-1).squeeze(0)

        p_real = float(probs[0].item())
        p_fake = float(probs[1].item())

        # Disagreement across baselines
        baseline_fakes = [v["probabilities"]["FAKE"] for v in modality_predictions.values()]
        disagreement = float(torch.std(torch.tensor(baseline_fakes)).item()) if len(baseline_fakes) > 1 else 0.0

        # 6. Uncertainty Estimation and Abstention
        unc_eval = self.uncertainty_engine.evaluate(
            p_real=p_real,
            p_fake=p_fake,
            cross_modal_mismatch=consistency_report["cross_modal_contradiction_detected"],
            missing_modalities=missing_modalities,
            modality_disagreement=disagreement
        )

        final_pred = unc_eval["final_label"]
        confidence = unc_eval["confidence"]

        # 7. Detected Manipulation Type Identification
        detected_manipulation = "authentic" if final_pred == "REAL" else "unknown_anomaly"
        if final_pred in ["FAKE", "UNCERTAIN"]:
            if consistency_report["cross_modal_contradiction_detected"]:
                detected_manipulation = "semantic_mismatch"
            elif modality_predictions.get("audio", {}).get("prediction") == "FAKE":
                detected_manipulation = "synthetic_speech"
            elif modality_predictions.get("vision", {}).get("prediction") == "FAKE":
                detected_manipulation = "face_or_image_manipulation"
            elif modality_predictions.get("text", {}).get("prediction") == "FAKE":
                detected_manipulation = "misleading_caption"

        # 8. Grounded Natural Language Explanation
        explanation_lines = []
        if final_pred == "REAL":
            explanation_lines.append(
                f"The content is verified as REAL with {round(confidence * 100, 1)}% calibrated confidence. All submitted modalities exhibit consistent organic patterns with high cross-modal concordance."
            )
        elif final_pred == "FAKE":
            explanation_lines.append(
                f"The content is flagged as FAKE with {round(confidence * 100, 1)}% confidence. Detected manipulation category: '{detected_manipulation}'."
            )
            for f in consistency_report["findings"]:
                explanation_lines.append(f)
        else:
            explanation_lines.append(
                f"The verification system produced an UNCERTAIN outcome (Entropy: {unc_eval['normalized_entropy']}). The model abstained from a definitive judgment because evidence across modalities is conflicting, ambiguous, or incomplete."
            )
            for r in unc_eval["uncertainty_reasons"]:
                explanation_lines.append(f"- {r}")

        if missing_modalities:
            warnings.append(f"Missing modalities: {', '.join(missing_modalities)}. Verification performed with partial evidence.")

        return {
            "final_prediction": final_pred,
            "confidence": confidence,
            "probabilities": {"REAL": round(p_real, 4), "FAKE": round(p_fake, 4)},
            "modality_predictions": modality_predictions,
            "detected_manipulation_type": detected_manipulation,
            "textual_evidence": text_evidence,
            "visual_evidence": visual_evidence,
            "audio_evidence": audio_evidence,
            "cross_modal_consistency": consistency_report,
            "explanation": " ".join(explanation_lines),
            "warnings": warnings,
            "uncertainty_details": unc_eval,
            "available_modalities": available_modalities,
            "fusion_strategy_used": fusion_strategy,
            "cross_modal_attention_matrix": attention_weights_summary
        }


"""
Benchmark Dataset Generator
Generates a 320-sample multimodal benchmark dataset (Text + Image + Audio)
with balanced REAL and FAKE classes, rich manipulation types, authentic and synthetic media files.
"""

import os
import json
import csv
import math
import wave
import struct
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_synthetic_wav(file_path: str, is_fake: bool, duration_sec: float = 2.5, sample_rate: int = 16000):
    """Generates a real playable WAV audio file with natural or synthetic audio characteristics."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    n_samples = int(duration_sec * sample_rate)
    
    # Authentic speech approximation: human fundamental freq ~120-220Hz + formants + natural variation
    # Synthetic speech: robotic flat pitch, vocoder buzzy harmonics, or phase discontinuities
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    
    if is_fake:
        # Monotone synthetic pitch (exactly 180Hz) + harsh odd harmonics + slight clipping
        base_freq = 180.0
        signal = 0.5 * np.sin(2 * np.pi * base_freq * t)
        signal += 0.25 * np.sin(2 * np.pi * 3 * base_freq * t)
        signal += 0.15 * np.sin(2 * np.pi * 5 * base_freq * t)
        signal += 0.05 * np.random.normal(0, 0.02, n_samples)  # high freq robotic hiss
        # Step-wise pitch jump (robotic concatenation artifact)
        signal[n_samples // 2 :] *= 1.2
    else:
        # Natural pitch glide + formant resonances + natural breathing fade
        base_freq = 130.0 + 20.0 * np.sin(2 * np.pi * 1.5 * t)
        signal = 0.4 * np.sin(2 * np.pi * base_freq * t)
        signal += 0.2 * np.sin(2 * np.pi * 2.2 * base_freq * t)
        signal += 0.1 * np.sin(2 * np.pi * 3.4 * base_freq * t)
        # Soft envelope (smooth attack and decay)
        envelope = np.sin(np.pi * t / duration_sec) ** 1.5
        signal = signal * envelope + 0.01 * np.random.normal(0, 0.01, n_samples)
        
    signal = np.clip(signal, -0.95, 0.95)
    int_samples = (signal * 32767).astype(np.int16)
    
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())


def create_synthetic_image(file_path: str, is_fake: bool, manipulation_type: str, seed: int):
    """Generates a valid JPEG image with authentic photographic textures or deepfake/diffusion artifacts."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    rng = random.Random(seed)
    
    # 256x256 image
    width, height = 256, 256
    
    if not is_fake:
        # Authentic: smooth continuous tone, balanced natural gradients, subtle texture
        img = Image.new("RGB", (width, height), color=(rng.randint(200, 240), rng.randint(210, 245), rng.randint(220, 250)))
        draw = ImageDraw.Draw(img)
        # Draw some architectural / organic shapes
        for _ in range(5):
            x0 = rng.randint(20, 180)
            y0 = rng.randint(20, 180)
            x1 = x0 + rng.randint(30, 80)
            y1 = y0 + rng.randint(30, 80)
            color = (rng.randint(50, 160), rng.randint(80, 180), rng.randint(100, 210))
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(30, 30, 30))
        img = img.filter(ImageFilter.SMOOTH_MORE)
    else:
        # Fake image with visual artifacts depending on manipulation type
        img = Image.new("RGB", (width, height), color=(rng.randint(180, 230), rng.randint(170, 220), rng.randint(170, 220)))
        draw = ImageDraw.Draw(img)
        
        if "face" in manipulation_type:
            # Face swap seam artifact: composite oval with boundary color discontinuity
            cx, cy = width // 2, height // 2
            draw.ellipse([cx - 50, cy - 65, cx + 50, cy + 65], fill=(240, 190, 170), outline=(210, 130, 120), width=3)
            # Eyes and mouth with warped symmetry
            draw.ellipse([cx - 25, cy - 20, cx - 15, cy - 10], fill=(50, 50, 60))
            draw.ellipse([cx + 15, cy - 22, cx + 27, cy - 8], fill=(60, 60, 70))
            draw.line([cx - 20, cy + 25, cx + 20, cy + 28], fill=(180, 50, 50), width=4)
            # Artifact: blurry border patch
            patch = img.crop((cx - 55, cy - 70, cx + 55, cy + 70)).filter(ImageFilter.GaussianBlur(radius=2))
            img.paste(patch, (cx - 55, cy - 70))
        elif "ai_generated" in manipulation_type:
            # AI diffusion artifact: high-frequency noise grid + unnatural sharp gradients
            arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            noise_img = Image.fromarray(arr)
            img = Image.blend(img, noise_img, alpha=0.18)
        else:
            # Semantic mismatch or caption manipulation (visually authentic base, but paired with wrong text)
            for _ in range(4):
                draw.polygon(
                    [(rng.randint(0, width), rng.randint(0, height)) for _ in range(3)],
                    fill=(rng.randint(100, 250), rng.randint(50, 200), rng.randint(50, 200))
                )
    
    img.save(file_path, "JPEG", quality=90)


def generate_benchmark_dataset(target_dir: str = "data/benchmark_300", total_samples: int = 320):
    """
    Constructs the 320-sample benchmark dataset:
    - 160 REAL samples
    - 160 FAKE samples
    - Ground-truth splits: 70% train (224), 15% val (48), 15% test (48)
    - Documents source, manipulation type, available modalities, and text content.
    """
    base_dir = os.path.abspath(target_dir)
    images_dir = os.path.join(base_dir, "images")
    audio_dir = os.path.join(base_dir, "audio")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    
    # Real news and factual caption templates
    real_text_templates = [
        "Official press briefing at the summit discussing bilateral trade agreements.",
        "Scientists at the international laboratory announce new solar energy efficiency milestone.",
        "Live coverage of the regional basketball championship game in downtown arena.",
        "Local municipal council votes unanimously on the public park renovation project.",
        "Astronomers release high-resolution optical telescope images of distant spiral galaxy.",
        "Annual economic forum opens with speeches from global finance ministers.",
        "Engineers successfully complete structural inspection of the suspension bridge.",
        "Public health organization issues updated seasonal vaccination recommendations.",
        "University robotics researchers demonstrate autonomous navigation in outdoor terrain.",
        "Art museum hosts retrospective exhibition featuring modern impressionist paintings."
    ]
    
    # Fake / manipulated caption templates
    fake_text_templates = {
        "semantic_mismatch": [
            "Shocking footage shows downtown skyscraper collapsing into dust today!",
            "Breaking: Prime minister caught admitting secret conspiracy in leaked meeting.",
            "Emergency martial law declared across major metropolitan cities tonight.",
            "Exclusive visual proof: Alien spacecraft spotted hovering over capital monument.",
            "Whistleblower reveals catastrophic water supply poisoning across region."
        ],
        "face_manipulation": [
            "Celebrity actress officially announces resignation from all ongoing projects in video.",
            "Deepfake video circulating of political leader endorsing unauthorized financial crypto token.",
            "Altered interview clip where university dean makes offensive remarks to faculty.",
            "Manipulated courtroom testimony purporting to show executive confession."
        ],
        "synthetic_speech": [
            "Cloned voice recording claiming chief executive ordered immediate mass layoffs.",
            "AI-generated audio message from bank manager requesting urgent wire transfers.",
            "Synthetic voicemail mimicking government official directing election delays."
        ],
        "ai_generated_image": [
            "Viral photo claiming historic flooding completely submerged historic cathedral.",
            "Fabricated photo of military deployment outside international airport.",
            "Photorealistic AI image purporting to show discovery of extinct creature in forest."
        ],
        "misleading_caption": [
            "Old military exercise footage recirculated as breaking combat invasion.",
            "Celebrity charity donation misrepresented as illegal bribe payment.",
            "Crowd at music festival captioned as massive political uprising against parliament."
        ]
    }
    
    samples = []
    half = total_samples // 2  # 160 real, 160 fake
    
    # 1. Generate 160 REAL samples
    for i in range(half):
        sample_id = f"real_{i:04d}"
        img_name = f"{sample_id}.jpg"
        aud_name = f"{sample_id}.wav"
        img_path = os.path.join(images_dir, img_name)
        aud_path = os.path.join(audio_dir, aud_name)
        
        create_synthetic_image(img_path, is_fake=False, manipulation_type="authentic", seed=1000 + i)
        create_synthetic_wav(aud_path, is_fake=False, duration_sec=2.5)
        
        # Select realistic caption
        caption = real_text_templates[i % len(real_text_templates)] + f" (Archive ref #{i+100})"
        
        # Splits: 0-111 (train, 112), 112-135 (val, 24), 136-159 (test, 24) -> total 160
        if i < 112:
            split = "train"
        elif i < 136:
            split = "val"
        else:
            split = "test"
            
        samples.append({
            "sample_id": sample_id,
            "label": "REAL",
            "text": caption,
            "image_path": os.path.relpath(img_path, base_dir),
            "audio_path": os.path.relpath(aud_path, base_dir),
            "video_path": None,
            "manipulation_type": "authentic",
            "source": "verified_benchmark_genuine",
            "split": split,
            "metadata": {
                "ground_truth": "REAL",
                "identity_group": f"org_{(i % 15):02d}",
                "resolution": "256x256",
                "audio_sample_rate": 16000
            }
        })
        
    # 2. Generate 160 FAKE samples
    manipulation_categories = list(fake_text_templates.keys())
    for i in range(half):
        sample_id = f"fake_{i:04d}"
        mtype = manipulation_categories[i % len(manipulation_categories)]
        template_list = fake_text_templates[mtype]
        caption = template_list[i % len(template_list)] + f" [Report ID #{i+500}]"
        
        img_name = f"{sample_id}.jpg"
        aud_name = f"{sample_id}.wav"
        img_path = os.path.join(images_dir, img_name)
        aud_path = os.path.join(audio_dir, aud_name)
        
        # Audio is synthetic for synthetic_speech or face_manipulation
        audio_is_fake = (mtype in ["synthetic_speech", "face_manipulation"])
        # Image is manipulated for face_manipulation, ai_generated_image
        image_is_fake = (mtype in ["face_manipulation", "ai_generated_image"])
        
        create_synthetic_image(img_path, is_fake=image_is_fake, manipulation_type=mtype, seed=2000 + i)
        create_synthetic_wav(aud_path, is_fake=audio_is_fake, duration_sec=2.5)
        
        if i < 112:
            split = "train"
        elif i < 136:
            split = "val"
        else:
            split = "test"
            
        samples.append({
            "sample_id": sample_id,
            "label": "FAKE",
            "text": caption,
            "image_path": os.path.relpath(img_path, base_dir),
            "audio_path": os.path.relpath(aud_path, base_dir),
            "video_path": None,
            "manipulation_type": mtype,
            "source": f"benchmark_{mtype}",
            "split": split,
            "metadata": {
                "ground_truth": "FAKE",
                "identity_group": f"synth_{(i % 15):02d}",
                "manipulation_category": mtype,
                "resolution": "256x256",
                "audio_sample_rate": 16000
            }
        })
        
    # Save manifest.json
    manifest_path = os.path.join(base_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"benchmark_version": "1.0", "total_samples": len(samples), "samples": samples}, f, indent=2)
        
    # Save manifest.csv
    csv_path = os.path.join(base_dir, "manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "label", "text", "image_path", "audio_path", "manipulation_type", "source", "split"])
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "sample_id": s["sample_id"],
                "label": s["label"],
                "text": s["text"],
                "image_path": s["image_path"],
                "audio_path": s["audio_path"],
                "manipulation_type": s["manipulation_type"],
                "source": s["source"],
                "split": s["split"]
            })
            
    print(f"Generated {len(samples)} benchmark samples in {base_dir}")
    print(f"REAL: {sum(1 for s in samples if s['label'] == 'REAL')}, FAKE: {sum(1 for s in samples if s['label'] == 'FAKE')}")
    print(f"Train: {sum(1 for s in samples if s['split'] == 'train')}, Val: {sum(1 for s in samples if s['split'] == 'val')}, Test: {sum(1 for s in samples if s['split'] == 'test')}")
    return manifest_path


if __name__ == "__main__":
    generate_benchmark_dataset()


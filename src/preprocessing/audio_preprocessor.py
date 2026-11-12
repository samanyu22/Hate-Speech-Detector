"""
Audio Preprocessing Pipeline
Handles WAV audio loading, sample rate inspection, resampling, amplitude normalization,
Mel-spectrogram extraction, and visual spectrogram preview generation.
"""

import os
import io
import wave
import base64
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import scipy.signal as signal
from PIL import Image
import torch


class AudioPreprocessor:
    """Preprocesses raw audio waveforms into normalized Mel-spectrogram tensors."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        max_duration_sec: float = 3.0
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_duration_sec = max_duration_sec
        self.max_samples = int(sample_rate * max_duration_sec)

    def load_audio(self, audio_source: Union[str, bytes]) -> Tuple[np.ndarray, int]:
        """Loads audio from WAV file path or bytes and returns floating numpy waveform array."""
        if isinstance(audio_source, str):
            if not os.path.exists(audio_source):
                raise FileNotFoundError(f"Audio file does not exist: {audio_source}")
            with wave.open(audio_source, "rb") as wf:
                orig_sr = wf.getframerate()
                n_channels = wf.getnchannels()
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)
                sampwidth = wf.getsampwidth()
        elif isinstance(audio_source, bytes):
            with wave.open(io.BytesIO(audio_source), "rb") as wf:
                orig_sr = wf.getframerate()
                n_channels = wf.getnchannels()
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)
                sampwidth = wf.getsampwidth()
        else:
            raise TypeError("Unsupported audio source type.")

        # Decode PCM bytes
        if sampwidth == 2:
            dtype = np.int16
            max_val = 32768.0
        elif sampwidth == 1:
            dtype = np.uint8
            max_val = 128.0
        else:
            dtype = np.int32
            max_val = 2147483648.0

        audio_arr = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32) / max_val
        if n_channels > 1:
            # Downmix multi-channel audio to mono
            audio_arr = audio_arr.reshape(-1, n_channels).mean(axis=1)

        return audio_arr, orig_sr

    def resample_and_normalize(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resamples to target sample rate and normalizes peak amplitude."""
        if orig_sr != self.sample_rate and len(audio) > 0:
            target_len = int(len(audio) * self.sample_rate / orig_sr)
            audio = signal.resample(audio, target_len)
            
        # Peak normalization
        peak = np.max(np.abs(audio))
        if peak > 1e-5:
            audio = audio / peak
            
        # Pad or truncate to max_samples
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)), mode="constant")
            
        return audio

    def compute_melspectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Computes log-magnitude STFT / Mel spectrogram representation."""
        # STFT via scipy
        f, t, Zxx = signal.stft(
            audio,
            fs=self.sample_rate,
            nperseg=min(self.n_fft, len(audio)),
            noverlap=self.n_fft - self.hop_length,
            boundary="zeros"
        )
        magnitude = np.abs(Zxx)
        log_spec = np.log1p(magnitude)  # Shape: (freq_bins, time_frames)
        
        # Rescale / interpolate freq_bins to n_mels for fixed tensor dimension
        if log_spec.shape[0] != self.n_mels:
            # Resample across frequency axis to n_mels
            resampled = signal.resample(log_spec, self.n_mels, axis=0)
            log_spec = np.clip(resampled, 0, None)
            
        # Ensure fixed time dimension (e.g. 96 frames for 3.0s with hop 512)
        target_time_frames = 96
        if log_spec.shape[1] > target_time_frames:
            log_spec = log_spec[:, :target_time_frames]
        elif log_spec.shape[1] < target_time_frames:
            log_spec = np.pad(log_spec, ((0, 0), (0, target_time_frames - log_spec.shape[1])), mode="constant")
            
        return log_spec.astype(np.float32)

    def generate_spectrogram_preview(self, spec: np.ndarray) -> str:
        """Converts spectrogram matrix into a colorized base64 PNG image for UI inspection."""
        norm_spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        uint8_spec = (norm_spec * 255).astype(np.uint8)
        # Flip vertically so low frequencies are at bottom
        uint8_spec = np.flipud(uint8_spec)
        
        # Color mapping (viridis-like gradient: purple to green to yellow)
        h, w = uint8_spec.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = uint8_spec  # Red
        rgb[:, :, 1] = (np.sin(uint8_spec / 255.0 * np.pi) * 255).astype(np.uint8)  # Green
        rgb[:, :, 2] = 255 - uint8_spec  # Blue
        
        img = Image.fromarray(rgb).resize((256, 128), Image.Resampling.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    def preprocess(self, audio_source: Union[str, bytes]) -> Dict[str, Any]:
        """Runs full audio preprocessing pipeline."""
        audio, orig_sr = self.load_audio(audio_source)
        duration_sec = len(audio) / orig_sr if orig_sr > 0 else 0.0
        
        normalized_audio = self.resample_and_normalize(audio, orig_sr)
        spec = self.compute_melspectrogram(normalized_audio)
        preview_b64 = self.generate_spectrogram_preview(spec)
        
        # Audio anomalies: pitch variance, harmonic sharpness
        spec_std = float(round(float(np.std(spec)), 4))
        spectral_flatness = float(round(float(np.exp(np.mean(np.log(spec + 1e-6))) / (np.mean(spec) + 1e-6)), 4))
        
        # PyTorch tensor shape: (1, n_mels, time_frames) -> (1, 64, 96)
        tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        
        return {
            "original_sample_rate": orig_sr,
            "target_sample_rate": self.sample_rate,
            "duration_seconds": round(duration_sec, 2),
            "spectrogram_shape": list(spec.shape),
            "tensor": tensor,
            "tensor_shape": list(tensor.shape),
            "spectral_variance": spec_std,
            "spectral_flatness": spectral_flatness,
            "preview_b64": preview_b64
        }


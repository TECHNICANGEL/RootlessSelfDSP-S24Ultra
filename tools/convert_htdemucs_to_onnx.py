#!/usr/bin/env python3
"""
HTDemucs v4 Fine-Tuned to ONNX Converter
Based on GSoC 2025 Mixxx project work (https://github.com/adefossez/demucs/pull/10)

This script converts the HTDemucs FT (fine-tuned) model to ONNX format
for use in Android with ONNX Runtime.

Requirements:
    pip install torch torchaudio onnx onnxruntime
    pip install git+https://github.com/dhunstack/demucs.git@allchanges

Usage:
    python convert_htdemucs_to_onnx.py

Output:
    htdemucs_ft.onnx - Place this in app/src/main/assets/models/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import subprocess

# Check dependencies
def install_gsoc_demucs():
    """Install the GSoC 2025 fork with ONNX-compatible STFT"""
    print("Installing GSoC 2025 demucs fork with ONNX support...")
    print("  Repository: dhunstack/demucs, Branch: allchanges")
    try:
        # Uninstall regular demucs first
        print("  Uninstalling standard demucs...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "demucs"],
                      capture_output=True)
        # Install the GSoC fork with ONNX-compatible STFT
        print("  Installing ONNX-compatible fork (this may take a minute)...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "git+https://github.com/dhunstack/demucs.git@allchanges"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"  ❌ pip install failed: {result.stderr[:500]}")
            return False
        print("  ✅ GSoC demucs fork installed!")
        return True
    except subprocess.TimeoutExpired:
        print("  ❌ Installation timed out. Run manually:")
        print("     pip install git+https://github.com/dhunstack/demucs.git@allchanges")
        return False
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

try:
    import demucs
    from demucs.pretrained import get_model
    from demucs.hdemucs import HDemucs
    from demucs.htdemucs import HTDemucs
    # Check if this is the ONNX-compatible fork
    if not hasattr(demucs, '__onnx_compatible__'):
        print("Standard demucs detected. Attempting to install GSoC ONNX-compatible fork...")
        if install_gsoc_demucs():
            # Reload modules
            import importlib
            importlib.reload(demucs)
            from demucs.pretrained import get_model
            from demucs.hdemucs import HDemucs
            from demucs.htdemucs import HTDemucs
except ImportError:
    print("Demucs not installed. Attempting to install GSoC ONNX-compatible fork...")
    if install_gsoc_demucs():
        import demucs
        from demucs.pretrained import get_model
        from demucs.hdemucs import HDemucs
        from demucs.htdemucs import HTDemucs
    else:
        print("ERROR: Could not install demucs. Run manually:")
        print("  pip install git+https://github.com/dhunstack/demucs.git@onnx-export")
        sys.exit(1)

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnx/onnxruntime not installed. Run: pip install onnx onnxruntime")
    sys.exit(1)


class STFTProcessor(nn.Module):
    """ONNX-compatible STFT using real-valued operations (no complex tensors)"""

    def __init__(self, n_fft=4096, hop_length=1024, center=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center

        # Pre-compute window
        window = torch.hann_window(n_fft)
        self.register_buffer('window', window)

        # Pre-compute DFT matrix for real-valued STFT
        # This avoids using torch.stft which uses complex tensors
        n_freq = n_fft // 2 + 1
        freq_indices = torch.arange(n_freq).float()
        time_indices = torch.arange(n_fft).float()

        # DFT basis: cos and sin components
        angles = 2 * np.pi * freq_indices.unsqueeze(1) * time_indices.unsqueeze(0) / n_fft
        cos_basis = torch.cos(angles)  # [n_freq, n_fft]
        sin_basis = -torch.sin(angles)  # [n_freq, n_fft] (negative for conjugate)

        self.register_buffer('cos_basis', cos_basis)
        self.register_buffer('sin_basis', sin_basis)

    def forward(self, x):
        """
        Args:
            x: [batch, channels, time] waveform
        Returns:
            real: [batch, channels, freq, frames]
            imag: [batch, channels, freq, frames]
        """
        batch, channels, length = x.shape

        # Pad if center=True
        if self.center:
            pad_amount = self.n_fft // 2
            x = F.pad(x, (pad_amount, pad_amount), mode='reflect')

        # Unfold into frames
        # x: [batch, channels, padded_length]
        x_unfolded = x.unfold(2, self.n_fft, self.hop_length)  # [batch, channels, frames, n_fft]

        # Apply window
        x_windowed = x_unfolded * self.window  # [batch, channels, frames, n_fft]

        # Compute STFT using matrix multiplication (real-valued)
        # real = sum(x * cos), imag = sum(x * sin)
        real = torch.matmul(x_windowed, self.cos_basis.T)  # [batch, channels, frames, n_freq]
        imag = torch.matmul(x_windowed, self.sin_basis.T)  # [batch, channels, frames, n_freq]

        # Transpose to [batch, channels, freq, frames]
        real = real.permute(0, 1, 3, 2)
        imag = imag.permute(0, 1, 3, 2)

        return real, imag


class ISTFTProcessor(nn.Module):
    """ONNX-compatible ISTFT using real-valued operations"""

    def __init__(self, n_fft=4096, hop_length=1024, center=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center

        # Pre-compute window
        window = torch.hann_window(n_fft)
        self.register_buffer('window', window)

        # Pre-compute inverse DFT matrix
        n_freq = n_fft // 2 + 1
        freq_indices = torch.arange(n_freq).float()
        time_indices = torch.arange(n_fft).float()

        angles = 2 * np.pi * freq_indices.unsqueeze(0) * time_indices.unsqueeze(1) / n_fft
        cos_basis = torch.cos(angles)  # [n_fft, n_freq]
        sin_basis = torch.sin(angles)   # [n_fft, n_freq]

        # Scale for inverse (account for conjugate symmetry)
        scale = torch.ones(n_freq)
        scale[1:-1] = 2.0  # Double non-DC, non-Nyquist bins
        cos_basis = cos_basis * scale / n_fft
        sin_basis = sin_basis * scale / n_fft

        self.register_buffer('cos_basis', cos_basis)
        self.register_buffer('sin_basis', sin_basis)

    def forward(self, real, imag, length):
        """
        Args:
            real: [batch, channels, freq, frames]
            imag: [batch, channels, freq, frames]
            length: output length
        Returns:
            x: [batch, channels, length] waveform
        """
        batch, channels, n_freq, frames = real.shape

        # Transpose to [batch, channels, frames, freq]
        real = real.permute(0, 1, 3, 2)
        imag = imag.permute(0, 1, 3, 2)

        # Inverse DFT: x = real * cos + imag * sin
        x_frames = torch.matmul(real, self.cos_basis.T) + torch.matmul(imag, self.sin_basis.T)
        # x_frames: [batch, channels, frames, n_fft]

        # Apply window
        x_frames = x_frames * self.window

        # Overlap-add
        output_length = (frames - 1) * self.hop_length + self.n_fft
        if self.center:
            output_length -= self.n_fft  # Remove padding

        output = torch.zeros(batch, channels, output_length + self.n_fft, device=real.device)
        window_sum = torch.zeros(output_length + self.n_fft, device=real.device)

        for i in range(frames):
            start = i * self.hop_length
            output[:, :, start:start + self.n_fft] += x_frames[:, :, i, :]
            window_sum[start:start + self.n_fft] += self.window ** 2

        # Normalize by window sum
        window_sum = torch.clamp(window_sum, min=1e-8)
        output = output / window_sum

        # Remove padding if center=True
        if self.center:
            pad = self.n_fft // 2
            output = output[:, :, pad:pad + length]
        else:
            output = output[:, :, :length]

        return output


class HTDemucsONNX(nn.Module):
    """
    ONNX-exportable wrapper for HTDemucs.
    Handles STFT/ISTFT externally with real-valued operations.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_fft = model.nfft
        self.hop_length = model.nfft // 4
        self.stft = STFTProcessor(self.n_fft, self.hop_length)
        self.istft = ISTFTProcessor(self.n_fft, self.hop_length)

        # Get model parameters
        self.sources = model.sources  # ['drums', 'bass', 'other', 'vocals']
        self.audio_channels = model.audio_channels  # 2 (stereo)
        self.samplerate = model.samplerate  # 44100

    def forward(self, x):
        """
        Args:
            x: [batch, channels, time] input waveform (stereo, 44.1kHz)
        Returns:
            stems: [batch, 4, channels, time] separated stems
                   Order: drums, bass, other, vocals
        """
        batch, channels, length = x.shape

        # Normalize input
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp(min=1e-5)
        x_norm = (x - mean) / std

        # The model internally handles the hybrid processing
        # We just need to call it with the waveform
        with torch.no_grad():
            stems = self.model(x_norm)  # [batch, sources, channels, time]

        # Denormalize
        stems = stems * std.unsqueeze(1) + mean.unsqueeze(1)

        return stems


def convert_to_onnx(output_path="htdemucs_ft.onnx", model_name="htdemucs_ft"):
    """Convert HTDemucs model to ONNX format"""

    print(f"Loading {model_name} model...")

    # Load the pretrained model
    bag = get_model(model_name)

    # get_model returns a BagOfModels, extract the actual model
    if hasattr(bag, 'models'):
        print(f"  BagOfModels contains {len(bag.models)} model(s)")
        model = bag.models[0]  # Get first model from the bag
    else:
        model = bag

    model.eval()

    print(f"Model loaded: {model_name}")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Sources: {model.sources}")
    print(f"  Sample rate: {model.samplerate}")
    print(f"  Audio channels: {model.audio_channels}")

    # Create dummy input (3 seconds of stereo audio at 44.1kHz)
    duration_samples = 44100 * 3  # 3 seconds for faster export
    dummy_input = torch.randn(1, 2, duration_samples)

    print(f"\nExporting to ONNX...")
    print(f"  Input shape: {dummy_input.shape}")

    # Try multiple export strategies
    export_success = False

    # Strategy 1: Legacy ONNX export (no dynamo)
    print("\n  Strategy 1: Legacy torch.onnx.export (dynamo=False)...")
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['waveform'],
                output_names=['stems'],
                dynamo=False,  # Use legacy exporter - critical for HTDemucs!
            )
        print(f"    ✅ Legacy export succeeded!")
        export_success = True
    except Exception as e:
        print(f"    ❌ Failed: {e}")

    # Strategy 2: JIT Trace + Legacy export
    if not export_success:
        print("\n  Strategy 2: torch.jit.trace + legacy export...")
        try:
            with torch.no_grad():
                traced = torch.jit.trace(model, dummy_input)
                torch.onnx.export(
                    traced,
                    dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=['waveform'],
                    output_names=['stems'],
                    dynamo=False,  # Legacy exporter
                )
            print(f"    ✅ Traced export succeeded!")
            export_success = True
        except Exception as e:
            print(f"    ❌ Failed: {e}")

    # Strategy 3: Wrapper with legacy export
    if not export_success:
        print("\n  Strategy 3: SimpleWrapper + legacy export...")
        try:
            class SimpleWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    return self.model(x)

            wrapper = SimpleWrapper(model)
            wrapper.eval()

            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=['waveform'],
                    output_names=['stems'],
                    dynamo=False,  # Legacy exporter
                )
            print(f"    ✅ Wrapper export succeeded!")
            export_success = True
        except Exception as e:
            print(f"    ❌ Failed: {e}")

    if not export_success:
        print("\n❌ All export strategies failed!")
        print("This may be due to complex operations in HTDemucs not supported by ONNX.")
        print("\nAlternative: Use the pre-converted model from the GSoC 2025 project:")
        print("  https://github.com/adefossez/demucs/pull/10")
        return None

    # Verify the model
    print(f"\nVerifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX model is valid!")

    # Get model size
    model_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Model size: {model_size:.1f} MB")

    # Test inference with ONNX Runtime
    print(f"\nTesting ONNX Runtime inference...")
    ort_session = ort.InferenceSession(output_path)

    # Run inference
    test_input = np.random.randn(1, 2, 44100).astype(np.float32)  # 1 second
    outputs = ort_session.run(None, {'waveform': test_input})

    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {outputs[0].shape}")
    print(f"  Output stems: drums, bass, other, vocals")

    print(f"\n✅ Conversion complete!")
    print(f"\nNext steps:")
    print(f"  1. Copy {output_path} to app/src/main/assets/models/")
    print(f"  2. Rebuild the Android app")

    return output_path


def main():
    # Check if output directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "..", "app", "src", "main", "assets", "models", "htdemucs_ft.onnx")

    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 60)
    print("HTDemucs v4 Fine-Tuned to ONNX Converter")
    print("=" * 60)
    print()

    try:
        convert_to_onnx(output_path, "htdemucs_ft")
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        print("\nTrying with base htdemucs model instead...")
        try:
            convert_to_onnx(output_path.replace("_ft", ""), "htdemucs")
        except Exception as e2:
            print(f"\n❌ Error with base model too: {e2}")
            print("\nPlease ensure you have the latest demucs installed:")
            print("  pip install --upgrade demucs")
            sys.exit(1)


if __name__ == "__main__":
    main()

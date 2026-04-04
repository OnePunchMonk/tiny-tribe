"""Extract backbone features from distillation_dataset/ mp4s for Tiny-TRIBE v3 KD training.

For each clip in distillation_dataset/<stem>/:
  - video_feat:  (T, 640)  from MobileViT-S (transformers), T=5 uniformly sampled frames
  - audio_feat:  (T, 384)  from Whisper-Tiny encoder, resampled to T=5
  - text_feat:   (T, 384)  zeros (no speech in clips)
  - teacher:     (T, 20484) from preds.npy
  - subject_id:  0

Saves features/<stem>.pt

Usage:
    pip install transformers torchaudio av torch
    python tiny_tribe/extract_features_v3.py \\
        --dataset_dir ./distillation_dataset \\
        --out_dir ./features
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ── Video helpers ─────────────────────────────────────────────────────────────

def load_video_frames(mp4_path: Path, n_frames: int = 5) -> torch.Tensor:
    """Sample n_frames uniformly from mp4. Returns (n_frames, 3, 256, 256) float32 in [0,1]."""
    try:
        import av
    except ImportError:
        raise ImportError("pip install av")

    container = av.open(str(mp4_path))
    stream = container.streams.video[0]
    total_frames = stream.frames or 32  # fallback for some encoders

    # Sample uniformly
    indices = set(int(i * total_frames / n_frames) for i in range(n_frames))

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            img = frame.to_ndarray(format="rgb24")  # (H, W, 3)
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)
            # Resize to 256x256 for MobileViT
            t = F.interpolate(t.unsqueeze(0), size=(256, 256), mode="bilinear",
                              align_corners=False).squeeze(0)
            frames.append(t)
        if len(frames) == n_frames:
            break

    container.close()

    # Pad with last frame if we got fewer (very short clips)
    while len(frames) < n_frames:
        frames.append(frames[-1] if frames else torch.zeros(3, 256, 256))

    return torch.stack(frames[:n_frames])  # (T, 3, 256, 256)


def load_audio_waveform(mp4_path: Path, target_sr: int = 16000) -> torch.Tensor:
    """Extract audio from mp4 as mono waveform at target_sr. Returns (n_samples,)."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(mp4_path))
    except Exception:
        # Fallback: try via av
        try:
            import av
            container = av.open(str(mp4_path))
            audio_frames = []
            for frame in container.decode(audio=0):
                audio_frames.append(frame.to_ndarray())
            container.close()
            if not audio_frames:
                return torch.zeros(target_sr * 4)
            audio = np.concatenate(audio_frames, axis=-1)  # (channels, samples)
            waveform = torch.from_numpy(audio).float()
            sr = 44100  # typical default
        except Exception:
            return torch.zeros(target_sr * 4)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)  # (n_samples,)

    # Resample to 16kHz
    if sr != target_sr:
        try:
            import torchaudio
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        except Exception:
            pass

    return waveform


# ── Model loaders (lazy, loaded once) ─────────────────────────────────────────

_video_model = None
_video_processor = None
_whisper_encoder = None
_whisper_feature_extractor = None


def get_video_model(device):
    global _video_model, _video_processor
    if _video_model is None:
        from transformers import MobileViTModel, MobileViTImageProcessor
        print("Loading MobileViT-S...")
        _video_processor = MobileViTImageProcessor.from_pretrained(
            "apple/mobilevit-small"
        )
        _video_model = MobileViTModel.from_pretrained("apple/mobilevit-small").to(device)
        _video_model.eval()
    return _video_model, _video_processor


def get_whisper_encoder(device):
    global _whisper_encoder, _whisper_feature_extractor
    if _whisper_encoder is None:
        from transformers import WhisperModel, WhisperFeatureExtractor
        print("Loading Whisper-Tiny...")
        _whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-tiny"
        )
        whisper = WhisperModel.from_pretrained("openai/whisper-tiny").to(device)
        _whisper_encoder = whisper.encoder
        _whisper_encoder.eval()
    return _whisper_encoder, _whisper_feature_extractor


# ── Per-clip feature extraction ───────────────────────────────────────────────

@torch.no_grad()
def extract_video_features(
    frames: torch.Tensor,     # (T, 3, 256, 256) float32
    device: torch.device,
    target_T: int = 5,
) -> torch.Tensor:
    """Returns (target_T, 640) video features via MobileViT-S."""
    model, processor = get_video_model(device)

    T = frames.shape[0]
    # Process each frame individually (processor expects PIL or np arrays)
    feats = []
    for i in range(T):
        frame_np = (frames[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        inputs = processor(images=frame_np, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        out = model(pixel_values=pixel_values)
        # pooler_output: global avg pool already applied → (1, 640)
        pooled = out.pooler_output
        feats.append(pooled.squeeze(0))

    feat_tensor = torch.stack(feats)  # (T, 640)

    # Resample to target_T if needed
    if T != target_T:
        feat_tensor = F.interpolate(
            feat_tensor.T.unsqueeze(0), size=target_T, mode="linear", align_corners=False
        ).squeeze(0).T

    return feat_tensor.cpu()  # (target_T, 640)


@torch.no_grad()
def extract_audio_features(
    waveform: torch.Tensor,   # (n_samples,) at 16kHz
    device: torch.device,
    target_T: int = 5,
) -> torch.Tensor:
    """Returns (target_T, 384) audio features via Whisper-Tiny encoder."""
    encoder, feature_extractor = get_whisper_encoder(device)

    audio_np = waveform.numpy().astype(np.float32)

    # Ensure minimum length (30s is Whisper's window — it pads internally)
    inputs = feature_extractor(
        audio_np, sampling_rate=16000, return_tensors="pt"
    )
    input_features = inputs["input_features"].to(device)  # (1, 80, 3000)

    out = encoder(input_features)
    hidden = out.last_hidden_state.squeeze(0)  # (T_whisper, 384)

    # Downsample to target_T
    feat = F.interpolate(
        hidden.T.unsqueeze(0), size=target_T, mode="linear", align_corners=False
    ).squeeze(0).T  # (target_T, 384)

    return feat.cpu()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def extract_clip(
    stem: str,
    mp4_path: Path,
    preds_path: Path,
    device: torch.device,
    target_T: int = 5,
) -> dict:
    """Extract all features for a single clip."""
    # Load teacher predictions
    preds = np.load(str(preds_path)).astype(np.float32)  # (T, 20484) or (20484, T)?
    # Ensure shape (T, n_v) with T == target_T
    if preds.shape[0] != target_T and preds.shape[1] == target_T:
        preds = preds.T  # transpose if needed
    teacher = torch.from_numpy(preds)  # (target_T, 20484)

    # Video features
    frames = load_video_frames(mp4_path, n_frames=target_T)
    video_feat = extract_video_features(frames, device, target_T)

    # Audio features
    waveform = load_audio_waveform(mp4_path)
    audio_feat = extract_audio_features(waveform, device, target_T)

    # Text = zeros (no speech)
    text_feat = torch.zeros(target_T, 384, dtype=torch.float32)

    return {
        "text":       text_feat.half(),     # (T, 384)
        "audio":      audio_feat.half(),    # (T, 384)
        "video":      video_feat.half(),    # (T, 640)
        "teacher":    teacher.half(),       # (T, 20484)
        "subject_id": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./distillation_dataset")
    parser.add_argument("--out_dir",     type=str, default="./features")
    parser.add_argument("--target_T",   type=int, default=5)
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip clips already extracted")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Find all clips
    clips = sorted(dataset_dir.glob("*"))
    clips = [c for c in clips if c.is_dir() and (c / "preds.npy").exists()]
    print(f"Found {len(clips)} clips in {dataset_dir}")

    ok, failed, skipped = 0, [], 0
    t_total = time.time()

    for i, clip_dir in enumerate(clips):
        stem = clip_dir.name
        out_path = out_dir / f"{stem}.pt"

        if args.resume and out_path.exists():
            skipped += 1
            continue

        mp4_path  = clip_dir / f"{stem}.mp4"
        preds_path = clip_dir / "preds.npy"

        if not mp4_path.exists():
            print(f"  [{i+1}/{len(clips)}] SKIP {stem}: mp4 missing")
            failed.append(stem)
            continue

        t0 = time.time()
        try:
            data = extract_clip(stem, mp4_path, preds_path, device, args.target_T)
            torch.save(data, out_path)
            ok += 1
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(clips)}] {stem} — {elapsed:.1f}s")
        except Exception as e:
            print(f"  [{i+1}/{len(clips)}] ERROR {stem}: {e}")
            failed.append(stem)

    total_elapsed = time.time() - t_total
    print(f"\nDone: {ok} ok | {len(failed)} failed | {skipped} skipped | "
          f"{total_elapsed/60:.1f} min total")
    if failed:
        print("Failed:", failed[:10])

    print(f"\nFeatures saved to {out_dir}/")
    print(f"  Keys per file: text(T,384), audio(T,384), video(T,640), "
          f"teacher(T,20484), subject_id")


if __name__ == "__main__":
    main()

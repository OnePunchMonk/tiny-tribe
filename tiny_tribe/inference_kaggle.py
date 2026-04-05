"""Tiny-TRIBE v3 — Kaggle T4 Inference Script.

Paste this into a Kaggle notebook (code cells separated by # ── CELL N ── comments)
or run as a plain script after the install cell.

The numpy-core-not-found error comes from a numpy 2.x / older-compiled-extension mismatch.
We pin numpy<2.0 before importing anything else to avoid it.

Usage (Kaggle notebook):
  1. Upload your checkpoint (.ckpt) and a test video (.mp4) as a dataset.
  2. Set CKPT_PATH and VIDEO_PATH below.
  3. Run all cells.

Usage (script):
  python tiny_tribe/inference_kaggle.py \
      --ckpt /kaggle/input/my-ckpt/best.ckpt \
      --video /kaggle/input/my-video/test.mp4 \
      --subject_id 0
"""

# ── CELL 1: install (run this first, then restart the kernel) ─────────────────
# Uncomment and run this block in Kaggle, then restart the kernel before running
# the rest.
#
# import subprocess, sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
#     "numpy<2.0",          # MUST be first — fixes numpy.core not found
#     "torch>=2.1",
#     "torchvision",
#     "torchaudio",
#     "transformers>=4.40",
#     "lightning>=2.2",
#     "scipy",
#     "av",                 # PyAV for video decoding
#     "matplotlib",
# ])
# print("Done — restart the Kaggle kernel now before importing anything.")

# ── CELL 2: imports ───────────────────────────────────────────────────────────

# Fix numpy ABI before any other import touches it.
# Importing numpy first with a <2.0 pin avoids "numpy.core not found".
import numpy as np          # noqa: E402  (must be before scipy / torch extensions)
print(f"numpy {np.__version__}")

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
print(f"torch  {torch.__version__}")
print(f"CUDA   {torch.cuda.is_available()} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

# Add repo root to sys.path so we can import from tiny_tribe/
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ── CELL 3: inline model definition (no-dependency fallback) ──────────────────
# If the tiny_tribe package is importable, we use it.
# Otherwise we define only what inference needs inline.

try:
    from tiny_tribe.v3_model import TinyTribeV3
    print("Loaded TinyTribeV3 from package.")
    _MODEL_FROM_PACKAGE = True
except Exception as e:
    print(f"Could not import from package ({e}). Using inline definition.")
    _MODEL_FROM_PACKAGE = False

    # ── Minimal inline re-definition (only what inference needs) ──────────────
    import math
    from scipy.stats import gamma as scipy_gamma

    class ModalityProjector(nn.Module):
        def __init__(self, in_dim, out_dim, intermediate=768, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim), nn.Linear(in_dim, intermediate), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(intermediate, intermediate), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(intermediate, out_dim), nn.LayerNorm(out_dim),
            )
        def forward(self, x): return self.net(x)

    class TemporalMotionModule(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.conv = nn.Conv1d(dim, dim, 3, padding=1, groups=dim, bias=False)
            nn.init.zeros_(self.conv.weight)
            self.conv.weight.data[:, 0, 0] = -0.5
            self.conv.weight.data[:, 0, 2] =  0.5
        def forward(self, x):
            return x + self.conv(x.transpose(1,2)).transpose(1,2)

    class TopKRouter(nn.Module):
        def __init__(self, dim, num_experts, top_k=2):
            super().__init__()
            self.num_experts = num_experts; self.top_k = top_k
            self.gate = nn.Linear(dim, num_experts, bias=False)
        def forward(self, x):
            logits = self.gate(x)
            top_logits, top_idx = logits.topk(self.top_k, dim=-1)
            weights = F.softmax(top_logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            mask = F.one_hot(top_idx[:,:,0], self.num_experts).float()
            aux = self.num_experts * (mask.mean([0,1]) * probs.mean([0,1])).sum()
            z   = (logits**2).mean() * 1e-3
            return weights, top_idx, aux + z

    class MoEFFN(nn.Module):
        def __init__(self, dim, num_experts=8, top_k=2, ff_mult=2, dropout=0.1):
            super().__init__()
            self.top_k = top_k; self.num_experts = num_experts
            self.router = TopKRouter(dim, num_experts, top_k)
            ff = dim * ff_mult
            scale = math.sqrt(2.0 / (dim + ff))
            bw1 = torch.randn(dim, ff) * scale; bw2 = torch.randn(ff, dim) * scale
            self.w1 = nn.Parameter(bw1.unsqueeze(0).expand(num_experts,-1,-1).clone() + torch.randn(num_experts,dim,ff)*0.01)
            self.w2 = nn.Parameter(bw2.unsqueeze(0).expand(num_experts,-1,-1).clone() + torch.randn(num_experts,ff,dim)*0.01)
            self.b1 = nn.Parameter(torch.zeros(num_experts, ff))
            self.b2 = nn.Parameter(torch.zeros(num_experts, dim))
            self.drop = nn.Dropout(dropout)
        def forward(self, x):
            B, T, D = x.shape
            weights, indices, aux = self.router(x)
            out = torch.zeros_like(x)
            for k in range(self.top_k):
                idx = indices[:,:,k]; w = weights[:,:,k].unsqueeze(-1)
                h = torch.einsum("btd,btdf->btf", x, self.w1[idx]) + self.b1[idx]
                h = self.drop(F.gelu(h))
                out = out + w * (torch.einsum("btf,btfd->btd", h, self.w2[idx]) + self.b2[idx])
            return out, aux

    class MoEBlock(nn.Module):
        def __init__(self, dim, num_heads=8, num_experts=8, top_k=2, ff_mult=2, dropout=0.1, hrf_decay=False):
            super().__init__()
            self.hrf_decay = hrf_decay
            self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            self.moe  = MoEFFN(dim, num_experts, top_k, ff_mult, dropout)
            if hrf_decay:
                self.log_alpha = nn.Parameter(torch.tensor(math.log(math.log(2)/6)))
        def _hrf_bias(self, seq_len, device):
            pos = torch.arange(seq_len, device=device)
            tr_idx = pos // 3
            dist = (tr_idx.unsqueeze(0) - tr_idx.unsqueeze(1)).abs().float()
            return -self.log_alpha.exp() * dist
        def forward(self, x):
            normed = self.norm1(x)
            bias = self._hrf_bias(x.shape[1], x.device) if self.hrf_decay else None
            attn_out, _ = self.attn(normed, normed, normed, attn_mask=bias, need_weights=False)
            x = x + attn_out
            moe_out, aux = self.moe(self.norm2(x))
            return x + moe_out, aux

    class GatedModalityPool(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.gate_net = nn.Linear(dim, 3, bias=True)
            nn.init.zeros_(self.gate_net.weight); nn.init.zeros_(self.gate_net.bias)
        def forward(self, x):
            B, T3, D = x.shape; T = T3 // 3
            tokens = x.reshape(B, T, 3, D)
            gates  = F.softmax(self.gate_net(tokens.mean(dim=2)), dim=-1)
            return (gates.unsqueeze(-1) * tokens).sum(dim=2)

    class HRFConvolution(nn.Module):
        def __init__(self, dim, tr=1.5, kernel_trs=8):
            super().__init__()
            self.kernel_trs = kernel_trs
            t = np.arange(kernel_trs) * tr
            h = scipy_gamma.pdf(t,6,scale=1) - scipy_gamma.pdf(t,16,scale=1)/6
            h = h / (np.abs(h).sum()+1e-8); h = h[::-1].copy()
            self.conv = nn.Conv1d(dim, dim, kernel_trs, padding=kernel_trs-1, groups=dim, bias=False)
            ht = torch.tensor(h, dtype=torch.float32)
            self.conv.weight.data.copy_(ht.unsqueeze(0).unsqueeze(0).expand(dim,1,-1))
        def forward(self, x):
            B, T, D = x.shape
            out = self.conv(x.transpose(1,2))[:,:,:T]
            return x + out.transpose(1,2)

    class FiLMConditioner(nn.Module):
        def __init__(self, dim, n_subjects):
            super().__init__()
            self.gamma = nn.Embedding(n_subjects, dim); self.beta = nn.Embedding(n_subjects, dim)
            nn.init.ones_(self.gamma.weight); nn.init.zeros_(self.beta.weight)
        def forward(self, x, subject_id):
            return self.gamma(subject_id).unsqueeze(1)*x + self.beta(subject_id).unsqueeze(1)

    class TinyTribeV3(nn.Module):
        def __init__(self, text_dim=384, audio_dim=384, video_dim=640,
                     hidden_dim=512, proj_inter=768, num_layers=4, num_heads=8,
                     num_experts=8, top_k=2, ff_mult=2, dropout=0.1, max_seq_len=2048,
                     n_vertices=400, n_subjects=3, low_rank_dim=256, tr=1.5,
                     modality_dropout=0.3, aux_loss_weight=0.01, stoch_depth_max=0.2):
            super().__init__()
            self.hidden_dim=hidden_dim; self.num_layers=num_layers
            self.modality_dropout=modality_dropout; self.aux_loss_weight=aux_loss_weight
            self.stoch_depth_max=stoch_depth_max
            self.text_proj  = ModalityProjector(text_dim,  hidden_dim, proj_inter, dropout)
            self.audio_proj = ModalityProjector(audio_dim, hidden_dim, proj_inter, dropout)
            self.video_proj = ModalityProjector(video_dim, hidden_dim, proj_inter, dropout)
            self.video_motion = TemporalMotionModule(hidden_dim)
            self.modality_embed = nn.Embedding(3, hidden_dim)
            self.text_time_embed  = nn.Embedding(max_seq_len, hidden_dim)
            self.audio_time_embed = nn.Embedding(max_seq_len, hidden_dim)
            self.video_time_embed = nn.Embedding(max_seq_len, hidden_dim)
            self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len*3, hidden_dim)*0.02)
            self.layers = nn.ModuleList([
                MoEBlock(hidden_dim, num_heads, num_experts, top_k, ff_mult, dropout, hrf_decay=(i<2))
                for i in range(num_layers)
            ])
            self.norm = nn.LayerNorm(hidden_dim)
            self.gate_pool  = GatedModalityPool(hidden_dim)
            self.hrf_conv   = HRFConvolution(hidden_dim, tr=tr, kernel_trs=8)
            self.output_norm = nn.LayerNorm(hidden_dim)
            self.output_mlp  = nn.Linear(hidden_dim, hidden_dim)
            self.film        = FiLMConditioner(hidden_dim, n_subjects)
            self.vertex_proj = nn.Linear(hidden_dim, n_vertices, bias=False)
            self.feat_proj   = nn.Linear(hidden_dim, 1152, bias=False)

        def forward(self, text_feat, audio_feat, video_feat, subject_id, n_out_trs=None):
            B, T, _ = text_feat.shape; device = text_feat.device
            tp = self.text_proj(text_feat)
            ap = self.audio_proj(audio_feat)
            vp = self.video_proj(video_feat)
            vp = self.video_motion(vp)
            t_idx = torch.arange(T, device=device)
            tp = tp + self.text_time_embed(t_idx)
            ap = ap + self.audio_time_embed(t_idx)
            vp = vp + self.video_time_embed(t_idx)
            mod = self.modality_embed(torch.arange(3, device=device))
            tp = tp + mod[0]; ap = ap + mod[1]; vp = vp + mod[2]
            T_max = max(tp.shape[1], ap.shape[1], vp.shape[1])
            tp = self._align(tp, T_max); ap = self._align(ap, T_max); vp = self._align(vp, T_max)
            x = torch.stack([tp, ap, vp], dim=2).reshape(B, T_max*3, self.hidden_dim)
            x = x + self.pos_embed[:, :x.shape[1]]
            total_aux = torch.tensor(0.0, device=device)
            for layer in self.layers:
                x_out, aux = layer(x); total_aux = total_aux + aux; x = x_out
            x = self.norm(x)
            fused = self.gate_pool(x)
            fused = self.hrf_conv(fused)
            out = F.gelu(self.output_mlp(self.output_norm(fused)))
            out = self.film(out, subject_id)
            vertices = self.vertex_proj(out).transpose(1,2)  # (B, n_v, T)
            if n_out_trs is not None and n_out_trs != T_max:
                vertices = F.adaptive_avg_pool1d(vertices, n_out_trs)
            return {"prediction": vertices, "fusion_feat": fused,
                    "aux_loss": total_aux * self.aux_loss_weight}

        def _align(self, x, target):
            if x.shape[1] == target: return x
            return F.interpolate(x.transpose(1,2), size=target, mode="linear", align_corners=False).transpose(1,2)


# ── CELL 4: checkpoint loader ─────────────────────────────────────────────────

def load_model_from_checkpoint(ckpt_path: str, device: torch.device) -> TinyTribeV3:
    """Load TinyTribeV3 from a Lightning .ckpt or a bare state_dict .pt/.pth."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Lightning checkpoint: state_dict is nested under 'state_dict'
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        hparams = ckpt.get("hyper_parameters", {})
    elif isinstance(ckpt, dict) and any(k.startswith("model.") for k in ckpt):
        state = ckpt
        hparams = {}
    else:
        # Bare state dict (no 'model.' prefix)
        state = {"model." + k: v for k, v in ckpt.items()}
        hparams = {}

    # Strip 'model.' prefix (Lightning wraps model in a LightningModule)
    model_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            model_state[k[len("model."):]] = v
        else:
            model_state[k] = v

    # Read hyperparams from checkpoint or fall back to defaults
    def hp(key, default):
        return hparams.get(key, default)

    model = TinyTribeV3(
        hidden_dim  = hp("hidden_dim",   256),
        num_layers  = hp("num_layers",   2),
        num_heads   = hp("num_heads",    4),
        num_experts = hp("num_experts",  4),
        top_k       = hp("top_k",        2),
        ff_mult     = hp("ff_mult",      2),
        dropout     = 0.0,           # no dropout at inference
        modality_dropout = 0.0,
        n_vertices  = hp("n_vertices",   400),
        n_subjects  = hp("n_subjects",   1),
    )

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"  [warn] missing keys  ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  [warn] unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded — {n_params/1e6:.2f}M params on {device}")
    return model


# ── CELL 5: feature extraction helpers ───────────────────────────────────────

def _load_video_frames(mp4_path: Path, n_frames: int = 5) -> "np.ndarray":
    """Decode n_frames uniformly from mp4. Returns uint8 (T, H, W, 3)."""
    try:
        import av
    except ImportError:
        raise ImportError("Run: pip install av")

    container = av.open(str(mp4_path))
    stream = container.streams.video[0]
    total = stream.frames or 64
    indices = set(int(i * total / n_frames) for i in range(n_frames))
    frames = []
    for fi, frame in enumerate(container.decode(video=0)):
        if fi in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) == n_frames:
            break
    container.close()
    while len(frames) < n_frames:
        frames.append(frames[-1] if frames else np.zeros((256, 256, 3), dtype=np.uint8))
    return np.stack(frames[:n_frames])   # (T, H, W, 3)


@torch.no_grad()
def extract_video_features(mp4_path: Path, device: torch.device,
                            n_frames: int = 5) -> torch.Tensor:
    """Returns (n_frames, 640) float32 on CPU."""
    from transformers import MobileViTModel, MobileViTImageProcessor
    processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")
    vmodel    = MobileViTModel.from_pretrained("apple/mobilevit-small").to(device).eval()

    frames_np = _load_video_frames(mp4_path, n_frames)   # (T, H, W, 3) uint8
    feats = []
    for i in range(n_frames):
        inp = processor(images=frames_np[i], return_tensors="pt")
        pv  = inp["pixel_values"].to(device)
        out = vmodel(pixel_values=pv)
        feats.append(out.pooler_output.squeeze(0).cpu())
    return torch.stack(feats)   # (T, 640)


@torch.no_grad()
def extract_audio_features(mp4_path: Path, device: torch.device,
                             target_T: int = 5) -> torch.Tensor:
    """Returns (target_T, 384) float32 on CPU."""
    import torchaudio
    from transformers import WhisperModel, WhisperFeatureExtractor

    fe      = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
    whisper = WhisperModel.from_pretrained("openai/whisper-tiny").to(device)
    encoder = whisper.encoder.eval()

    try:
        waveform, sr = torchaudio.load(str(mp4_path))
    except Exception:
        waveform, sr = torch.zeros(1, 16000 * 4), 16000

    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    audio_np = waveform.squeeze(0).numpy().astype(np.float32)
    inputs   = fe(audio_np, sampling_rate=16000, return_tensors="pt")
    hidden   = encoder(inputs["input_features"].to(device)).last_hidden_state  # (1, T', 384)
    hidden   = hidden.squeeze(0)  # (T', 384)

    # Downsample to target_T
    feat = F.interpolate(
        hidden.T.unsqueeze(0), size=target_T, mode="linear", align_corners=False
    ).squeeze(0).T.cpu()   # (target_T, 384)
    return feat


def build_input_from_video(mp4_path: str, device: torch.device,
                            n_frames: int = 5) -> dict:
    """Extract all three modalities from a single mp4 file."""
    mp4 = Path(mp4_path)
    print(f"Extracting video features...")
    video = extract_video_features(mp4, device, n_frames)          # (T, 640)
    print(f"Extracting audio features...")
    audio = extract_audio_features(mp4, device, target_T=n_frames) # (T, 384)
    text  = torch.zeros(n_frames, 384, dtype=torch.float32)        # (T, 384) zeros

    return {
        "text":  text.unsqueeze(0),   # (1, T, 384)
        "audio": audio.unsqueeze(0),  # (1, T, 384)
        "video": video.unsqueeze(0),  # (1, T, 640)
    }


# ── CELL 6: inference ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: TinyTribeV3, inputs: dict, subject_id: int,
                  device: torch.device) -> np.ndarray:
    """Run forward pass. Returns (n_vertices, T) numpy array."""
    text  = inputs["text"].float().to(device)
    audio = inputs["audio"].float().to(device)
    video = inputs["video"].float().to(device)
    sid   = torch.tensor([subject_id], dtype=torch.long, device=device)

    out   = model(text, audio, video, sid)
    pred  = out["prediction"].squeeze(0).cpu().numpy()  # (n_vertices, T)
    return pred


def save_and_visualise(pred: np.ndarray, out_dir: str = "/kaggle/working"):
    """Save prediction as .npy and plot a quick heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / "tinytribe_prediction.npy"
    np.save(str(npy_path), pred)
    print(f"Saved prediction → {npy_path}  shape={pred.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(pred, aspect="auto", cmap="RdBu_r")
    axes[0].set_xlabel("TR"); axes[0].set_ylabel("vertex")
    axes[0].set_title(f"Predicted BOLD — ({pred.shape[0]} vertices × {pred.shape[1]} TRs)")
    plt.colorbar(axes[0].images[0], ax=axes[0])

    # Per-vertex mean activation
    axes[1].plot(pred.mean(axis=1))
    axes[1].set_xlabel("vertex"); axes[1].set_ylabel("mean predicted activation")
    axes[1].set_title("Mean activation per vertex")

    plt.tight_layout()
    fig_path = out_dir / "tinytribe_prediction.png"
    plt.savefig(str(fig_path), dpi=150)
    plt.close()
    print(f"Saved figure  → {fig_path}")
    return pred


# ── CELL 7: main ──────────────────────────────────────────────────────────────

def main():
    # ── Config: edit these three lines ─────────────────────────────────────────
    CKPT_PATH  = "/kaggle/input/tinytribe-ckpt/best.ckpt"   # your uploaded ckpt
    VIDEO_PATH = "/kaggle/input/test-video/test.mp4"        # mp4 to run on
    SUBJECT_ID = 0                                           # 0-indexed subject
    OUT_DIR    = "/kaggle/working"
    N_FRAMES   = 5   # temporal resolution T
    # ───────────────────────────────────────────────────────────────────────────

    # Allow CLI overrides
    parser = argparse.ArgumentParser(description="TinyTribeV3 inference on Kaggle T4")
    parser.add_argument("--ckpt",       default=CKPT_PATH)
    parser.add_argument("--video",      default=VIDEO_PATH)
    parser.add_argument("--subject_id", type=int, default=SUBJECT_ID)
    parser.add_argument("--out_dir",    default=OUT_DIR)
    parser.add_argument("--n_frames",   type=int, default=N_FRAMES)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"TinyTribeV3 Inference")
    print(f"  device     : {device}")
    print(f"  checkpoint : {args.ckpt}")
    print(f"  video      : {args.video}")
    print(f"  subject_id : {args.subject_id}")
    print(f"  n_frames   : {args.n_frames}")
    print(f"{'='*60}\n")

    # 1. Load model
    model = load_model_from_checkpoint(args.ckpt, device)

    # 2. Extract features
    inputs = build_input_from_video(args.video, device, args.n_frames)

    # 3. Inference
    print("Running inference...")
    pred = run_inference(model, inputs, args.subject_id, device)
    print(f"Prediction shape: {pred.shape}  (n_vertices × T)")

    # 4. Save & plot
    save_and_visualise(pred, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()

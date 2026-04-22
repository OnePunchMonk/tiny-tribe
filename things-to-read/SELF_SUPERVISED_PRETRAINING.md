# Self-Supervised Multimodal Pre-Training for Tiny-TRIBE v3

## Overview

This document details Phase 1 of the v3 strategy: pre-training the fusion
transformer on unlabeled multimodal data without any teacher inferences.

The goal is to learn **cross-modal temporal representations** that transfer
well to brain encoding, using only frozen tiny backbone features.

---

## 1. Why Self-Supervised Pre-Training Works Here

### The brain encoding task decomposes into two sub-problems:

**Sub-problem A: Multimodal fusion**
- How does speech relate to the visual scene at the same moment?
- How does narrative context build over 10-30 seconds?
- What happens when modalities conflict (e.g., sarcasm — tone vs words)?

**Sub-problem B: Brain mapping**
- Given a fused multimodal representation, which cortical vertices activate?
- How does the hemodynamic delay affect the mapping?
- How do individual subjects differ?

Sub-problem A requires **diverse multimodal data** but no brain data.
Sub-problem B requires brain data but is a **simpler mapping** once A is solved.

Current Tiny-TRIBE v2 tries to learn both simultaneously from limited
teacher-labeled data. v3 separates them.

### Evidence this works:

1. **data2vec** (Meta, 2022): Self-supervised pre-training on speech, vision, and text
   with a shared framework. Pre-trained representations transfer to downstream tasks
   with 10x less labeled data.

2. **VideoMAE** (2022): Masked video pre-training. The pre-trained encoder achieves
   87.4% on Kinetics-400 with only 1% of labels — demonstrating that self-supervised
   features are nearly as good as supervised ones.

3. **Brain alignment studies**: Pre-trained language models (even without any brain data)
   already predict 30-50% of brain variance. The representations learned from text
   alone are partially brain-aligned. Adding multimodal pre-training should push this further.

---

## 2. Pre-Training Tasks in Detail

### Task 1: Masked Modality Reconstruction (MMR) — Weight: 0.4

**Intuition:** If you can predict what the video looks like from just the audio and
transcript, you understand cross-modal relationships.

**Implementation:**

```python
class MaskedModalityReconstruction(nn.Module):
    """
    Mask one modality entirely, predict its projected features
    from the remaining modalities.
    """
    def __init__(self, hidden_dim=512):
        super().__init__()
        # One reconstruction head per modality
        self.heads = nn.ModuleDict({
            'text': nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            'audio': nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            'video': nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
        })

    def forward(self, transformer_output, original_projected, masked_modality):
        """
        Args:
            transformer_output: (B, T*3, 512) from fusion transformer
            original_projected: dict of (B, T, 512) per modality (before masking)
            masked_modality: str, which modality was masked
        """
        B, T3, D = transformer_output.shape
        T = T3 // 3

        # Reshape to (B, T, 3, D), extract the masked modality's positions
        reshaped = transformer_output.view(B, T, 3, D)
        modality_idx = {'text': 0, 'audio': 1, 'video': 2}[masked_modality]
        masked_positions = reshaped[:, :, modality_idx, :]  # (B, T, D)

        # Predict original features
        predicted = self.heads[masked_modality](masked_positions)
        target = original_projected[masked_modality].detach()

        # MSE + cosine loss (cosine preserves direction, MSE preserves magnitude)
        mse_loss = F.mse_loss(predicted, target)
        cos_loss = 1 - F.cosine_similarity(predicted, target, dim=-1).mean()

        return 0.7 * mse_loss + 0.3 * cos_loss
```

**Masking strategy per batch:**
- 50% of samples: mask one random modality
- 25% of samples: mask two modalities (harder, forces single-modality understanding)
- 25% of samples: no masking (reconstruction from full context as baseline)

**What this teaches the fusion model:**
- Text→Video: "the word 'explosion' predicts bright, high-motion video features"
- Audio→Text: "this speech pattern predicts these semantic features"
- Video→Audio: "a person's mouth moving predicts speech-like audio features"

---

### Task 2: Cross-Modal Contrastive Learning (CMC) — Weight: 0.2

**Intuition:** Representations at the same timestamp across modalities should be
more similar than representations at different timestamps.

**Implementation:**

```python
class CrossModalContrastive(nn.Module):
    """
    InfoNCE loss: aligned multimodal representations at the same
    timestamp are positive pairs, different timestamps are negatives.
    """
    def __init__(self, hidden_dim=512, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128)  # Project to smaller space for contrastive
        )

    def forward(self, transformer_output):
        """
        Args:
            transformer_output: (B, T*3, 512)
        """
        B, T3, D = transformer_output.shape
        T = T3 // 3

        # Pool across modalities at each timestep
        reshaped = transformer_output.view(B, T, 3, D)
        pooled = reshaped.mean(dim=2)  # (B, T, D)

        # Project to contrastive space
        projected = self.projection(pooled)  # (B, T, 128)
        projected = F.normalize(projected, dim=-1)

        # Flatten batch and time: each (sample, timestep) is a data point
        flat = projected.view(B * T, -1)  # (B*T, 128)

        # Similarity matrix
        sim = flat @ flat.T / self.temperature  # (B*T, B*T)

        # Positive pairs: same sample, adjacent timesteps (within ±2 TRs)
        labels = torch.arange(B * T, device=sim.device)

        # Mask: positive if same sample and |t1-t2| <= 2
        sample_ids = torch.arange(B).unsqueeze(1).expand(B, T).reshape(-1)
        time_ids = torch.arange(T).unsqueeze(0).expand(B, T).reshape(-1)

        same_sample = (sample_ids.unsqueeze(1) == sample_ids.unsqueeze(0))
        close_time = (time_ids.unsqueeze(1) - time_ids.unsqueeze(0)).abs() <= 2

        positives = same_sample & close_time
        positives.fill_diagonal_(False)  # Don't count self

        # InfoNCE with multiple positives
        # For each anchor, average the positive similarities
        pos_sim = (sim * positives.float()).sum(dim=1) / positives.float().sum(dim=1).clamp(min=1)
        neg_mask = ~same_sample  # Negatives: different samples
        neg_sim = sim.masked_fill(~neg_mask, float('-inf'))

        # Simplified: use cross-entropy with the positive as target
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(B * T, dtype=torch.long, device=sim.device)

        return F.cross_entropy(logits, labels)
```

**What this teaches:**
- Temporally aligned multimodal features cluster together
- Features from different events are pushed apart
- The representation space becomes structured by content, not modality

---

### Task 3: Next-TR Feature Prediction (NTP) — Weight: 0.2

**Intuition:** If you can predict what the multimodal representation will be
1.5 seconds from now, you understand temporal dynamics — exactly what the brain does.

```python
class NextTRPrediction(nn.Module):
    """
    Predict the fused representation at t+1 from the sequence up to t.
    Uses a causal (unidirectional) attention mask.
    """
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, transformer, interleaved_tokens, pos_emb):
        """
        Run transformer twice:
        1. With causal mask (for prediction)
        2. Without mask (for target)
        """
        B, T3, D = interleaved_tokens.shape
        T = T3 // 3

        # Causal forward pass
        causal_mask = torch.triu(
            torch.ones(T3, T3, device=interleaved_tokens.device),
            diagonal=1
        ).bool()
        causal_out = transformer(interleaved_tokens + pos_emb, src_mask=causal_mask)

        # Pool modalities at each timestep
        causal_pooled = causal_out.view(B, T, 3, D).mean(dim=2)  # (B, T, D)

        # Non-causal forward pass (target — detached)
        with torch.no_grad():
            full_out = transformer(interleaved_tokens + pos_emb)
            full_pooled = full_out.view(B, T, 3, D).mean(dim=2)  # (B, T, D)

        # Predict t+1 from causal representation at t
        predicted_next = self.predictor(causal_pooled[:, :-1, :])  # (B, T-1, D)
        target_next = full_pooled[:, 1:, :].detach()               # (B, T-1, D)

        mse = F.mse_loss(predicted_next, target_next)
        cos = 1 - F.cosine_similarity(predicted_next, target_next, dim=-1).mean()

        return 0.5 * mse + 0.5 * cos
```

**What this teaches:**
- Temporal dynamics: how scenes evolve, how speech unfolds
- Anticipation: what comes next given current context
- Hemodynamic-compatible representations: fMRI inherently reflects past+present

---

### Task 4: Temporal Order Prediction (TOP) — Weight: 0.1

**Intuition:** A model that understands temporal structure should detect when
segments are out of order.

```python
class TemporalOrderPrediction(nn.Module):
    """
    Shuffle temporal segments, predict the correct permutation.
    Simplified: predict if a pair of segments is in the correct order.
    """
    def __init__(self, hidden_dim=512, n_segments=4):
        super().__init__()
        self.n_segments = n_segments
        self.order_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)  # Binary: correct or swapped
        )

    def forward(self, transformer_output):
        B, T3, D = transformer_output.shape
        T = T3 // 3

        # Pool modalities
        pooled = transformer_output.view(B, T, 3, D).mean(dim=2)  # (B, T, D)

        # Split into segments and get segment representations
        seg_len = T // self.n_segments
        segments = []
        for i in range(self.n_segments):
            seg = pooled[:, i*seg_len:(i+1)*seg_len, :].mean(dim=1)  # (B, D)
            segments.append(seg)
        segments = torch.stack(segments, dim=1)  # (B, n_seg, D)

        # For each pair of adjacent segments, predict if they're in order
        loss = 0
        count = 0
        for i in range(self.n_segments - 1):
            # Correct order
            pair_correct = torch.cat([segments[:, i], segments[:, i+1]], dim=-1)
            logits_correct = self.order_head(pair_correct)
            labels_correct = torch.ones(B, dtype=torch.long, device=pooled.device)

            # Swapped order
            pair_swapped = torch.cat([segments[:, i+1], segments[:, i]], dim=-1)
            logits_swapped = self.order_head(pair_swapped)
            labels_swapped = torch.zeros(B, dtype=torch.long, device=pooled.device)

            loss += F.cross_entropy(logits_correct, labels_correct)
            loss += F.cross_entropy(logits_swapped, labels_swapped)
            count += 2

        return loss / count
```

---

## 3. Data Pipeline

### Feature Extraction (Offline, One-Time)

Extract and cache tiny backbone features for all training videos:

```python
class FeatureExtractor:
    """
    Extract features from frozen tiny backbones.
    Run once per video, cache to disk.
    """
    def __init__(self):
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.audio_model = WhisperTinyEncoder()  # encoder-only
        self.video_model = MobileViTFeatureExtractor()

    def extract(self, video_path, output_dir):
        # Extract at 2 Hz (matching TRIBE v2 feature rate)

        # Text: WhisperX transcription → sentence embeddings at 2Hz
        transcript = whisperx_transcribe(video_path)
        text_events = align_words_to_2hz(transcript)
        text_features = self.text_model.encode(text_events)  # (T, 384)

        # Audio: 16kHz waveform → Whisper-Tiny encoder features
        audio = load_audio(video_path, sr=16000)
        audio_features = self.audio_model(audio)  # (T, 384)

        # Video: frames at 2fps → MobileViT-S features
        frames = extract_frames(video_path, fps=2)
        video_features = self.video_model(frames)  # (T, 640)

        # Save to disk
        torch.save({
            'text': text_features,
            'audio': audio_features,
            'video': video_features,
            'duration_s': len(audio) / 16000,
            'n_timesteps': len(text_features),
        }, f"{output_dir}/{Path(video_path).stem}.pt")
```

**Cost estimate for 1000 hours of video:**
- Text (MiniLM, 22.7M): ~20x realtime → ~50h of compute
- Audio (Whisper-Tiny, 39M): ~15x realtime → ~67h of compute
- Video (MobileViT-S, 5.6M): ~30x realtime → ~33h of compute
- **Total: ~150 GPU-hours on T4** (or ~24h with 6-8 parallel workers)
- **Storage: ~50-100GB** for cached features

This is much cheaper than running TRIBE v2 (which would take ~500 GPU-hours
for the same amount of video).

### Dataset Class

```python
class MultimodalPretrainDataset(Dataset):
    """
    Loads cached backbone features for self-supervised pre-training.
    Returns fixed-length segments with random crops.
    """
    def __init__(self, feature_dir, segment_length=60, stride=30):
        self.files = sorted(Path(feature_dir).glob("*.pt"))
        self.segment_length = segment_length  # in timesteps (at 2Hz = 30s)
        self.stride = stride

        # Build index: (file_idx, start_timestep)
        self.index = []
        for i, f in enumerate(self.files):
            meta = torch.load(f, map_location='cpu', weights_only=True)
            n_t = meta['n_timesteps']
            for start in range(0, n_t - segment_length, stride):
                self.index.append((i, start))

    def __getitem__(self, idx):
        file_idx, start = self.index[idx]
        data = torch.load(self.files[file_idx], map_location='cpu', weights_only=True)

        end = start + self.segment_length
        return {
            'text': data['text'][start:end],      # (T, 384)
            'audio': data['audio'][start:end],     # (T, 384)
            'video': data['video'][start:end],     # (T, 640)
        }
```

---

## 4. Training Loop

```python
class Phase1Trainer:
    def __init__(self, model, tasks, optimizer, scheduler):
        self.model = model  # TinyTribeMoE (without output head)
        self.tasks = tasks  # dict of task modules
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_step(self, batch):
        text, audio, video = batch['text'], batch['audio'], batch['video']

        # Project through modality projectors
        text_proj = self.model.text_projector(text)    # (B, T, 512)
        audio_proj = self.model.audio_projector(audio)  # (B, T, 512)
        video_proj = self.model.video_projector(video)  # (B, T, 512)

        original_projected = {
            'text': text_proj.detach(),
            'audio': audio_proj.detach(),
            'video': video_proj.detach()
        }

        # Choose masking strategy
        r = random.random()
        if r < 0.5:
            # Mask one modality
            masked = random.choice(['text', 'audio', 'video'])
            if masked == 'text': text_proj = torch.zeros_like(text_proj)
            elif masked == 'audio': audio_proj = torch.zeros_like(audio_proj)
            else: video_proj = torch.zeros_like(video_proj)
        elif r < 0.75:
            # Mask two modalities
            keep = random.choice(['text', 'audio', 'video'])
            if keep != 'text': text_proj = torch.zeros_like(text_proj)
            if keep != 'audio': audio_proj = torch.zeros_like(audio_proj)
            if keep != 'video': video_proj = torch.zeros_like(video_proj)
            masked = 'two'
        else:
            masked = None

        # Interleave and run through transformer
        interleaved = self.model.interleave(text_proj, audio_proj, video_proj)
        interleaved = interleaved + self.model.pos_embedding[:, :interleaved.size(1)]
        output = self.model.transformer(interleaved)

        # Compute losses
        losses = {}

        if masked and masked != 'two':
            losses['mmr'] = self.tasks['mmr'](output, original_projected, masked)
        elif masked == 'two':
            # Reconstruct both masked modalities
            for m in ['text', 'audio', 'video']:
                if m != keep:
                    losses[f'mmr_{m}'] = self.tasks['mmr'](output, original_projected, m)

        losses['cmc'] = self.tasks['cmc'](output)
        losses['ntp'] = self.tasks['ntp'](
            self.model.transformer, interleaved, self.model.pos_embedding
        )
        losses['top'] = self.tasks['top'](output)
        losses['aux'] = self.model.get_aux_loss()

        # Weighted total
        total = (0.4 * losses.get('mmr', 0)
               + 0.2 * losses['cmc']
               + 0.2 * losses['ntp']
               + 0.1 * losses['top']
               + 0.01 * losses['aux'])

        # Handle multi-modality MMR
        if masked == 'two':
            mmr_losses = [v for k, v in losses.items() if k.startswith('mmr_')]
            total += 0.4 * sum(mmr_losses) / len(mmr_losses)

        return total, losses
```

---

## 5. Hyperparameters and Schedule

### Recommended Configuration

```yaml
# Phase 1: Self-supervised pre-training
phase1:
  data:
    sources:
      - howto100m_subset: 500h  # Diverse instructional videos
      - librispeech: 200h       # Clean speech + text (no video — use modality dropout)
      - vggsound_subset: 200h   # Video + audio events
      - posted_videos: ~10h     # Domain-specific
    segment_length: 60          # timesteps at 2Hz = 30 seconds
    stride: 30                  # 50% overlap
    batch_size: 32

  model:
    hidden_dim: 512
    n_layers: 4
    n_heads: 8
    n_experts: 8
    top_k: 2
    projector_layers: 3
    projector_intermediate: 768

  training:
    epochs: 25
    lr: 3e-4
    scheduler: cosine_with_warmup
    warmup_fraction: 0.05
    weight_decay: 0.01
    gradient_clip: 1.0
    router_warmup_steps: 1000
    aux_loss_warmup: [0.1, 0.01, 1000]  # start, end, steps

  loss_weights:
    mmr: 0.4
    cmc: 0.2
    ntp: 0.2
    top: 0.1
    aux: 0.01

  masking:
    one_modality_prob: 0.5
    two_modality_prob: 0.25
    no_mask_prob: 0.25

  augmentation:
    time_warp: 0.1            # ±10% speed variation
    feature_noise_std: 0.02   # Gaussian noise on backbone features
    segment_shuffle_prob: 0.05 # Occasionally shuffle within segment
```

### Expected Timeline

| Step | GPU-hours (T4) | Wall time (1 GPU) |
|------|---------------|-------------------|
| Feature extraction (1000h video) | ~150h | ~6 days |
| Pre-training (25 epochs) | ~15-20h | ~1 day |
| **Total Phase 1** | **~170h** | **~7 days** |

Compare: Running TRIBE v2 teacher on 1000h = ~500 GPU-hours.
We get more data for less compute, and no teacher is involved.

---

## 6. Evaluation During Pre-Training

### Proxy Metrics (No Brain Data Needed)

| Metric | What it measures | Target |
|--------|-----------------|--------|
| MMR reconstruction loss | Cross-modal prediction quality | Decreasing, plateau by epoch 15 |
| CMC retrieval accuracy | Temporal alignment quality | >80% R@5 for aligned pairs |
| NTP prediction cosine sim | Temporal prediction quality | >0.6 |
| TOP accuracy | Temporal order understanding | >85% |
| Expert utilization entropy | MoE health | >1.5 (of max 2.08) |
| Per-expert token fraction | MoE balance | 10-15% each |

### Downstream Validation (Optional, Requires Brain Data)

After pre-training, train a **linear probe** on the frozen fusion features:
- Freeze the entire pre-trained model
- Train only: Linear(512, 20484) on Algonauts 2025 training data
- If this probe achieves Pearson r > 0.15, the representations are brain-relevant
- Compare to a random-init baseline probe

---

## 7. What to Save After Phase 1

```
checkpoints/
  phase1_final.pt              # Full model state dict
  phase1_best_mmr.pt           # Best masked modality reconstruction
  phase1_epoch_10.pt           # Intermediate checkpoint
  phase1_config.yaml           # Hyperparameters
  phase1_metrics.json          # Training curves

features/                       # Keep cached backbone features
  howto100m/*.pt
  librispeech/*.pt
  vggsound/*.pt
  posted/*.pt
```

The pre-trained model (minus task heads) becomes the initialization for Phase 2.
The cached backbone features are reused in Phase 2 (no re-extraction needed).

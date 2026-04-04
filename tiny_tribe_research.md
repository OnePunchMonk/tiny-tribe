# Tiny-TRIBE: Compressing TRIBE v2 for Browser & Edge Deployment

## Executive Summary

TRIBE v2 totals ~4.7B parameters (frozen backbones) + ~70-130M trainable. Recent research shows brain alignment saturates early — 3B models match 7-14B in neural predictivity. This means substantial compression is viable.

**Target: <120MB browser-deployable model achieving 80-90%+ of full TRIBE v2 performance.**

---

## 1. Backbone Replacement Options

### Text (Current: LLaMA 3.2-3B, 3,072D)

| Model | Params | Output Dim | Size | ONNX | Browser | Notes |
|-------|--------|------------|------|------|---------|-------|
| **all-MiniLM-L6-v2** | 22.7M | 384 | 85MB | Yes | Yes | Sentence embeddings, smallest. May lose word-level temporal alignment |
| DistilBERT | 66M | 768 | 250MB | Yes | Yes | Good general-purpose, 6 layers |
| TinyBERT-4L | 14M | 312 | 50MB | Yes | Yes | Extreme compression, weak semantics |
| Qwen-0.5B | 500M | 896 | 1GB (250MB INT4) | Yes | Yes | Best quality of small options, multilingual |

**Recommendation:** all-MiniLM-L6-v2 for PoC, Qwen-0.5B INT4 as fallback for better semantics.

### Audio (Current: Wav2Vec-BERT 2.0, ~600M, 1,024D)

| Model | Params | Output Dim | Size | ONNX | Browser | Notes |
|-------|--------|------------|------|------|---------|-------|
| **Whisper-Tiny** | 39M | 384 | 150MB | Yes | Yes | Robust speech, multilingual. Use encoder only |
| Wav2Vec2-Base | 95M | 768 | 370MB | Yes | Yes | Self-supervised, good speech representations |
| HuBERT-Base | 95M | 768 | 370MB | Yes | Yes | Similar to Wav2Vec2, different training objective |
| AST-Tiny | 6.1M | 768 | 25MB | Yes | Yes | Spectrogram-based, better on sound events than speech |

**Recommendation:** Whisper-Tiny encoder-only (39M, 150MB).

### Video (Current: V-JEPA2-ViT-G, 1.1B, 1,536D)

| Model | Params | Output Dim | Size | ONNX | Browser | Notes |
|-------|--------|------------|------|------|---------|-------|
| **MobileViT-S** | 5.6M | 256-512 | 22MB | Yes | Yes | Designed for mobile, fast |
| EfficientNet-B0 | 5.3M | 1,280 | 20MB | Yes | Yes | Strong accuracy/efficiency tradeoff |
| TinyViT-11M | 11M | 256 | 45MB | Yes | Partial | Pure ViT |
| CLIP-ViT-B/16 | 86M | 512 | 330MB | Yes | Yes | Semantically rich but large |

**Recommendation:** MobileViT-S (5.6M, 22MB). Note: single-frame models lose temporal coherence — consider 3D-MobileNet-v2 (~2.2M) for clip-level features.

### Combined Tiny Backbone Stack

| Modality | Model | Params | Size | Output Dim |
|----------|-------|--------|------|------------|
| Text | all-MiniLM-L6-v2 | 22.7M | 85MB | 384 |
| Audio | Whisper-Tiny | 39M | 150MB | 384 |
| Video | MobileViT-S | 5.6M | 22MB | 256 |
| **Total** | | **67.3M** | **~260MB** | **~1,024 fused** |

Compare: TRIBE v2 backbones = 4.7B params, ~10GB.

---

## 2. Fusion Model Compression

### Current Fusion Architecture
```
Per-modality projectors (3x MLP) → each to 384D
Concatenate → (B, T, 1152)
+ Positional embeddings (1024 max)
8-layer Transformer (1152 hidden, 8 heads)
Low-rank head: 1152 → 2048
SubjectLayers: 2048 → 20,484 vertices
```

### Compression Strategies

#### A. Reduce Depth & Width

| Config | Layers | Hidden | Heads | Transformer Params | Expected Accuracy |
|--------|--------|--------|-------|-------------------|-------------------|
| Full | 8 | 1152 | 8 | ~15M | 100% (baseline) |
| **Tiny-1** | 4 | 512 | 4 | ~3M | 85-90% |
| Tiny-2 | 2 | 256 | 4 | ~0.8M | 70-80% |

**Recommended: 4 layers, 512 hidden, 4 heads.**

#### B. Alternative Architectures

- **MLP-Mixer:** No quadratic attention, O(T*H^2). Fewer params but weaker long-range dependencies
- **Linear Attention:** O(T*H) instead of O(T^2*H). Same param count, research ongoing
- **PoolFormer:** Replace attention with average pooling. Minimal params (~1-2M), loses fine-grained interaction

**Verdict:** Stick with standard transformer for v1. Alternatives are experimental.

#### C. Quantization

| Precision | Size Reduction | Accuracy Impact |
|-----------|---------------|-----------------|
| FP32 (current) | 1x | baseline |
| FP16 | 2x | negligible |
| INT8 | 4x | <1% drop |
| INT4 | 8x | 2-5% drop |

**Apply INT8 quantization to all ONNX exports.**

---

## 3. Knowledge Distillation

### 3.1 Combined Loss Function

```python
L_total = 0.7 * MSE(student_pred, fmri_target)      # Task loss
        + 0.2 * MSE(student_pred, teacher_pred)       # Output KD
        + 0.1 * CKA(student_fusion, teacher_fusion)   # Feature KD
```

### 3.2 Feature-Level Distillation (Recommended)

Match intermediate fusion layer activations between teacher and student:
- Project student features to teacher dimension via bottleneck linear
- Use cosine similarity or MSE
- Higher weight on middle layers (capture semantic information)
- Recent work (2025): block-wise matching outperforms layer-wise

### 3.3 Progressive Distillation

```
Stage 1: 8→6 layers with KD
Stage 2: 6→4 layers with KD
Stage 3: Reduce hidden 1152→512 with KD
```

Each stage starts from a good checkpoint. Easier to debug accuracy drops.

### 3.4 Cross-Modal Distillation (Phase 2)

TRIBE paper shows multimodal gains are ~30% in associative cortices, but individual modalities are strong baselines. Could achieve 70-80% with single modality — useful for ultra-tiny variants.

---

## 4. Output Space Compression

### Option A: fsaverage4 (Recommended for browser)

| Mesh | Vertices | Reduction | Spatial Resolution | Accuracy |
|------|----------|-----------|-------------------|----------|
| fsaverage5 (current) | 20,484 | 1x | ~3mm | 100% |
| **fsaverage4** | 5,124 | 4x | ~4mm | 90-95% |
| fsaverage3 | 1,284 | 16x | ~8mm | 80-85% |

SubjectLayers: 2048x20484 → 2048x5124 (4x fewer params).

### Option B: ROI-Based Prediction

Predict 200-400 atlas parcels, interpolate to full mesh:
- Destrieux Atlas: ~150 parcels/hemisphere
- Schaefer-400: 400 parcels total
- Output head: 2048x400 = 0.8M (vs 42M for fsaverage5)
- Interpretable, but loses sub-parcel detail

### Option C: PCA on Vertex Space

Keep top-K eigenvectors of training vertex covariance:
- K=100 preserves ~85-90% variance
- Head: 2048→100, reconstruct at inference
- <10% accuracy drop

---

## 5. Browser Deployment

### 5.1 Size Budget

| Component | Format | Size |
|-----------|--------|------|
| all-MiniLM-L6-v2 | ONNX INT8 | 40MB |
| Whisper-Tiny | ONNX INT8 | 60MB |
| MobileViT-S | ONNX INT8 | 10MB |
| Fusion (4L transformer) | ONNX INT8 | 8MB |
| Output projection | ONNX INT8 | 5MB |
| **Total** | | **~120MB** |

### 5.2 Latency (MacBook M1)

| Stage | Latency |
|-------|---------|
| Text (10 words @ 2Hz) | 30ms |
| Audio (10s @ 2Hz) | 200ms |
| Video (10s @ 2fps, 20 frames) | 100ms |
| Fusion transformer (4L) | 50ms |
| Output projection | 100ms |
| **Total (parallel backbones)** | **~350ms** |

### 5.3 Architecture

```
Browser UI (React/Vue)
  ├── Video upload + live preview
  ├── Feature Extraction (Transformers.js, WebWorker)
  │     ├── Text tokenizer + encoder
  │     ├── Audio processor + encoder
  │     └── Video frame sampler + encoder
  ├── ONNX Runtime Web (WebGPU / WASM fallback)
  │     ├── Fusion transformer
  │     └── Output projection
  └── 3D Brain Visualization (Three.js + fsaverage mesh)
```

### 5.4 Export Pipeline

```bash
# 1. Export to ONNX
python -m torch.onnx.export model=fusion_model opset_version=14

# 2. Quantize to INT8
python -m onnxruntime.quantization.quantize_static model.onnx model_int8.onnx

# 3. Browser integration
npm install onnxruntime-web @xenova/transformers
```

```javascript
// Browser inference
const session = await ort.InferenceSession.create("model.onnx", {
  executionProviders: ["webgpu", "wasm"]  // WebGPU with WASM fallback
});
```

### 5.5 Memory: ~380MB peak (weights + activations + output buffers). Within browser limits.

---

## 6. Training Strategy

### Stage 1: Frozen Backbone Mapping (2-3 hours, 1 GPU)

- Freeze all tiny backbones
- Train projectors + fusion transformer + output head
- lr=1e-3, Adam, 5 epochs
- Use all 4 datasets (25 subjects, ~500 hours)
- Target: 80% of full TRIBE v2 Pearson r

### Stage 2: End-to-End Fine-Tuning with KD (5-10 hours, 1 GPU)

- Unfreeze tiny backbones
- Combined loss: MSE + output KD + feature KD
- lr=1e-4, OneCycleLR, 10 epochs
- Target: 90%+ of full TRIBE v2

### Data Augmentation

- Modality dropout: increase from 30% to 50% (force robustness with weaker backbones)
- Feature noise: Gaussian σ=0.05 on backbone outputs
- Time warping: 10-20% speed variation
- Layer dropout in transformer: 10-15%

---

## 7. Phased Implementation Plan

### Phase 1: Proof-of-Concept (2-3 weeks)
- Implement tiny backbones with flexible dims in model.py
- Stage 1 training with frozen backbones
- Validate: target 80% mean Pearson r

### Phase 2: Distillation (2 weeks)
- Implement KD losses (feature + output level)
- Stage 2 end-to-end fine-tuning
- Progressive layer reduction with KD
- Target: 90%+ mean Pearson r

### Phase 3: Output Compression (1 week)
- Switch to fsaverage4 or ROI-based output
- Retrain output head
- Compare visualization quality side-by-side

### Phase 4: Browser Export (2 weeks)
- Export all models to ONNX INT8
- Build React app with Transformers.js + ONNX Runtime Web
- 3D brain visualization with Three.js
- Target: <1s inference on M1 GPU, <5s on CPU

### Phase 5: Polish (1-2 weeks)
- INT4 quantization if needed for size
- WebWorker for non-blocking processing
- IndexedDB caching for offline support
- Mobile optimization

---

## 8. Expected Performance

| Metric | Full TRIBE v2 | Tiny-TRIBE (Phase 1) | Tiny-TRIBE (Phase 2) | Tiny-TRIBE (Final) |
|--------|---------------|---------------------|---------------------|-------------------|
| Mean Pearson r | 0.31 | 0.24-0.26 (77-84%) | 0.29-0.30 (94-97%) | 0.28-0.29 (90-94%) |
| Parameters | 4.7B | 67M | 67M | 67M |
| Model size | 10GB | 260MB | 260MB | 120MB (INT8) |
| Inference (GPU) | ~500ms | ~350ms | ~350ms | ~350ms |
| Inference (CPU) | 10-15s | 3-5s | 3-5s | 3-5s |
| Browser deployable | No | Yes | Yes | Yes |

---

## 9. Open Questions

1. **Does all-MiniLM-L6-v2 preserve temporal alignment?** Sentence embeddings aren't designed for word-level timing — may need adapter
2. **Optimal fusion for tiny models?** MLP-Mixer or pooling might suffice over full transformer
3. **Can we predict on fsaverage3 without major accuracy loss?** Would enable ~80MB total
4. **How much does retinotopy matter in compressed models?** Accepting blurry low-level vision saves bandwidth
5. **Cross-modal dropout scaling:** For tiny models, optimal might be 50-70% vs current 30%

---

## 10. Key References

- TRIBE v2 paper (d'Ascoli et al., Meta FAIR, 2026)
- "Brain alignment saturates early" — 3B matches 7-14B in neural predictivity (2024-2025)
- MobileViT (Mehta & Rastegari, ICLR 2022)
- Counterclockwise block-by-block KD (Nature, 2025)
- ONNX Runtime Web with WebGPU (Microsoft, 2024)
- Transformers.js (Hugging Face)

### Model Hubs
- `sentence-transformers/all-MiniLM-L6-v2`
- `openai/whisper-tiny`
- `apple/mobilevit-small`
- `Qwen/Qwen3-0.5B-ONNX`

"""Generate Tiny-TRIBE architecture diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(20, 28))
ax.set_xlim(0, 20)
ax.set_ylim(0, 32)
ax.axis('off')
fig.patch.set_facecolor('#0a0a0a')

# Colors
C_BG = '#0a0a0a'
C_TEXT_BG = '#1a1a3e'
C_AUDIO_BG = '#1a2e1a'
C_VIDEO_BG = '#2e1a1a'
C_PROJ = '#2a2a4a'
C_FUSION = '#1a1a2e'
C_TRANSFORMER = '#252550'
C_OUTPUT = '#2e2e1a'
C_BRAIN = '#3a1a3a'
C_ARROW = '#6366f1'
C_TEXT_COLOR = '#e0e0e0'
C_DIM = '#888888'
C_ACCENT1 = '#818cf8'  # indigo
C_ACCENT2 = '#4ade80'  # green
C_ACCENT3 = '#f87171'  # red
C_ACCENT4 = '#fbbf24'  # yellow
C_KD = '#f472b6'       # pink for KD

def draw_box(x, y, w, h, color, label, sublabel=None, alpha=0.85, fontsize=11, border_color=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor=border_color or '#444',
                         linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=C_TEXT_COLOR)
        ax.text(x + w/2, y + h/2 - 0.25, sublabel, ha='center', va='center',
                fontsize=8, color=C_DIM, style='italic')
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=C_TEXT_COLOR)

def draw_arrow(x1, y1, x2, y2, color=C_ARROW, style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def draw_dim_label(x, y, text, color=C_DIM):
    ax.text(x, y, text, ha='center', va='center', fontsize=7.5,
            color=color, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#111', edgecolor='#333', alpha=0.8))

# ============================================================
# TITLE
# ============================================================
ax.text(10, 31.2, 'Tiny-TRIBE Architecture', ha='center', va='center',
        fontsize=22, fontweight='bold', color='#a78bfa')
ax.text(10, 30.6, 'Compressed multimodal brain encoding for browser deployment',
        ha='center', va='center', fontsize=10, color=C_DIM)

# ============================================================
# INPUT LAYER (Raw Stimuli)
# ============================================================
ax.text(10, 29.8, '─── Raw Multimodal Stimuli ───', ha='center', fontsize=9, color='#555')

# Text input
draw_box(0.5, 28.5, 5.0, 1.0, '#111', 'Text Input', '"a person walking..."', border_color='#334')
# Audio input
draw_box(7.5, 28.5, 5.0, 1.0, '#111', 'Audio Input', 'Waveform @ 16kHz', border_color='#334')
# Video input
draw_box(14.5, 28.5, 5.0, 1.0, '#111', 'Video Input', 'Frames @ 2fps', border_color='#334')

# ============================================================
# FROZEN BACKBONES
# ============================================================
ax.text(10, 27.8, '─── Frozen Backbone Encoders ───', ha='center', fontsize=9, color='#555')

# Text backbone
draw_box(0.5, 26.2, 5.0, 1.3, C_TEXT_BG, 'all-MiniLM-L6-v2', '22.7M params │ frozen',
         border_color=C_ACCENT1)
draw_arrow(3.0, 28.5, 3.0, 27.5)
draw_dim_label(3.0, 25.8, '(B, T_text, 384)', C_ACCENT1)

# Audio backbone
draw_box(7.5, 26.2, 5.0, 1.3, C_AUDIO_BG, 'Whisper-Tiny Encoder', '39M params │ frozen*',
         border_color=C_ACCENT2)
draw_arrow(10.0, 28.5, 10.0, 27.5)
draw_dim_label(10.0, 25.8, '(B, 1500, 384)', C_ACCENT2)

# Video backbone
draw_box(14.5, 26.2, 5.0, 1.3, C_VIDEO_BG, 'MobileViT-S', '5.6M params │ frozen*',
         border_color=C_ACCENT3)
draw_arrow(17.0, 28.5, 17.0, 27.5)
draw_dim_label(17.0, 25.8, '(B, T_vid, 640)', C_ACCENT3)

ax.text(10, 25.3, '* unfrozen in Stage 2 (KD fine-tuning)', ha='center',
        fontsize=7, color='#666', style='italic')

# ============================================================
# PROJECTORS
# ============================================================
ax.text(10, 24.9, '─── Per-Modality Projectors (MLP) ───', ha='center', fontsize=9, color='#555')

draw_box(0.5, 23.5, 5.0, 1.1, C_PROJ, 'Text Projector', 'LN → Linear → GELU → Linear → LN',
         border_color=C_ACCENT1)
draw_arrow(3.0, 25.55, 3.0, 24.6)
draw_dim_label(3.0, 23.1, '(B, T, 170)', C_ACCENT1)

draw_box(7.5, 23.5, 5.0, 1.1, C_PROJ, 'Audio Projector', 'LN → Linear → GELU → Linear → LN',
         border_color=C_ACCENT2)
draw_arrow(10.0, 25.55, 10.0, 24.6)
draw_dim_label(10.0, 23.1, '(B, T, 170)', C_ACCENT2)

draw_box(14.5, 23.5, 5.0, 1.1, C_PROJ, 'Video Projector', 'LN → Linear → GELU → Linear → LN',
         border_color=C_ACCENT3)
draw_arrow(17.0, 25.55, 17.0, 24.6)
draw_dim_label(17.0, 23.1, '(B, T, 170)', C_ACCENT3)

# ============================================================
# TEMPORAL ALIGNMENT + CONCAT
# ============================================================
# Arrows converging
draw_arrow(3.0, 23.1, 8.5, 22.3)
draw_arrow(10.0, 23.1, 10.0, 22.3)
draw_arrow(17.0, 23.1, 11.5, 22.3)

draw_box(5.5, 21.5, 9.0, 0.8, '#1a1a1a', 'Temporal Align (interpolate) → Concatenate → Combiner MLP',
         border_color='#555', fontsize=9)
draw_dim_label(10.0, 21.1, 'concat(170,170,170) = 510 → Linear → 512', '#999')

# Modality dropout annotation
ax.text(17.5, 22.0, 'Modality\nDropout 50%', ha='center', fontsize=7,
        color=C_ACCENT4, style='italic',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a0a', edgecolor=C_ACCENT4, alpha=0.6))

# ============================================================
# POSITIONAL EMBEDDING
# ============================================================
draw_arrow(10.0, 21.5, 10.0, 20.9)
draw_box(6.5, 20.2, 7.0, 0.7, '#1a1a2e', '+ Positional Embedding (max 2048)',
         border_color='#444', fontsize=9)
draw_dim_label(10.0, 19.8, '(B, T, 512)')

# ============================================================
# FUSION TRANSFORMER
# ============================================================
draw_arrow(10.0, 20.2, 10.0, 19.5)

ax.text(10, 19.3, '─── Fusion Transformer (4 layers) ───', ha='center', fontsize=9, color='#555')

# Draw 4 transformer blocks
block_y = 18.6
for i in range(4):
    y = block_y - i * 1.4
    # Block background
    draw_box(3.5, y - 0.5, 13.0, 1.2, C_TRANSFORMER,
             f'Transformer Block {i+1}', None, border_color='#4a4a7a')

    # Sub-components inside
    ax.text(6.5, y + 0.15, 'Pre-Norm', ha='center', fontsize=7, color='#aaa')
    ax.text(6.5, y - 0.15, 'MultiHead Attn', ha='center', fontsize=8,
            color=C_ACCENT1, fontweight='bold')

    ax.text(13.5, y + 0.15, 'Pre-Norm', ha='center', fontsize=7, color='#aaa')
    ax.text(13.5, y - 0.15, 'FFN (512→2048→512)', ha='center', fontsize=8,
            color=C_ACCENT2, fontweight='bold')

    ax.text(10.0, y + 0.05, '→', ha='center', fontsize=14, color='#555')

    if i < 3:
        draw_arrow(10.0, y - 0.5, 10.0, y - 0.7, color='#444')

    # Layer dropout annotation on first block
    if i == 0:
        ax.text(17.5, y, 'Layer\nDrop 15%', ha='center', fontsize=7,
                color=C_ACCENT4, style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a0a',
                          edgecolor=C_ACCENT4, alpha=0.6))

# Dims
draw_dim_label(2.5, block_y - 2.2, '4 heads, 512D')
draw_dim_label(10.0, block_y - 4.7, '(B, T, 512)')

# Final norm
fn_y = block_y - 5.2
draw_box(7.0, fn_y, 6.0, 0.6, '#1a1a2e', 'LayerNorm', border_color='#444', fontsize=9)
draw_arrow(10.0, block_y - 4.45, 10.0, fn_y + 0.6)

# ============================================================
# OUTPUT HEAD
# ============================================================
out_y = fn_y - 0.8
ax.text(10, out_y + 0.2, '─── Output Head ───', ha='center', fontsize=9, color='#555')

# Low rank projection
lr_y = out_y - 0.6
draw_box(6.0, lr_y, 8.0, 0.7, C_OUTPUT, 'Low-Rank Projection',
         '512 → 256 (no bias)', border_color=C_ACCENT4, fontsize=10)
draw_arrow(10.0, fn_y, 10.0, lr_y + 0.7)
draw_dim_label(10.0, lr_y - 0.3, '(B, T, 256)', C_ACCENT4)

# Subject layers
sl_y = lr_y - 1.2
draw_box(4.5, sl_y, 11.0, 0.9, C_BRAIN, 'SubjectLayers (per-subject linear)',
         'W[subj]: (256, n_vertices) + bias', border_color='#a855f7', fontsize=10)
draw_arrow(10.0, lr_y - 0.05, 10.0, sl_y + 0.9)

# Subject ID input
draw_box(16.5, sl_y + 0.1, 3.0, 0.7, '#111', 'Subject ID', '(B,)', border_color='#a855f7', fontsize=9)
draw_arrow(16.5, sl_y + 0.45, 15.5, sl_y + 0.45, color='#a855f7')

draw_dim_label(10.0, sl_y - 0.35, '(B, T, 5124)  or  (B, T, 400)', '#a855f7')

# Temporal pooling
tp_y = sl_y - 1.1
draw_box(6.0, tp_y, 8.0, 0.65, '#1a1a1a', 'AdaptiveAvgPool1d → n_output_TRs',
         border_color='#555', fontsize=9)
draw_arrow(10.0, sl_y - 0.1, 10.0, tp_y + 0.65)

# Output
out_final_y = tp_y - 0.9
draw_box(5.0, out_final_y, 10.0, 0.7, '#0a2e1a', 'Predicted Brain Activation',
         '(B, n_vertices, n_TRs)', border_color=C_ACCENT2, fontsize=11)
draw_arrow(10.0, tp_y, 10.0, out_final_y + 0.7)

# ============================================================
# KD ANNOTATION (right side)
# ============================================================
kd_x = 0.3
kd_y = 8.0

# KD loss box
draw_box(kd_x, kd_y - 1.5, 3.2, 3.5, '#1a0a1a',
         '', None, border_color=C_KD, alpha=0.7)
ax.text(kd_x + 1.6, kd_y + 1.6, 'KD Loss (Stage 2)', ha='center', fontsize=9,
        fontweight='bold', color=C_KD)
ax.text(kd_x + 1.6, kd_y + 1.0, '0.7 × MSE(pred, fMRI)', ha='center',
        fontsize=7.5, color=C_TEXT_COLOR, fontfamily='monospace')
ax.text(kd_x + 1.6, kd_y + 0.6, '0.2 × MSE(pred, teacher)', ha='center',
        fontsize=7.5, color=C_TEXT_COLOR, fontfamily='monospace')
ax.text(kd_x + 1.6, kd_y + 0.2, '0.1 × CKA(features)', ha='center',
        fontsize=7.5, color=C_TEXT_COLOR, fontfamily='monospace')
ax.text(kd_x + 1.6, kd_y - 0.4, 'Teacher: TRIBE v2', ha='center',
        fontsize=7.5, color='#888', style='italic')
ax.text(kd_x + 1.6, kd_y - 0.8, '4.7B params, 8L/1152D', ha='center',
        fontsize=7, color='#666')

# ============================================================
# STATS BOX (bottom)
# ============================================================
stats_y = 1.5
draw_box(0.5, stats_y - 1.2, 19.0, 2.8, '#111', '', None, border_color='#333', alpha=0.9)

ax.text(10.0, stats_y + 1.2, 'Model Summary', ha='center', fontsize=12,
        fontweight='bold', color='#a78bfa')

col1_x, col2_x, col3_x = 3.5, 10.0, 16.5
ax.text(col1_x, stats_y + 0.6, 'Backbones (frozen)', ha='center', fontsize=9, color='#aaa')
ax.text(col1_x, stats_y + 0.1, '67.3M params', ha='center', fontsize=11,
        fontweight='bold', color=C_ACCENT1)
ax.text(col1_x, stats_y - 0.3, '~260MB (FP32)', ha='center', fontsize=8, color='#888')

ax.text(col2_x, stats_y + 0.6, 'Fusion (trainable)', ha='center', fontsize=9, color='#aaa')
ax.text(col2_x, stats_y + 0.1, '~14-47M params', ha='center', fontsize=11,
        fontweight='bold', color=C_ACCENT2)
ax.text(col2_x, stats_y - 0.3, '4L / 512D / 4H', ha='center', fontsize=8, color='#888')

ax.text(col3_x, stats_y + 0.6, 'Browser Deploy', ha='center', fontsize=9, color='#aaa')
ax.text(col3_x, stats_y + 0.1, '~120MB total', ha='center', fontsize=11,
        fontweight='bold', color=C_ACCENT4)
ax.text(col3_x, stats_y - 0.3, 'ONNX INT8 + WebGPU', ha='center', fontsize=8, color='#888')

# ============================================================
# SAVE
# ============================================================
plt.tight_layout()
plt.savefig('/Users/avaya.aggarwal@zomato.com/tribe/tiny_tribe_architecture.png',
            dpi=200, bbox_inches='tight', facecolor=C_BG, edgecolor='none')
plt.savefig('/Users/avaya.aggarwal@zomato.com/tribe/tiny_tribe_architecture.pdf',
            bbox_inches='tight', facecolor=C_BG, edgecolor='none')
print("Saved: tiny_tribe_architecture.png and .pdf")
plt.show()

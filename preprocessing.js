/**
 * Media preprocessing for Tiny-TRIBE browser inference.
 *
 * Extracts text (from subtitles/transcript), audio mel spectrograms,
 * and video frames from an uploaded video file.
 */

import { AutoTokenizer } from "@xenova/transformers";

const AUDIO_SAMPLE_RATE = 16000;
const VIDEO_FPS = 2;
const FRAME_SIZE = 256;
const MEL_BINS = 80;
const MEL_HOP = 160; // 10ms hop at 16kHz

let tokenizer = null;

export async function initTokenizer() {
  tokenizer = await AutoTokenizer.from_pretrained(
    "Xenova/all-MiniLM-L6-v2"
  );
  return tokenizer;
}

/**
 * Tokenize text input for the text encoder.
 * @param {string} text
 * @returns {{ inputIds: Int32Array, attentionMask: Int32Array }}
 */
export function tokenizeText(text) {
  if (!tokenizer) throw new Error("Tokenizer not initialized");
  const encoded = tokenizer(text, {
    padding: true,
    truncation: true,
    max_length: 128,
    return_tensors: false,
  });
  return {
    inputIds: new Int32Array(encoded.input_ids),
    attentionMask: new Int32Array(encoded.attention_mask),
  };
}

/**
 * Extract audio from video and compute mel spectrogram.
 * @param {File} videoFile
 * @returns {{ data: Float32Array, melLength: number }}
 */
export async function extractAudioMel(videoFile) {
  const audioCtx = new OfflineAudioContext(1, 1, AUDIO_SAMPLE_RATE);
  const arrayBuffer = await videoFile.arrayBuffer();

  // Decode full audio
  const onlineCtx = new AudioContext({ sampleRate: AUDIO_SAMPLE_RATE });
  const audioBuffer = await onlineCtx.decodeAudioData(arrayBuffer);
  await onlineCtx.close();

  // Get mono PCM
  const pcm = audioBuffer.getChannelData(0);

  // Compute log-mel spectrogram (simplified — real Whisper uses specific filterbank)
  const melFrames = computeLogMelSpectrogram(pcm, AUDIO_SAMPLE_RATE, MEL_BINS, MEL_HOP);

  // Pad or truncate to Whisper's expected 3000 frames (30s)
  const targetFrames = 3000;
  const melData = new Float32Array(MEL_BINS * targetFrames);
  const copyFrames = Math.min(melFrames.length / MEL_BINS, targetFrames);
  melData.set(melFrames.subarray(0, copyFrames * MEL_BINS));

  return { data: melData, melLength: targetFrames };
}

/**
 * Simplified log-mel spectrogram computation.
 * For production, use Whisper's exact feature extractor via Transformers.js.
 */
function computeLogMelSpectrogram(pcm, sampleRate, nMels, hopLength) {
  const fftSize = 400; // 25ms window at 16kHz
  const numFrames = Math.floor((pcm.length - fftSize) / hopLength) + 1;
  const mel = new Float32Array(nMels * numFrames);

  for (let frame = 0; frame < numFrames; frame++) {
    const start = frame * hopLength;
    // Compute energy in mel-spaced frequency bands (simplified)
    for (let m = 0; m < nMels; m++) {
      let energy = 0;
      const bandStart = Math.floor((m / nMels) * (fftSize / 2));
      const bandEnd = Math.floor(((m + 1) / nMels) * (fftSize / 2));
      for (let k = bandStart; k < bandEnd && k < fftSize; k++) {
        const idx = start + k;
        if (idx < pcm.length) {
          energy += pcm[idx] * pcm[idx];
        }
      }
      mel[m * numFrames + frame] = Math.log(Math.max(energy, 1e-10));
    }
  }

  return mel;
}

/**
 * Extract video frames at target FPS.
 * @param {File} videoFile
 * @returns {Array<{ data: Float32Array, height: number, width: number }>}
 */
export async function extractVideoFrames(videoFile) {
  const video = document.createElement("video");
  video.src = URL.createObjectURL(videoFile);
  video.muted = true;

  await new Promise((resolve) => {
    video.onloadedmetadata = resolve;
  });

  const duration = video.duration;
  const numFrames = Math.ceil(duration * VIDEO_FPS);
  const interval = 1.0 / VIDEO_FPS;

  const canvas = document.createElement("canvas");
  canvas.width = FRAME_SIZE;
  canvas.height = FRAME_SIZE;
  const ctx = canvas.getContext("2d");

  const frames = [];

  for (let i = 0; i < numFrames; i++) {
    video.currentTime = i * interval;
    await new Promise((resolve) => {
      video.onseeked = resolve;
    });

    ctx.drawImage(video, 0, 0, FRAME_SIZE, FRAME_SIZE);
    const imageData = ctx.getImageData(0, 0, FRAME_SIZE, FRAME_SIZE);

    // Convert RGBA to CHW float32 normalized [0,1]
    const chw = new Float32Array(3 * FRAME_SIZE * FRAME_SIZE);
    for (let y = 0; y < FRAME_SIZE; y++) {
      for (let x = 0; x < FRAME_SIZE; x++) {
        const srcIdx = (y * FRAME_SIZE + x) * 4;
        const dstIdx = y * FRAME_SIZE + x;
        chw[0 * FRAME_SIZE * FRAME_SIZE + dstIdx] = imageData.data[srcIdx] / 255.0;     // R
        chw[1 * FRAME_SIZE * FRAME_SIZE + dstIdx] = imageData.data[srcIdx + 1] / 255.0; // G
        chw[2 * FRAME_SIZE * FRAME_SIZE + dstIdx] = imageData.data[srcIdx + 2] / 255.0; // B
      }
    }

    frames.push({ data: chw, height: FRAME_SIZE, width: FRAME_SIZE });
  }

  URL.revokeObjectURL(video.src);
  return frames;
}

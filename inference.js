/**
 * ONNX Runtime Web inference engine for Tiny-TRIBE.
 *
 * Loads 4 ONNX models (text, audio, video encoders + fusion)
 * and runs the full pipeline: raw input → backbone features → fusion → vertex predictions.
 */

import * as ort from "onnxruntime-web";

const MODEL_BASE_URL = "./models/";

export class TinyTribeInference {
  constructor() {
    this.sessions = {};
    this.ready = false;
  }

  async init() {
    // Prefer WebGPU, fall back to WASM
    const providers = ["webgpu", "wasm"];

    const modelFiles = [
      "text_encoder_int8.onnx",
      "audio_encoder_int8.onnx",
      "video_encoder_int8.onnx",
      "fusion_int8.onnx",
    ];

    const loadPromises = modelFiles.map(async (file) => {
      const name = file.split("_int8")[0];
      try {
        this.sessions[name] = await ort.InferenceSession.create(
          MODEL_BASE_URL + file,
          { executionProviders: providers }
        );
        console.log(`Loaded ${name}`);
      } catch (e) {
        console.warn(`Failed to load ${file} with WebGPU, falling back to WASM`);
        this.sessions[name] = await ort.InferenceSession.create(
          MODEL_BASE_URL + file,
          { executionProviders: ["wasm"] }
        );
      }
    });

    await Promise.all(loadPromises);
    this.ready = true;
    return this;
  }

  /**
   * Run text encoder on tokenized input.
   * @param {Int32Array} inputIds - Token IDs (1, T)
   * @param {Int32Array} attentionMask - Attention mask (1, T)
   * @returns {Float32Array} embeddings (1, T, 384)
   */
  async encodeText(inputIds, attentionMask) {
    const feeds = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(inputIds.map(BigInt)), [1, inputIds.length]),
      attention_mask: new ort.Tensor("int64", BigInt64Array.from(attentionMask.map(BigInt)), [1, attentionMask.length]),
    };
    const result = await this.sessions.text_encoder.run(feeds);
    return result.embeddings;
  }

  /**
   * Run audio encoder on mel spectrogram.
   * @param {Float32Array} melSpec - Mel spectrogram (1, 80, T_mel)
   * @param {number} melLength - Number of mel frames
   * @returns {Float32Array} embeddings (1, T, 384)
   */
  async encodeAudio(melSpec, melLength) {
    const feeds = {
      input_features: new ort.Tensor("float32", melSpec, [1, 80, melLength]),
    };
    const result = await this.sessions.audio_encoder.run(feeds);
    return result.embeddings;
  }

  /**
   * Run video encoder on a single frame.
   * @param {Float32Array} pixelValues - Frame pixels (1, 3, H, W)
   * @param {number} height
   * @param {number} width
   * @returns {Float32Array} embeddings (1, D)
   */
  async encodeVideoFrame(pixelValues, height, width) {
    const feeds = {
      pixel_values: new ort.Tensor("float32", pixelValues, [1, 3, height, width]),
    };
    const result = await this.sessions.video_encoder.run(feeds);
    return result.embeddings;
  }

  /**
   * Run fusion model on pre-extracted features.
   * @param {ort.Tensor} textFeatures - (1, T, 384)
   * @param {ort.Tensor} audioFeatures - (1, T, 384)
   * @param {ort.Tensor} videoFeatures - (1, T, 640)
   * @param {number} subjectId
   * @returns {Float32Array} vertex predictions (1, 5124, T)
   */
  async runFusion(textFeatures, audioFeatures, videoFeatures, subjectId) {
    const feeds = {
      text_features: textFeatures,
      audio_features: audioFeatures,
      video_features: videoFeatures,
      subject_id: new ort.Tensor("int64", BigInt64Array.from([BigInt(subjectId)]), [1]),
    };
    const result = await this.sessions.fusion.run(feeds);
    return result.vertex_predictions;
  }

  /**
   * Full pipeline: encode all modalities in parallel, then fuse.
   */
  async predict(textTokens, audioMel, videoFrames, subjectId) {
    const startTime = performance.now();

    // Run backbone encoders in parallel
    const [textEmb, audioEmb, videoEmbs] = await Promise.all([
      this.encodeText(textTokens.inputIds, textTokens.attentionMask),
      this.encodeAudio(audioMel.data, audioMel.melLength),
      this.encodeVideoFrames(videoFrames),
    ]);

    // Fuse and predict
    const predictions = await this.runFusion(textEmb, audioEmb, videoEmbs, subjectId);

    const latencyMs = performance.now() - startTime;
    return { predictions, latencyMs };
  }

  /**
   * Encode multiple video frames and stack into (1, T, D).
   */
  async encodeVideoFrames(frames) {
    const embeddings = [];
    for (const frame of frames) {
      const emb = await this.encodeVideoFrame(frame.data, frame.height, frame.width);
      embeddings.push(emb.data);
    }

    // Stack: (T, D) → (1, T, D)
    const D = embeddings[0].length;
    const T = embeddings.length;
    const stacked = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      stacked.set(embeddings[t], t * D);
    }
    return new ort.Tensor("float32", stacked, [1, T, D]);
  }
}

# STT / TTS on the QCS6490 NPU — investigation (2026-09-06)

Question: which speech-to-text (STT) and text-to-speech (TTS) models can run on
the Radxa Dragon Q6A's QCS6490 HTP (v68, 8 MB VTCM, 4 HMX), given the
firmware op-package constraints proven for the Immich NPU work?

**Bottom line**

- **STT: YES.** Whisper (tiny/base/small) and encoder-only Distil-Whisper are
  fully coverable by ops this firmware runs. No blocked ops in the
  architectures. First target: **distil-whisper-small.en** or **whisper-base**
  through the proven QAIRT 2.37.1 INT8 pipeline.
- **TTS: YES (with one rewrite).** Piper (VITS) is fully coverable — every op
  it needs was probe-verified on this firmware today, including
  `ConvTranspose` (vocoder upsample). Its *official* ONNX ships with dynamic
  control-flow ops and must be re-exported with fixed shapes (Qualcomm's open
  `pipertts_en` recipe already does exactly this).
- **Precompiled AI Hub speech assets do not exist for QCS6490** (checked the
  current catalog and every past release). They would also be compiled with
  QAIRT 2.45, whose op packages this firmware (QNN interface cap 2.32)
  cannot load. **The only viable path is compile-from-source with QAIRT
  2.37.1** — our existing pipeline.

---

## 1. Method

Same evidence discipline as the SCRFD/Resize finding: minimal single-op ONNX
models → `qairt-converter` (HTP) → `qairt-quantizer` (w8a8 per-channel, the
production settings) → `tools/compile_htp.py` (SM7325 / v68 / soc_model 35)
→ register+execute on the board with `qnn-net-run` (2.37.1 aarch64, Radxa
runtime libs).

Board-side gotchas found while doing this (useful for any future load test):

- `/dev/fastrpc-cdsp` is `root:render` — board user `buga` cannot open
  fastrpc sessions; run load tests with `sudo` (password in
  `/home/buga/code/vllm-setup/.env`, key `RADXA_SUDO`; never publish).
- `/home/buga/immich-ml-qnn/rt6490/qnn-net-run` is **v2.42** while the
  Radxa runtime libs are 2.37.1 → mixed versions fail fastrpc handshake
  (`qnn_open 0x72 / Failed to load skel 1002`). Use the SDK's
  `bin/aarch64-ubuntu-gcc9.4/qnn-net-run` (2.37.1) together with the Radxa
  2.37.1 `libQnnHtp{,V68Stub,V68Skel}.so`.
- The SDK's prebuilt stub/skel are the **unsigned** DSP variant: on this
  board `createUnsignedPD ... not supported by HTP`. Always pair the SDK
  2.37.1 `qnn-net-run` binary with the Radxa (signed) runtime libs.
- Probe workspace kept on the board at
  `/home/buga/immich-ml-qnn/probe-asr-tts/` (qnn-net-run, libs, contexts,
  logs) and on the build host at `~/qairt/work/probes/`.

## 2. Firmware op matrix (QCS6490 HTP v68, QAIRT 2.37.1)

Empirical — what the firmware actually registers and executes:

| Status | Ops (evidence) |
|---|---|
| Proven by production models | Conv, Gemm/MatMul, Add/Mul/Sub, Relu, Sigmoid, Softmax, LayerNorm, MaxPool, **Erf** (CLIP GELU), Reshape, Transpose, Concat, BatchNorm (fused) |
| **Probe-verified this session** | **Gather** (embedding lookup, axis-0, int32 idx via Cast), **ConvTranspose** (2D stride-2 — vocoder upsample), **Conv on 3-D input** (1-D conv — Whisper stem / VITS), **InstanceNormalization** (VITS norm), **Pad** (constant, 1-D zero pad) |
| **Blocked (firmware op package missing)** | **Resize** (`0x138d` OP_PACKAGE_NOT_FOUND — the SCRFD finding) |
| Avoid (dynamic / data-dependent; not NPU-compatible by design) | NonZero, ScatterND/GatherND, dynamic Slice/Where/Range control flow, RandomNormalLike, CumSum (unprobed — not needed by either recommended architecture) |

Conclusion: the entire op set of Whisper (conv stem + SHA/MHA attention +
layernorm + erf-gelu) and of VITS/Piper (embedding gather + 1-D conv stacks +
instance/layer norm + pad + ConvTranspose vocoder) is available on this
firmware. `Resize` is the only known hole, and neither recommended
architecture uses it.

## 3. Qualcomm AI Hub — why not just download precompiled models

- QCS6490 **is** a supported AI Hub chipset (v0.52.0: `qualcomm-qcs6490`,
  HTP v68, soc_model 93; device "Dragonwing RB3", QC Linux 1.6).
- But **no speech asset targets it**: `qai-hub-models find <model>
  -c qualcomm-qcs6490 --all` returns nothing for whisper-tiny/base/small,
  whisper-small-quantized, distil-whisper, melotts-en, piper-tts-en. Current
  + all past releases ship whisper/melo assets only for QCM6690/QCS8450/
  QCS8550/QCS9075/SA7255P/SA8295P/SA8775P/Snapdragon 7 Gen 4/8 Elite/
  8 Gen 3/X Elite/X2 Elite (HTP v73/v75+).
- The Dragonwing docs page (staging) that advertises a Whisper download for
  "Qualcomm QCS6490 (Proxy)" has no matching public asset — stale doc.
- Even a hypothetical QCS6490 asset would be built with QAIRT 2.45
  (`tool_versions.qairt: 2.45.0.260326154327`) → op packages newer than this
  firmware's 2.32 interface cap → registration failure (same failure class
  as the proven 2.42 context-binary incompatibility).
- TTS on AI Hub (MeloTTS/PiperTTS) targets the **VoiceAI TTS runtime**
  (separate SDK, QPM package) built for the same newer parts.

**Therefore: compile from source with QAIRT 2.37.1.250807 (our toolchain).**

## 4. STT candidates

All encoder sizes below assume 30 s of 80×3000 mel input (Whisper
convention); the ai-hub-models optimized exports replace MHA with single-head
attention and linears with convs — better NPU fit, Apache-2.0.

| Model | Params | Architecture | NPU fit | Notes |
|---|---|---|---|---|
| **whisper-tiny** | ~39 M | enc 4L/384 + AR dec | ✅ all ops | English-only quality; good first integration target (fastest to prove the loop) |
| **whisper-base** | ~74 M | enc 6L/512 + AR dec | ✅ all ops | practical minimum for multilingual use |
| **whisper-small** | ~244 M | enc 12L/768 + AR dec | ✅, slower | 1500² attention @ d=768 spills VTCM → DDR; expect 0.3–1 s encode; quality ceiling for this NPU |
| **distil-whisper-small.en** | ~244 M | **encoder-only (CTC-style), no AR decoder** | ✅✅ | single forward pass per 30 s clip; decode is cheap CPU CTC; English-only; best quality/latency ratio for English |
| zipformer (ai-hub recipe) | ~30–150 M | Conformer (conv + attn + FFN) | ✅ | alternative; Shakti |
| moonshine tiny/base | 26–95 M | enc conv+RoPE transformer + small AR dec | ✅ (RoPE = baked/const math) | designed for embedded; English; worth an A/B vs whisper |
| wav2vec2-base CTC (960h) | ~94 M | 7 conv blocks + 12L transformer | ✅ | older quality than whisper |
| parakeet-TDT 0.6B | ~600 M | Conformer | ⚠️ too big | skip for v68 |

Decoder note: Whisper is autoregressive — run the decoder as a **fixed
single-step context** (one NPU execute per token; the optimized export uses a
200-token max length). At ~1–5 ms/step that's 0.2–1 s per clip. No
context create/free per step (load once — same discipline as the Immich
daemon).

Rough latency budget (extrapolated from CLIP ViT-B/32 @ 17 ms, 4 HMX):
encoder tiny ~60–150 ms, base ~150–350 ms, small/distil ~300–900 ms per
30 s of audio → **>30× realtime for STT** is a plausible target; measure.

## 5. TTS candidates

| Model | Params | Architecture | NPU fit | Notes |
|---|---|---|---|---|
| **Piper EN (VITS)** | ~11–20 M / voice | enc CNN + SDP + decoder CNN + HiFi-GAN (ConvTranspose) | ✅ all ops probe-verified | 22.05 kHz, single forward (non-AR); many EN voices; `rhasspy/piper-voices` weights; the official ONNX must be re-exported fixed-shape (see below) |
| Kokoro-82M | 82 M + ~100 M vocoder | parallel transformer + depthwise conv + HiFiGAN | ✅ in theory (ops proven) | 24 kHz, better quality; larger; no Qualcomm recipe — own export work |
| MeloTTS (en/es/zh) | ~300 M+, 44.1 kHz | BERT + flow + GAN vocoder | ⚠️ size + 44.1 kHz rate + BERT/flow ops unprobed | skip for v68 first pass |
| F5-TTS / XTTS / StyleTTS2 | 335 M+ / AR or heavy flow | — | ⚠️ | skip |

Piper-specific findings (from the official `en_US-lessac-low.onnx`,
downloaded + op-counted this session):

- 129× Conv (1-D/3-D), 168× Gather (phoneme embedding), 24× Erf, 3×
  ConvTranspose (upsample), InstanceNorm/LayerNorm/Pad/Softmax — **all in the
  proven set above**.
- But also: `RandomNormalLike` (stochastic sampling), `NonZero`, `ScatterND`,
  `GatherND`, `CumSum`, and dynamic `Slice`/`Where`/`Range` control flow —
  **incompatible with fixed-shape HTP contexts**. Qualcomm's open
  `pipertts_en` recipe (ai-hub-models repo) rewrites the model into
  fixed-shape Encoder/SDP/Flow/Decoder components with a deterministic noise
  buffer — use that recipe as the ONNX generator and compile each component
  with our pipeline.
- Text → phonemes stays on CPU (espeak-ng G2P); only the VITS nets run on NPU.
- Expected: mel synthesis + vocoder well under 100 ms per 10 s of speech on
  HTP (Piper already runs realtime on a Pi 4 CPU).

## 6. Recommended plan

**Phase 1 — STT (whisper, ~1–2 days)**
1. Export `distil-whisper-small.en` (or `whisper-base` if multilingual is
   required) with the ai-hub-models recipe (SHA + conv, fixed 30 s).
2. `qairt-converter` (HTP) → `qairt-quantizer` w8a8 per-channel with a small
   in-distribution calibration set (LibriSpeech/own voice, mel 80×3000) →
   `tools/compile_htp.py` (vtcm 2).
3. Board load test in `probe-asr-tts/` (sudo; 2.37.1 qnn-net-run + Radxa
   libs), latency + WER sanity check on a few sentences.
4. Fallback if w8a8 accuracy is poor: try their w8a16 export via AIMET
   (`~/qairt/work/aimet-venv` exists) — HTP has an INT16 precision column, so
   w8a16 is not ruled out on v68.
5. CPU reference: `whisper.cpp` base int8 (also recommended by the Dragonwing
   docs) for A/B.

**Phase 2 — TTS (piper EN, ~1–2 days)**
1. Run the `pipertts_en` export for a chosen voice (e.g.
   `en_US-lessac-medium`) → fixed-shape component ONNXes.
2. Same pipeline per component (encoder/SDP/flow/decoder + vocoder).
3. Host-side orchestration (C++ daemon, second daemon socket or extend
   `qnn_dsp_daemon` with new model keys): phoneme ids → encoder+SDP →
   duration repeat → decoder (mel) → vocoder (waveform) → 22.05 kHz WAV.
4. A/B listen vs CPU Piper; check INT8 vocoder artifacts.

**Phase 3 — productize**
- Load contexts once (no create/free cycles — CDSP stability).
- Decide hosting: standalone service first; Immich has no voice API yet, so
  no Immich fork changes are implied.
- Optional: KWS/VAD front-end for continuous dictation (separate small
  model; not investigated here).

## 7. What was NOT verified (honest scope)

- No full-model compile/run yet (only single-op probes + the op-set
  analysis of real model ONNX files). Whisper/Piper end-to-end accuracy and
  latency on this NPU are estimates until Phases 1–2 run.
- `CumSum`, `ScatterND`, `GatherND`, `NonZero`, `RandomNormalLike` were not
  probe-tested (they are only needed by the *unrevised* Piper graph, which is
  rejected by design).
- AI Hub's "QCS6490 (Proxy)" assets could not be inspected because they do
  not exist in the public catalog; the firmware-cap argument is based on the
  proven 2.42-context-binary incompatibility, not a 2.45 test.

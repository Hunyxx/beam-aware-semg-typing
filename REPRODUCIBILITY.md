# Experimental settings

This document summarizes the model configurations and evaluation procedures
for sEMG typing transcription. Installation and training commands are in
[README.md](README.md).

## CNN–Conformer frontend

The frontend combines multi-scale convolutions, a shared bilateral Conformer
encoder, and CTC decoding. Side-position and typing-activity losses provide
auxiliary supervision during training.

The main configuration is
[config/model/split_share_position_aux.yaml](config/model/split_share_position_aux.yaml),
with training settings in [config/frontend.yaml](config/frontend.yaml).

| Setting | Value |
| --- | --- |
| Sampling rate | 2000 Hz |
| Input window | 8000 samples (4 seconds) |
| Past / future context | 1800 / 200 samples |
| Multi-scale convolution kernels | 7, 11, 15 |
| Conformer layers / hidden size / attention heads | 6 / 256 / 16 |
| Training epochs | 200 |
| Optimizer | AdamW |
| Learning rate / weight decay | 0.0003 / 0.1 |
| Schedule | Linear warmup followed by cosine annealing |
| Warmup epochs / starting learning rate | 5 / 1e-8 |
| Minimum learning rate | 1e-6 |
| Batch size per process / devices | 320 / 2 |
| Position loss coefficient | 0.29022504474034927 |
| Activity loss coefficient | 0.012597707807099066 |

Training uses band rotation and temporal alignment jitter. The generic and
personalized data splits are defined in [config/user](config/user).
Personalized training initializes the frontend from a generic `.ckpt`.

## Beam candidates

The frontend exports twenty ordered CTC candidates for each four-second
window. The beam decoder uses a beam size of 50 and a character language
model, configured in
[config/decoder/ctc_beam.yaml](config/decoder/ctc_beam.yaml).

Each CSV row records the split, session, window index, candidate rank,
prediction, reference text, and score. The corrector groups three consecutive
windows and uses the first ten candidates from each window. Backspaces are
resolved within each window, and windows with empty references are filtered.

## Beam-aware corrector

The corrector fine-tunes Qwen3.5-9B with LoRA using the configuration in
[config/beam_corrector.yaml](config/beam_corrector.yaml).

| Setting | Value |
| --- | --- |
| Base model | Qwen3.5-9B |
| Candidates per window / windows per input | 10 / 3 |
| Training epochs | 5 |
| Global batch size | 8 (1 per process × 2 devices × 4 accumulation steps) |
| Optimizer | AdamW |
| Learning rate / weight decay | 0.0001 / 0.1 |
| Schedule | Cosine annealing |
| Maximum input / generated tokens | 4096 / 256 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Standard / degraded / severely degraded sampling ratios | 0.2 / 0.3 / 0.5 |

The candidate pools contain exported `*_topk.csv` files. Their paths and the
validation pool are selected through the environment variables described in
the README.

During training, candidate order is shuffled within each window, while the
concatenated rank-1 transcription remains the baseline. Loss is computed on
response tokens. Inference uses candidates in beam order.

Training and inference share the prompt in
[config/templates/beam_correction.txt](config/templates/beam_correction.txt).
The output filter checks the generated transcription against the beam
candidates and retains the rank-1 baseline when the correction fails its
acceptance conditions.

Training saves the best and latest `.ckpt` files in the run's `checkpoints/`
directory. Evaluation loads these checkpoints directly.

## Evaluation

Frontend evaluation encodes complete recordings in 100000-sample blocks with
10000-sample overlap, joins their emissions, and decodes each session.
Corrector evaluation concatenates the corrected window groups in temporal
order to reconstruct session text.

Character error rate (CER) and word error rate (WER) are computed from the
reconstructed sessions. Corrector outputs include the rank-1 baseline,
generated text, and filtered transcription.

The analysis scripts provide the following measurements:

| Script in `scripts/analysis/` | Measurement |
| --- | --- |
| `compute_topk_oracle_cer.py` | Best-candidate CER and WER, selected independently within each window |
| `audit_beam_supported_edits.py` | Beam support and accuracy of correction edits |
| `analyze_rare_unseen_words.py` | Word errors by training-set frequency |
| `paired_sign_flip_test.py` | Two-sided exact paired sign-flip test across users |
| `benchmark_frontend_latency.py` | Preprocessing, encoding, and beam-decoding time |
| `benchmark_llm_correction_latency.py` | Prompt construction, generation, and output-filtering time |

The frontend processes four-second windows, and each correction uses three
windows of history. Latency measurements report the computation time of each
stage after its input is available.

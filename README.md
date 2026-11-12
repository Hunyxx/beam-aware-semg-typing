# sEMG Typing Transcription with Conformer Encoding and Beam-Aware Large Language Model Correction

Official implementation of our **EMNLP 2026 Main Conference** paper.

**Yuanxi Wang, Lun Shu, and Yasuharu Koike**

Proceedings of the 2026 Conference on Empirical Methods in Natural Language
Processing, pages 13763–13780, Budapest, Hungary, October 2026.

[Paper](https://aclanthology.org/2026.emnlp-main.598/) ·
[PDF](https://aclanthology.org/2026.emnlp-main.598.pdf) ·
[DOI](https://doi.org/10.18653/2026.emnlp-main.598)

The method transcribes surface electromyography (sEMG) recorded during typing
in two stages:

1. A multi-scale CNN–Conformer encodes the left and right hands with shared
   parameters and predicts characters using CTC. Side-position and typing
   activity supervision support frontend training.
2. A Qwen3.5-9B corrector combines the top-10 beam candidates from three
   consecutive four-second windows. It produces a joint transcription, followed
   by beam-aware output filtering and session reconstruction.

## Installation

```bash
git clone https://github.com/Hunyxx/beam-aware-semg-typing.git
cd beam-aware-semg-typing
conda env create -f environment.yml
conda activate semg-beam-aware-transcription
```

The environment uses Python 3.10 and pinned PyTorch builds for Linux x86_64
with CUDA 13.2. Runtime and test dependencies are listed in
[requirements.txt](requirements.txt).

## Data

Follow the [emg2qwerty dataset instructions](https://github.com/facebookresearch/emg2qwerty)
and set the directory containing the session HDF5 files:

```bash
export EMG2QWERTY_DATA_ROOT=/path/to/emg2qwerty-data-2021-08
```

[config/user](config/user) contains the generic and personalized data splits.

## Train the frontend

```bash
python -m emg2qwerty.train --config-name frontend
```

The default configuration trains the split-and-share CNN–Conformer with
side-position and activity losses. Select a model using `model=NAME`:

| Model | Multi-scale CNN | Split-and-share encoder | Auxiliary losses |
| --- | --- | --- | --- |
| `conformer_base` | | | |
| `conformer_multiscale` | ✓ | | |
| `split_share` | ✓ | ✓ | |
| `split_share_position_aux` | ✓ | ✓ | ✓ |

For personalized training, select a user and initialize from your generic
training checkpoint:

```bash
python -m emg2qwerty.train --config-name frontend \
  user=user0 checkpoint=/path/to/generic.ckpt
```

Training settings are in [config/frontend.yaml](config/frontend.yaml).

## Export beam candidates

Export the top-20 candidates per four-second window using your trained frontend:

```bash
python scripts/export_beam_hypotheses.py \
  --weights /path/to/frontend.ckpt \
  --output-dir outputs/beams \
  --top-k 20 --device cuda:0
```

The exporter writes CSV files containing `split`, `session`, `window_index`,
`rank`, `prediction`, `target`, and `score`. The corrector uses the first ten
candidates from each window.

## Train the corrector

Configure the three candidate pools, validation pool, and model revision:

```bash
export SEMG_BEAM_STANDARD_ROOT=/path/to/standard-beams
export SEMG_BEAM_DEGRADED_ROOT=/path/to/degraded-beams
export SEMG_BEAM_SEVERELY_DEGRADED_ROOT=/path/to/severely-degraded-beams
export SEMG_BEAM_VALIDATION_SOURCE=severely_degraded
export SEMG_QWEN_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a

python scripts/train_beam_corrector.py
```

Each pool uses the CSV format above. `SEMG_BEAM_VALIDATION_SOURCE` selects
`standard`, `degraded`, or `severely_degraded`. Training samples these pools
in proportions 0.2, 0.3, and 0.5.

The configuration in [config/beam_corrector.yaml](config/beam_corrector.yaml)
uses Qwen3.5-9B with LoRA, five training epochs, and a global batch size of
eight. Training saves the best and latest `.ckpt` files in the run's
`checkpoints/` directory. The prompt is in
[config/templates/beam_correction.txt](config/templates/beam_correction.txt).

## Evaluate

Evaluate the frontend on complete sessions:

```bash
python scripts/evaluate_frontend_sessions.py \
  --weights /path/to/frontend.ckpt \
  --output-dir outputs/frontend-evaluation \
  --override decoder=ctc_beam --device cuda:0
```

Evaluate a trained corrector on one user's beam candidates:

```bash
python scripts/evaluate_beam_corrector.py \
  --topk-root /path/to/user0-test-beams \
  --checkpoint /path/to/corrector.ckpt \
  --output-dir outputs/correction-user0 \
  --user user0 --device cuda:0
```

The corrector evaluator writes chunk predictions, reconstructed sessions, and
character and word error rates (CER/WER). See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for training parameters, window
handling, and analysis commands.

## Repository layout

```text
emg2qwerty/transcription/   frontend, corrector, and data processing
config/                    model, training, data, and decoder settings
scripts/                   training, export, and evaluation
scripts/analysis/          error analysis and latency measurements
emg2qwerty/tests/           model and data tests
scripts/analysis/tests/    analysis tests
```

## Tests

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -q emg2qwerty/tests scripts/analysis/tests
```

Contribution instructions are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you use this code or build on this work, please cite our paper using the
official [ACL Anthology BibTeX](https://aclanthology.org/2026.emnlp-main.598.bib):

```bibtex
@inproceedings{wang-etal-2026-semg,
    title = "s{EMG} Typing Transcription with Conformer Encoding and Beam-Aware Large Language Model Correction",
    author = "Wang, Yuanxi  and
      Shu, Lun  and
      Koike, Yasuharu",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Junyi Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.598/",
    doi = "10.18653/2026.emnlp-main.598",
    pages = "13763--13780",
    ISBN = "979-8-89176-459-0"
}
```

## Acknowledgments and license

This work builds on [emg2qwerty](https://github.com/facebookresearch/emg2qwerty).
The code is distributed under the license in [LICENSE](LICENSE). Attribution
and dependency information are in [NOTICE.md](NOTICE.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

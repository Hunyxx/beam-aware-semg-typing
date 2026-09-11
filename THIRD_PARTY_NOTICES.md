# Third-party acknowledgments

## emg2qwerty

The dataset interface, CTC decoding utilities, and supporting infrastructure
build on [emg2qwerty](https://github.com/facebookresearch/emg2qwerty).
Its license is retained in [LICENSE](LICENSE).

```bibtex
@inproceedings{emg2qwerty_neurips2024,
  author = {Sivakumar, Viswanath and Seely, Jeffrey and Du, Alan and
            Bittner, Sean R and Berenzweig, Adam and Bolarinwa, Anuoluwapo and
            Gramfort, Alexandre and Mandel, Michael I},
  title = {emg2qwerty: A Large Dataset with Baselines for Touch Typing using
           Surface Electromyography},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {37},
  year = {2024}
}
```

## Qwen3.5

The corrector uses [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
through Transformers and PEFT. The model card provides its documentation and
license.

## Character language model

The CTC beam decoder uses the emg2qwerty character language model, built from
WikiText-103 with KenLM. The model files are in [models/lm](models/lm).

# Contributing

Contributions to the models, data processing, documentation, and tests are
welcome.

## Issues and pull requests

For a bug report, include the command, configuration, dependency versions, and
error message. A small input that reproduces the problem is helpful.

For a pull request, describe the behavior being changed and how you checked it.
Update the relevant configuration or documentation when changing a user-facing
option.

## Development checks

Activate the environment described in [README.md](README.md), then run:

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -q emg2qwerty/tests scripts/analysis/tests

pre-commit run --all-files
```

The tests cover frontend training, checkpoint continuation, tokenizer setup,
adapter loading, beam processing, and analysis. The pre-commit configuration
provides formatting, linting, and type checks. Use a working branch for changes.

## License

Contributions use the license in [LICENSE](LICENSE). Preserve source attribution
and copyright notices. Community participation follows
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

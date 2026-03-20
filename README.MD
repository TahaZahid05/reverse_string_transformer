# String Reversing Transformer

Train a decoder-only transformer on a synthetic string-reversal task using PyTorch and PyTorch-Ignite.

## Quick start

Install dependencies:

```bash
pip install torch pytorch-ignite pyyaml
```

Run training:

```bash
python main.py configs/config.yaml
```

Run a quick test:

```bash
python main.py configs/config.yaml --epochs 1
```

Save outputs to a custom location:

```bash
python main.py configs/config.yaml --output outputs
```

## What the training command shows

- run info (device, seed, epoch count, dataset sizes)
- training progress bar with loss
- per-epoch train/validation loss
- in-distribution sample predictions
- out-of-distribution sample predictions
- exact-match accuracy on validation and test sets

## Config file

Main settings are in `configs/config.yaml`:

- `run_name`, `seed`, `device`
- `data` (sizes, string lengths, batch size, workers)
- `model` (transformer hyperparameters)
- `optim` (learning rate)
- `trainer` (epochs, early stopping)
- `eval` (example strings, OOD lengths)
- `output` (output directory, checkpoint options)

## Output structure

Each run is saved to:

```text
<output_root>/<run_name>/<timestamp>/
```

Run directory includes:

- a copy of the config file
- checkpoint files (if enabled)

## Project layout

```text
configs/
  config.yaml

src/trainite/
  config.py
  inference.py
  data/
    tokenizer.py
    string_reverse_dataset.py
  models/
    decoder_transformer.py
  engine/
    trainer.py

main.py
```


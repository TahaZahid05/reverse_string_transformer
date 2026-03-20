import argparse
import datetime
import pathlib
import random
import shutil

import torch

from src.trainite.config import load_config
from src.trainite.data.string_reverse_dataset import build_dataloaders
from src.trainite.data.tokenizer import PAD_ID, TOKEN_MAPPING, VOCAB_SIZE
from src.trainite.engine.trainer import attach_validation, build_engines
from src.trainite.inference import compute_sequence_accuracy, run_example_checks, run_ood_checks
from src.trainite.models.decoder_transformer import DecoderOnlyTransformer


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a string reversing transformer model.")
	parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
	parser.add_argument("--output", type=str, default=None, help="Optional output root directory override.")
	parser.add_argument("--epochs", type=int, default=None, help="Optional number of epochs override.")
	return parser.parse_args()


def set_seed(seed: int):
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> torch.device:
	if device_cfg == "auto":
		return torch.device("cuda" if torch.cuda.is_available() else "cpu")
	return torch.device(device_cfg)


def main() -> None:
	args = parse_args()
	config_path = pathlib.Path(args.config).resolve()
	cfg = load_config(str(config_path))

	if args.epochs is not None:
		cfg.trainer_config.epochs = args.epochs

	set_seed(cfg.seed)
	device = resolve_device(cfg.device)

	output_root = pathlib.Path(args.output) if args.output else pathlib.Path(cfg.output_config.root_dir)
	run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	run_dir = output_root / cfg.run_name / run_stamp
	checkpoint_dir = run_dir / "checkpoints"
	run_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	shutil.copy2(config_path, run_dir / config_path.name)

	train_dataset, val_dataset, test_dataset, train_loader, val_loader, _test_loader = build_dataloaders(
		cfg.data_config,
		TOKEN_MAPPING,
	)

	model = DecoderOnlyTransformer(
		vocab_size=VOCAB_SIZE,
		d_model=cfg.model_config.d_model,
		nhead=cfg.model_config.nhead,
		num_layers=cfg.model_config.num_layers,
		dim_feedforward=cfg.model_config.dim_feedforward,
		max_seq_len=cfg.model_config.max_seq_len,
		dropout=cfg.model_config.dropout,
	).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim_config.lr)

	trainer, evaluator = build_engines(
		model=model,
		optimizer=optimizer,
		device=device,
		pad_id=PAD_ID,
		vocab_size=VOCAB_SIZE,
		patience=cfg.trainer_config.patience,
		checkpoint_dir=str(checkpoint_dir),
		n_saved=cfg.output_config.n_saved,
		save_checkpoints=cfg.output_config.save_checkpoints,
	)
	attach_validation(trainer, evaluator, val_loader)

	print(f"Run name: {cfg.run_name}")
	print(f"Device: {device}")
	print(f"Seed: {cfg.seed}")
	print(f"Epochs: {cfg.trainer_config.epochs} | Batch size: {cfg.data_config.batch_size}")
	print(
		"Dataset sizes: "
		f"train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
	)
	print(
		f"Checkpoints: {'enabled' if cfg.output_config.save_checkpoints else 'disabled'}"
		+ (f" (dir: {checkpoint_dir})" if cfg.output_config.save_checkpoints else "")
	)
	print(f"Run directory: {run_dir}")
	print("Starting training...")

	trainer.run(train_loader, max_epochs=cfg.trainer_config.epochs)

	print("\nChecking on some examples (In-distribution)")
	run_example_checks(model, cfg.eval_config.sample_strings, device)

	print("\nChecking on some examples (Out-of-distribution)")
	run_ood_checks(model, cfg.eval_config.ood_lengths, device)

	val_accuracy = compute_sequence_accuracy(model, val_dataset, device)
	print(f"\nExact-match accuracy (val set): {val_accuracy:.1%}")

	test_accuracy = compute_sequence_accuracy(model, test_dataset, device)
	print(f"Exact-match accuracy (test set): {test_accuracy:.1%}")

	print(f"\nRun outputs saved to: {run_dir}")


if __name__ == "__main__":
	main()


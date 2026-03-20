import os
from typing import Callable

import torch
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint, ProgressBar
from ignite.metrics import RunningAverage

def compute_masked_loss(
    logits: torch.Tensor,
    target_seq: torch.Tensor,
    mask: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    logits_flat = logits.reshape(-1, vocab_size)
    target_flat = target_seq.reshape(-1)
    mask_flat = mask.reshape(-1)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    losses_flat = loss_fn(logits_flat, target_flat)
    masked_losses = losses_flat * mask_flat
    mean_loss = masked_losses.sum() / mask_flat.sum()

    return mean_loss


def make_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_id: int,
    vocab_size: int,
) -> Callable:
    def train_step(engine, batch):
        model.train()
        input_ids = batch["input_ids"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        input_seq = input_ids[:, :-1]
        target_seq = input_ids[:, 1:]
        loss_mask = loss_mask[:, 1:]

        pad_mask = (input_seq == pad_id)
        logits = model(input_seq, src_pad_mask=pad_mask)
        loss = compute_masked_loss(logits, target_seq, loss_mask, vocab_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    return train_step

def make_val_step(
    model: torch.nn.Module,
    device: torch.device,
    pad_id: int,
    vocab_size: int,
) -> Callable:
    def val_step(engine, batch):
        model.eval()
        
        input_ids = batch["input_ids"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        input_seq = input_ids[:, :-1]
        target_seq = input_ids[:, 1:]
        loss_mask = loss_mask[:, 1:]

        pad_mask = (input_seq == pad_id)
        with torch.no_grad():
            logits = model(input_seq, src_pad_mask=pad_mask)
            loss = compute_masked_loss(logits, target_seq, loss_mask, vocab_size)

        return loss.item()

    return val_step


def build_engines(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_id: int,
    vocab_size: int,
    patience: int,
    checkpoint_dir: str,
    n_saved: int = 1,
    save_checkpoints: bool = True,
) -> tuple[Engine, Engine]:
    trainer = Engine(make_train_step(model, optimizer, device, pad_id, vocab_size))
    evaluator = Engine(make_val_step(model, device, pad_id, vocab_size))

    RunningAverage(output_transform=lambda x: x).attach(trainer, "avg_loss")
    ProgressBar().attach(trainer, output_transform=lambda x: {"loss": x})

    def score_function(engine):
        val_loss = engine.state.output
        return -val_loss
    
    es_handler = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, es_handler)

    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_handler = ModelCheckpoint(
            checkpoint_dir,
            filename_prefix="best",
            n_saved=n_saved,
            score_function=score_function,
            score_name="neg_val_loss",
            require_empty=False,
            global_step_transform=lambda *_: trainer.state.epoch
        )
        evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {"model": model})

    return trainer, evaluator

def attach_validation(trainer: Engine, evaluator: Engine, val_loader) -> None:
    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        evaluator.run(val_loader)
        val_loss = evaluator.state.output
        train_loss = engine.state.metrics["avg_loss"]
        print(
            f"Epoch {engine.state.epoch} — "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
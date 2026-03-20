import random
import string

import torch

from .data.tokenizer import encode, decode, SEP_ID, EOS_ID

def predict(model: torch.nn.Module, input_string: str, device: torch.device) -> str:
    model.eval()

    tokens = encode(input_string) + [SEP_ID]
    with torch.no_grad():
        for _ in range(len(input_string) + 1):
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            logits = model(input_tensor)
            next_token = logits[0, -1, :].argmax().item()

            if next_token == EOS_ID:
                break
            tokens.append(next_token)

    sep_idx = len(encode(input_string))
    output_tokens = tokens[sep_idx + 1:]
    return decode(output_tokens)


def compute_sequence_accuracy(model: torch.nn.Module, dataset, device: torch.device) -> float:
    model.eval()
    correct = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            # Subset indexes first need to be mapped to original and then we can access data.
            if hasattr(dataset, 'dataset'):
                real_idx = dataset.indices[idx]
                original = dataset.dataset.all_samples[real_idx]
            else:
                original = dataset.all_samples[idx]

            input_str = original["input"]
            expected = original["output"]
            predicted = predict(model, input_str, device)

            if predicted == expected:
                correct += 1

    return correct / len(dataset)


def run_example_checks(model: torch.nn.Module, examples: list[str], device: torch.device) -> None:
    for sample in examples:
        predicted = predict(model, sample, device)
        expected = sample[::-1]
        status = "Correct!" if predicted == expected else "Incorrect!"
        print(f"  {status}  {sample} -> {predicted} (expected: {expected})")


def run_ood_checks(model: torch.nn.Module, lengths: list[int], device: torch.device) -> None:
    for length in lengths:
        sample = "".join(random.choices(string.ascii_lowercase, k=length))
        predicted = predict(model, sample, device)
        expected = sample[::-1]
        status = "Correct!" if predicted == expected else "Incorrect!"
        print(f"  {status}  len={length}: {sample} -> {predicted} (expected: {expected})")
    
import random
from math import ceil

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


MODEL_NAME = "FacebookAI/xlm-roberta-base"
MAX_LENGTH = 64
EPOCHS = 5
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

languages = [
    ("English", "en-US"),
    ("German", "de-DE"),
    ("Russian", "ru-RU"),
]

split_names = {
    "en-US": "en_US",
    "de-DE": "de_DE",
    "ru-RU": "ru_RU",
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_split(split, tokenizer):
    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch["utt"],
            max_length=MAX_LENGTH,
            truncation=True,
        )
        # MASSIVE's existing ClassLabel IDs are already the model's target IDs.
        tokenized["labels"] = batch["intent"]
        return tokenized

    return split.map(
        tokenize_batch,
        batched=True,
        remove_columns=split.column_names,
    )


def evaluate(
    model,
    data_loader,
    device,
    amp_dtype,
):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions = []
    gold_labels = []

    with torch.inference_mode():
        for batch in data_loader:
            batch = {
                name: tensor.to(device, non_blocking=True)
                for name, tensor in batch.items()
            }
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(**batch)

            batch_size = batch["labels"].size(0)
            total_loss += outputs.loss.item() * batch_size
            total_examples += batch_size
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            gold_labels.extend(batch["labels"].cpu().tolist())

    metrics = {
        "loss": total_loss / total_examples,
        "accuracy": accuracy_score(gold_labels, predictions),
        "macro_f1": f1_score(gold_labels, predictions, average="macro"),
    }
    return metrics


def train_epoch(
    model,
    data_loader,
    optimizer,
    scheduler,
    device,
    amp_dtype,
    scaler,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_examples = 0
    number_of_batches = len(data_loader)

    for batch_index, batch in enumerate(data_loader, start=1):
        batch = {
            name: tensor.to(device, non_blocking=True)
            for name, tensor in batch.items()
        }

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(**batch)

        batch_size = batch["labels"].size(0)
        total_loss += outputs.loss.item() * batch_size
        total_examples += batch_size

        # A shorter final accumulation group must not receive a reduced gradient.
        group_start = (
            (batch_index - 1) // GRADIENT_ACCUMULATION_STEPS
        ) * GRADIENT_ACCUMULATION_STEPS
        accumulation_group_size = min(
            GRADIENT_ACCUMULATION_STEPS,
            number_of_batches - group_start,
        )
        loss_for_backward = outputs.loss / accumulation_group_size
        if scaler is None:
            loss_for_backward.backward()
        else:
            scaler.scale(loss_for_backward).backward()

        should_update = (
            batch_index % GRADIENT_ACCUMULATION_STEPS == 0
            or batch_index == number_of_batches
        )
        if should_update:
            if scaler is None:
                optimizer.step()
                scheduler.step()
            else:
                scale_before_update = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before_update:
                    scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / total_examples


def main():
    print("XLM-R zero-shot cross-lingual transfer", flush=True)
    print("\nGPU check")
    print(f"torch version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal benchmark")

    device = torch.device("cuda:0")
    print(f"CUDA device name: {torch.cuda.get_device_name(device)}")
    if torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        precision_name = "BF16"
    else:
        amp_dtype = torch.float16
        precision_name = "FP16"
    print(f"Mixed precision: {precision_name}")

    set_seed(SEED)

    print("\nFrozen training configuration")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Train batch size: {TRAIN_BATCH_SIZE}")
    print(f"Evaluation batch size: {EVAL_BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(
        "Effective training batch size: "
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Warmup ratio: {WARMUP_RATIO}")
    print(f"Maximum token length: {MAX_LENGTH}")
    print(f"Seed: {SEED}")

    print("Training language: English (en-US) only")
    print("Evaluation checkpoint: final epoch (no checkpoint selection)")

    data_files = {
        "train": (
            "hf://datasets/AmazonScience/massive@~parquet/"
            "en-US/train/0000.parquet"
        ),
    }
    for language, locale in languages:
        data_files[split_names[locale]] = (
            "hf://datasets/AmazonScience/massive@~parquet/"
            f"{locale}/validation/0000.parquet"
        )
    dataset = load_dataset("parquet", data_files=data_files)
    train_split = dataset["train"]

    print("\nDataset sanity checks")
    print(f"English train: {len(train_split)}")
    print("Input: utt; target: intent")
    if len(train_split) != 11514:
        raise ValueError("Unexpected English MASSIVE train size")
    intent_feature = train_split.features["intent"]
    if intent_feature.num_classes != 60:
        raise ValueError(f"Expected 60 intents, got {intent_feature.num_classes}")
    if set(train_split["locale"]) != {"en-US"}:
        raise ValueError("Training data must contain English only")
    for language, locale in languages:
        split = dataset[split_names[locale]]
        print(f"{language} ({locale}) validation: {len(split)}")
        if len(split) != 2033:
            raise ValueError(f"Unexpected {language} MASSIVE validation size")
        feature = split.features["intent"]
        if feature.num_classes != 60 or feature.names != intent_feature.names:
            raise ValueError(f"{language} intent schema differs from English train")
        if set(split["locale"]) != {locale}:
            raise ValueError(f"Unexpected locale in {language} validation")
    print("All splits share the same 60-intent ClassLabel schema")
    print("Original MASSIVE ClassLabel IDs retained")

    id2label = {
        intent_id: intent_feature.int2str(intent_id)
        for intent_id in range(intent_feature.num_classes)
    }
    label2id = {
        intent_name: intent_id
        for intent_id, intent_name in id2label.items()
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_train = tokenize_split(train_split, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)
    train_loader = DataLoader(
        tokenized_train,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True,
        generator=train_generator,
    )
    validation_loaders = {}
    for language, locale in languages:
        tokenized_validation = tokenize_split(
            dataset[split_names[locale]], tokenizer,
        )
        validation_loaders[language] = DataLoader(
            tokenized_validation,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            collate_fn=data_collator,
            pin_memory=True,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=intent_feature.num_classes,
        label2id=label2id,
        id2label=id2label,
    ).to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if trainable_parameters != total_parameters:
        raise RuntimeError("The full XLM-R model should be trainable")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer_steps_per_epoch = ceil(
        len(train_loader) / GRADIENT_ACCUMULATION_STEPS
    )
    total_optimizer_steps = optimizer_steps_per_epoch * EPOCHS
    warmup_steps = int(total_optimizer_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    if amp_dtype == torch.bfloat16:
        scaler = None
    else:
        scaler = torch.amp.GradScaler("cuda")
    print(f"Trainable parameters: {trainable_parameters:,}/{total_parameters:,}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Total optimizer steps: {total_optimizer_steps}")
    print("Dynamic batch padding enabled", flush=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            amp_dtype,
            scaler,
        )
        print(f"\nEpoch {epoch}/{EPOCHS}: English train loss {train_loss:.4f}", flush=True)

    # Evaluate one fixed final model; target languages never guide training.
    results = {}
    for language, locale in languages:
        results[language] = evaluate(
            model, validation_loaders[language], device, amp_dtype,
        )

    print("\nFinal-epoch cross-lingual validation results")
    print(
        f"{'Train language':<16} {'Evaluation language':<20} "
        f"{'Accuracy':>10} {'Macro-F1':>10}"
    )
    print("-" * 59)
    for language, _ in languages:
        metrics = results[language]
        print(
            f"{'English':<16} {language:<20} "
            f"{metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f}"
        )

    print("\nTransfer gaps (English score - target score)")
    print("Positive values mean a drop; units are score points (0-1 scale).")
    print(f"{'Transfer':<22} {'Accuracy drop':>14} {'Macro-F1 drop':>14}")
    print("-" * 52)
    english = results["English"]
    for language, _ in languages[1:]:
        metrics = results[language]
        print(
            f"{('English -> ' + language):<22} "
            f"{english['accuracy'] - metrics['accuracy']:>+14.4f} "
            f"{english['macro_f1'] - metrics['macro_f1']:>+14.4f}"
        )


if __name__ == "__main__":
    main()

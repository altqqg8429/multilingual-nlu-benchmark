import random
from math import ceil
from statistics import median

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
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

classical_results = {
    "English": {
        "Word TF-IDF": {"accuracy": 0.8057, "macro_f1": 0.7619},
        "Char TF-IDF": {"accuracy": 0.8175, "macro_f1": 0.7634},
    },
    "German": {
        "Word TF-IDF": {"accuracy": 0.7688, "macro_f1": 0.6920},
        "Char TF-IDF": {"accuracy": 0.7964, "macro_f1": 0.7444},
    },
    "Russian": {
        "Word TF-IDF": {"accuracy": 0.7614, "macro_f1": 0.7105},
        "Char TF-IDF": {"accuracy": 0.7954, "macro_f1": 0.7514},
    },
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
    return_predictions=False,
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
    if return_predictions:
        metrics["gold_labels"] = gold_labels
        metrics["predictions"] = predictions
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


def run_language(language, locale, tokenizer, device, amp_dtype):
    set_seed(SEED)

    print(f"\n{'=' * 64}")
    print(f"{language} ({locale})")
    print("=" * 64, flush=True)
    data_files = {
        "train": (
            "hf://datasets/AmazonScience/massive@~parquet/"
            f"{locale}/train/0000.parquet"
        ),
        "validation": (
            "hf://datasets/AmazonScience/massive@~parquet/"
            f"{locale}/validation/0000.parquet"
        ),
    }
    dataset = load_dataset("parquet", data_files=data_files)
    train_split = dataset["train"]
    validation_split = dataset["validation"]

    print("\nDataset sanity checks")
    print(
        f"{language} ({locale}): "
        f"train={len(train_split)}, validation={len(validation_split)}"
    )
    print("Input: utt")
    print("Target: intent")

    if len(train_split) != 11514 or len(validation_split) != 2033:
        raise ValueError(f"Unexpected {language} MASSIVE split sizes")

    intent_feature = train_split.features["intent"]
    validation_intent_feature = validation_split.features["intent"]
    if intent_feature.num_classes != 60:
        raise ValueError(f"Expected 60 intents, got {intent_feature.num_classes}")
    if validation_intent_feature.names != intent_feature.names:
        raise ValueError("Train and validation intent schemas do not match")

    id2label = {
        intent_id: intent_feature.int2str(intent_id)
        for intent_id in range(intent_feature.num_classes)
    }
    label2id = {
        intent_name: intent_id
        for intent_id, intent_name in id2label.items()
    }
    print(f"Intent classes: {intent_feature.num_classes}")
    print("Original MASSIVE ClassLabel IDs retained")
    print("Train and validation intent schemas match")

    full_encoding = tokenizer(
        list(train_split["utt"]),
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    lengths = [len(input_ids) for input_ids in full_encoding["input_ids"]]
    over_limit = sum(length > MAX_LENGTH for length in lengths)
    length_stats = {
        "maximum": max(lengths),
        "median": median(lengths),
        "over_limit": over_limit,
        "over_limit_percentage": over_limit / len(lengths),
    }
    print("\nTrain-only token lengths (including special tokens)")
    print(f"Maximum: {length_stats['maximum']}")
    print(f"Median: {length_stats['median']}")
    print(
        f"Exceeding {MAX_LENGTH}: {over_limit}/{len(lengths)} "
        f"({length_stats['over_limit_percentage']:.2%})"
    )

    tokenized_train = tokenize_split(train_split, tokenizer)
    tokenized_validation = tokenize_split(validation_split, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if amp_dtype == torch.bfloat16:
        scaler = None
    else:
        scaler = torch.amp.GradScaler("cuda")

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
    validation_loader = DataLoader(
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

    print("\nTraining configuration")
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
    print(f"Warmup steps: {warmup_steps}")
    print(f"Total optimizer steps: {total_optimizer_steps}")
    print(f"Maximum token length: {MAX_LENGTH} (dynamic batch padding)")
    print(f"Seed: {SEED}")
    print(f"Trainable parameters: {trainable_parameters:,}/{total_parameters:,}")

    epoch_results = []
    best_result = None
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
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            amp_dtype,
            return_predictions=epoch == EPOCHS,
        )
        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_metrics["loss"],
            "accuracy": validation_metrics["accuracy"],
            "macro_f1": validation_metrics["macro_f1"],
        }
        epoch_results.append(result)
        if best_result is None or result["macro_f1"] > best_result["macro_f1"]:
            best_result = result

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_metrics['loss']:.4f}")
        print(f"Validation Accuracy: {validation_metrics['accuracy']:.4f}")
        print(f"Validation Macro-F1: {validation_metrics['macro_f1']:.4f}")

    print(f"\nBest {language} validation result")
    print(f"Epoch: {best_result['epoch']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"Macro-F1: {best_result['macro_f1']:.4f}")

    gold_labels = validation_metrics["gold_labels"]
    predictions = validation_metrics["predictions"]
    gold_intent_ids = sorted(set(gold_labels))
    _, _, per_class_f1, _ = precision_recall_fscore_support(
        gold_labels,
        predictions,
        labels=gold_intent_ids,
        zero_division=0,
    )
    print("\nFinal-epoch validation class diagnostic")
    print(f"Distinct gold intent classes: {len(gold_intent_ids)}")
    print(f"Distinct predicted intent classes: {len(set(predictions))}")
    zero_f1_count = sum(f1 == 0 for f1 in per_class_f1)
    print(f"Gold-present intents with F1 == 0: {zero_f1_count}")

    language_result = {
        "language": language,
        "locale": locale,
        "epochs": epoch_results,
        "best_epoch": best_result["epoch"],
        "accuracy": best_result["accuracy"],
        "macro_f1": best_result["macro_f1"],
        "length_stats": length_stats,
        "gold_classes": len(gold_intent_ids),
        "predicted_classes": len(set(predictions)),
        "zero_f1_classes": zero_f1_count,
    }

    del model, optimizer, scheduler
    del train_loader, validation_loader, tokenized_train, tokenized_validation
    if scaler is not None:
        del scaler
    torch.cuda.empty_cache()
    return language_result


def main():
    print("Multilingual XLM-R fine-tuning benchmark", flush=True)
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    results = []
    for language, locale in languages:
        results.append(
            run_language(
                language,
                locale,
                tokenizer,
                device,
                amp_dtype,
            )
        )

    print("\nFinal XLM-R validation results")
    print(f"{'Language':<10} {'Best epoch':>10} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 44)
    for result in results:
        print(
            f"{result['language']:<10} "
            f"{result['best_epoch']:>10} "
            f"{result['accuracy']:>10.4f} "
            f"{result['macro_f1']:>10.4f}"
        )

    print("\nClassical and XLM-R comparison")
    print(f"{'Language':<10} {'Representation':<15} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 49)
    for result in results:
        language = result["language"]
        for representation in ["Word TF-IDF", "Char TF-IDF"]:
            scores = classical_results[language][representation]
            print(
                f"{language:<10} {representation:<15} "
                f"{scores['accuracy']:>10.4f} "
                f"{scores['macro_f1']:>10.4f}"
            )
        print(
            f"{language:<10} {'XLM-R':<15} "
            f"{result['accuracy']:>10.4f} "
            f"{result['macro_f1']:>10.4f}"
        )

    print("\nXLM-R deltas over classical baselines")
    print(f"{'Language':<10} {'Comparison':<22} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 56)
    for result in results:
        language = result["language"]
        for representation in ["Word TF-IDF", "Char TF-IDF"]:
            baseline = classical_results[language][representation]
            print(
                f"{language:<10} {('XLM-R - ' + representation):<22} "
                f"{result['accuracy'] - baseline['accuracy']:>+10.4f} "
                f"{result['macro_f1'] - baseline['macro_f1']:>+10.4f}"
            )

    result_by_language = {
        result["language"]: result for result in results
    }
    print("\nCross-language score gaps (maximum - minimum)")
    print(f"{'Representation':<15} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 37)
    for representation in ["Word TF-IDF", "Char TF-IDF", "XLM-R"]:
        if representation == "XLM-R":
            accuracies = [result["accuracy"] for result in results]
            macro_f1_scores = [result["macro_f1"] for result in results]
        else:
            accuracies = [
                classical_results[language][representation]["accuracy"]
                for language, _ in languages
            ]
            macro_f1_scores = [
                classical_results[language][representation]["macro_f1"]
                for language, _ in languages
            ]
        print(
            f"{representation:<15} "
            f"{max(accuracies) - min(accuracies):>10.4f} "
            f"{max(macro_f1_scores) - min(macro_f1_scores):>10.4f}"
        )

    print("\nFinal-epoch class coverage")
    print(f"{'Language':<10} {'Gold':>6} {'Predicted':>10} {'Zero F1':>8}")
    print("-" * 38)
    for language, _ in languages:
        result = result_by_language[language]
        print(
            f"{language:<10} "
            f"{result['gold_classes']:>6} "
            f"{result['predicted_classes']:>10} "
            f"{result['zero_f1_classes']:>8}"
        )


if __name__ == "__main__":
    main()

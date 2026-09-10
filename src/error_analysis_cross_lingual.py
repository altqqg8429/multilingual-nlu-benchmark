import csv
from collections import Counter
from pathlib import Path

from sklearn.metrics import classification_report


results_dir = Path(__file__).resolve().parent.parent / "results"
with (results_dir / "xlmr_cross_lingual_predictions.csv").open(
    newline="", encoding="utf-8"
) as file:
    cross_lingual_rows = list(csv.DictReader(file))

for language in ["German", "Russian"]:
    print(f"\n{'=' * 80}\n{language}\n{'=' * 80}")
    with (results_dir / f"xlmr_predictions_{language}.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        monolingual_rows = list(csv.DictReader(file))
    target_rows = [
        row for row in cross_lingual_rows
        if row["evaluation_language"] == language
    ]

    if not monolingual_rows or len(monolingual_rows) != len(target_rows):
        raise ValueError(f"{language}: empty predictions or example count mismatch")
    for row_number, (mono_row, cross_row) in enumerate(
        zip(monolingual_rows, target_rows), start=1
    ):
        if cross_row["train_language"] != "English":
            raise ValueError(f"{language}: expected English-trained predictions")
        if (
            mono_row["utterance"] != cross_row["utterance"]
            or int(mono_row["true_id"]) != int(cross_row["true_id"])
            or mono_row["true_intent"] != cross_row["true_intent"]
        ):
            raise ValueError(f"{language}: alignment mismatch at example {row_number}")
    print(f"Alignment verified for {len(monolingual_rows)} validation examples")

    gold_labels = [int(row["true_id"]) for row in monolingual_rows]
    monolingual_predictions = [int(row["predicted_id"]) for row in monolingual_rows]
    cross_lingual_predictions = [int(row["predicted_id"]) for row in target_rows]
    groups = {
        "Both models correct": [],
        "Target-language XLM-R correct, cross-lingual XLM-R wrong": [],
        "Target-language XLM-R wrong, cross-lingual XLM-R correct": [],
        "Both models wrong": [],
    }
    for mono_row, cross_row, true_id, mono_id, cross_id in zip(
        monolingual_rows, target_rows, gold_labels,
        monolingual_predictions, cross_lingual_predictions,
    ):
        record = {
            "utterance": mono_row["utterance"],
            "true_intent": mono_row["true_intent"],
            "monolingual_intent": mono_row["predicted_intent"],
            "cross_lingual_intent": cross_row["predicted_intent"],
        }
        if mono_id == true_id and cross_id == true_id:
            groups["Both models correct"].append(record)
        elif mono_id == true_id:
            groups["Target-language XLM-R correct, cross-lingual XLM-R wrong"].append(record)
        elif cross_id == true_id:
            groups["Target-language XLM-R wrong, cross-lingual XLM-R correct"].append(record)
        else:
            groups["Both models wrong"].append(record)

    print("\nComparison (percentages of all validation examples)")
    for group, records in groups.items():
        print(f"{group}: {len(records)} ({len(records) / len(gold_labels):.2%})")

    transfer_errors = groups[
        "Target-language XLM-R correct, cross-lingual XLM-R wrong"
    ]
    confusion_counts = Counter(
        (error["true_intent"], error["cross_lingual_intent"])
        for error in transfer_errors
    )
    top_confusions = confusion_counts.most_common(10)
    print("\nTop 10 transfer-induced confusion pairs (true -> cross-lingual -> count)")
    for (true_intent, predicted_intent), count in top_confusions:
        print(f"{true_intent} -> {predicted_intent} -> {count}")

    print("\nExamples for the top 5 transfer-induced pairs")
    for (true_intent, predicted_intent), _ in top_confusions[:5]:
        examples = [
            error["utterance"]
            for error in transfer_errors
            if error["true_intent"] == true_intent
            and error["cross_lingual_intent"] == predicted_intent
        ][:3]
        print(f"\nTrue intent / target-language XLM-R: {true_intent}")
        print(f"Cross-lingual prediction: {predicted_intent}")
        for utterance in examples:
            print(f"- {utterance}")

    id2label = {
        int(row["true_id"]): row["true_intent"] for row in monolingual_rows
    }
    present_labels = sorted(set(gold_labels))
    present_intent_names = [id2label[label_id] for label_id in present_labels]
    monolingual_report = classification_report(
        gold_labels,
        monolingual_predictions,
        labels=present_labels,
        target_names=present_intent_names,
        output_dict=True,
        zero_division=0,
    )
    cross_lingual_report = classification_report(
        gold_labels,
        cross_lingual_predictions,
        labels=present_labels,
        target_names=present_intent_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_results = []
    for intent_name in present_intent_names:
        monolingual_f1 = monolingual_report[intent_name]["f1-score"]
        cross_lingual_f1 = cross_lingual_report[intent_name]["f1-score"]
        per_class_results.append(
            {
                "intent": intent_name,
                "support": int(monolingual_report[intent_name]["support"]),
                "monolingual_f1": monolingual_f1,
                "cross_lingual_f1": cross_lingual_f1,
                "delta": cross_lingual_f1 - monolingual_f1,
            }
        )

    largest_drops = sorted(
        per_class_results, key=lambda result: result["delta"],
    )[:10]
    largest_gains = sorted(
        per_class_results, key=lambda result: result["delta"], reverse=True,
    )[:10]
    for title, class_results in [
        ("10 largest cross-lingual F1 drops", largest_drops),
        ("10 largest cross-lingual F1 gains", largest_gains),
    ]:
        print(f"\n{title}")
        print(
            f"{'Intent':<35} {'Support':>7} {'Monolingual F1':>14} "
            f"{'Cross-lingual F1':>16} {'Delta':>9}"
        )
        print("-" * 85)
        for result in class_results:
            print(
                f"{result['intent']:<35} {result['support']:>7} "
                f"{result['monolingual_f1']:>14.4f} "
                f"{result['cross_lingual_f1']:>16.4f} "
                f"{result['delta']:>+9.4f}"
            )

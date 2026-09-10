import csv
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


data_files = {
    "train": "hf://datasets/AmazonScience/massive@~parquet/en-US/train/0000.parquet",
    "validation": "hf://datasets/AmazonScience/massive@~parquet/en-US/validation/0000.parquet",
}
dataset = load_dataset("parquet", data_files=data_files)

X_train = dataset["train"]["utt"]
y_train = dataset["train"]["intent"]
X_val = dataset["validation"]["utt"]
y_val = dataset["validation"]["intent"]
intent_names = dataset["train"].features["intent"].names

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
tfidf_predictions = model.predict(X_val_tfidf)

results_dir = Path(__file__).resolve().parent.parent / "results"
with (results_dir / "xlmr_predictions_English.csv").open(
    newline="", encoding="utf-8"
) as file:
    xlmr_rows = list(csv.DictReader(file))

if len(xlmr_rows) != len(X_val):
    raise ValueError("XLM-R prediction count differs from validation size")

xlmr_predictions = []
for row_number, (row, utterance, true_id) in enumerate(
    zip(xlmr_rows, X_val, y_val), start=1
):
    if row["utterance"] != utterance:
        raise ValueError(f"Validation utterance mismatch at example {row_number}")
    if (
        int(row["true_id"]) != true_id
        or row["true_intent"] != intent_names[true_id]
    ):
        raise ValueError(f"Gold intent mismatch at example {row_number}")
    predicted_id = int(row["predicted_id"])
    if (
        not 0 <= predicted_id < len(intent_names)
        or row["predicted_intent"] != intent_names[predicted_id]
    ):
        raise ValueError(f"Predicted intent mismatch at example {row_number}")
    xlmr_predictions.append(predicted_id)
print(f"Alignment verified for {len(X_val)} validation examples")

groups = {
    "Both models correct": [],
    "TF-IDF wrong, XLM-R correct": [],
    "TF-IDF correct, XLM-R wrong": [],
    "Both models wrong": [],
}
for utterance, true_id, tfidf_id, xlmr_id in zip(
    X_val, y_val, tfidf_predictions, xlmr_predictions
):
    record = {
        "utterance": utterance,
        "true_intent": intent_names[true_id],
        "tfidf_intent": intent_names[tfidf_id],
        "xlmr_intent": intent_names[xlmr_id],
    }
    if tfidf_id == true_id and xlmr_id == true_id:
        groups["Both models correct"].append(record)
    elif xlmr_id == true_id:
        groups["TF-IDF wrong, XLM-R correct"].append(record)
    elif tfidf_id == true_id:
        groups["TF-IDF correct, XLM-R wrong"].append(record)
    else:
        groups["Both models wrong"].append(record)

print("\nValidation comparison (percentages of all validation examples)")
for group, records in groups.items():
    print(f"{group}: {len(records)} ({len(records) / len(X_val):.2%})")

xlmr_errors = (
    groups["TF-IDF correct, XLM-R wrong"] + groups["Both models wrong"]
)
confusion_counts = Counter(
    (error["true_intent"], error["xlmr_intent"])
    for error in xlmr_errors
)
print("\nTop 15 XLM-R confusion pairs (true -> XLM-R -> count)")
for (true_intent, predicted_intent), count in confusion_counts.most_common(15):
    print(f"{true_intent} -> {predicted_intent} -> {count}")

solved_errors = groups["TF-IDF wrong, XLM-R correct"]
solved_counts = Counter(
    (error["true_intent"], error["tfidf_intent"])
    for error in solved_errors
)
top_solved = solved_counts.most_common(10)
print("\nTop 10 TF-IDF confusion pairs solved by XLM-R (true -> TF-IDF -> count)")
for (true_intent, predicted_intent), count in top_solved:
    print(f"{true_intent} -> {predicted_intent} -> {count}")

print("\nExamples for the top 5 solved pairs")
for (true_intent, predicted_intent), _ in top_solved[:5]:
    examples = [
        error["utterance"]
        for error in solved_errors
        if error["true_intent"] == true_intent
        and error["tfidf_intent"] == predicted_intent
    ][:3]
    print(f"\nTrue intent / XLM-R: {true_intent}")
    print(f"TF-IDF prediction: {predicted_intent}")
    for utterance in examples:
        print(f"- {utterance}")

remaining_errors = groups["Both models wrong"]
remaining_counts = Counter(
    (error["true_intent"], error["xlmr_intent"])
    for error in remaining_errors
)
top_remaining = remaining_counts.most_common(10)
print("\nTop 10 pairs where both models are wrong (true -> XLM-R -> count)")
for (true_intent, predicted_intent), count in top_remaining:
    print(f"{true_intent} -> {predicted_intent} -> {count}")

print("\nExamples for the top 5 pairs where both models are wrong")
for (true_intent, predicted_intent), _ in top_remaining[:5]:
    examples = [
        error
        for error in remaining_errors
        if error["true_intent"] == true_intent
        and error["xlmr_intent"] == predicted_intent
    ][:3]
    print(f"\nTrue intent: {true_intent}")
    print(f"XLM-R prediction: {predicted_intent}")
    for error in examples:
        print(f"- {error['utterance']} (TF-IDF: {error['tfidf_intent']})")

present_labels = sorted(set(y_val))
present_intent_names = [intent_names[label_id] for label_id in present_labels]

tfidf_report = classification_report(
    y_val,
    tfidf_predictions,
    labels=present_labels,
    target_names=present_intent_names,
    output_dict=True,
    zero_division=0,
)
xlmr_report = classification_report(
    y_val,
    xlmr_predictions,
    labels=present_labels,
    target_names=present_intent_names,
    output_dict=True,
    zero_division=0,
)

per_class_results = []
for intent_name in present_intent_names:
    tfidf_f1 = tfidf_report[intent_name]["f1-score"]
    xlmr_f1 = xlmr_report[intent_name]["f1-score"]
    per_class_results.append(
        {
            "intent": intent_name,
            "support": int(tfidf_report[intent_name]["support"]),
            "tfidf_f1": tfidf_f1,
            "xlmr_f1": xlmr_f1,
            "delta": xlmr_f1 - tfidf_f1,
        }
    )

with (results_dir / "xlmr_per_class_f1_English.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file, fieldnames=["intent", "support", "tfidf_f1", "xlmr_f1", "delta"]
    )
    writer.writeheader()
    writer.writerows(per_class_results)

largest_gains = sorted(
    per_class_results,
    key=lambda result: result["delta"],
    reverse=True,
)[:10]
largest_losses = sorted(
    per_class_results,
    key=lambda result: result["delta"],
)[:10]

for title, class_results in [
    ("10 intents with largest XLM-R F1 gains", largest_gains),
    ("10 intents with largest XLM-R F1 losses", largest_losses),
]:
    print(f"\n{title}")
    print(
        f"{'Intent':<35} {'Support':>7} "
        f"{'TF-IDF F1':>9} {'XLM-R F1':>9} {'Delta':>9}"
    )
    print("-" * 73)
    for result in class_results:
        print(
            f"{result['intent']:<35} "
            f"{result['support']:>7} "
            f"{result['tfidf_f1']:>9.4f} "
            f"{result['xlmr_f1']:>9.4f} "
            f"{result['delta']:>+9.4f}"
        )

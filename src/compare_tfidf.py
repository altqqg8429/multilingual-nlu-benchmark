from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score


data_files = {
    "train": "hf://datasets/AmazonScience/massive@~parquet/en-US/train/0000.parquet",
    "validation": "hf://datasets/AmazonScience/massive@~parquet/en-US/validation/0000.parquet",
}

dataset = load_dataset("parquet", data_files=data_files)

X_train = dataset["train"]["utt"]
y_train = dataset["train"]["intent"]
X_val = dataset["validation"]["utt"]
y_val = dataset["validation"]["intent"]

representations = [
    ("Word TF-IDF", TfidfVectorizer()),
    (
        "Character TF-IDF (3-5)",
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)),
    ),
]

results = []

for name, vectorizer in representations:
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    y_val_pred = model.predict(X_val_tfidf)

    results.append(
        {
            "representation": name,
            "accuracy": accuracy_score(y_val, y_val_pred),
            "macro_f1": f1_score(y_val, y_val_pred, average="macro"),
            "predictions": y_val_pred,
        }
    )

print("\nValidation results")
print(f"{'Representation':<28} {'Accuracy':>10} {'Macro-F1':>10}")
print("-" * 50)
for result in results:
    print(
        f"{result['representation']:<28} "
        f"{result['accuracy']:>10.4f} "
        f"{result['macro_f1']:>10.4f}"
    )

word_result, char_result = results

accuracy_delta = char_result["accuracy"] - word_result["accuracy"]
macro_f1_delta = char_result["macro_f1"] - word_result["macro_f1"]

print("\nCharacter - Word")
print(f"Accuracy: {accuracy_delta:+.4f}")
print(f"Macro-F1: {macro_f1_delta:+.4f}")

intent_names = dataset["train"].features["intent"].names
present_labels = sorted(
    set(y_val)
    | set(word_result["predictions"])
    | set(char_result["predictions"])
)
present_intent_names = [intent_names[label_id] for label_id in present_labels]

word_report = classification_report(
    y_val,
    word_result["predictions"],
    labels=present_labels,
    target_names=present_intent_names,
    output_dict=True,
    zero_division=0,
)
char_report = classification_report(
    y_val,
    char_result["predictions"],
    labels=present_labels,
    target_names=present_intent_names,
    output_dict=True,
    zero_division=0,
)

per_class_results = []
for intent_name in present_intent_names:
    support = int(word_report[intent_name]["support"])
    if support > 0:
        word_f1 = word_report[intent_name]["f1-score"]
        char_f1 = char_report[intent_name]["f1-score"]
        per_class_results.append(
            {
                "intent": intent_name,
                "support": support,
                "word_f1": word_f1,
                "char_f1": char_f1,
                "delta": char_f1 - word_f1,
            }
        )

largest_gains = sorted(
    per_class_results,
    key=lambda result: result["delta"],
    reverse=True,
)[:10]
largest_losses = sorted(
    per_class_results,
    key=lambda result: result["delta"],
)[:10]

print("\n10 intents with largest positive F1 deltas")
print(f"{'Intent':<35} {'Support':>7} {'Word F1':>9} {'Char F1':>9} {'Delta':>9}")
print("-" * 73)
for result in largest_gains:
    print(
        f"{result['intent']:<35} "
        f"{result['support']:>7} "
        f"{result['word_f1']:>9.4f} "
        f"{result['char_f1']:>9.4f} "
        f"{result['delta']:>+9.4f}"
    )

print("\n10 intents with largest negative F1 deltas")
print(f"{'Intent':<35} {'Support':>7} {'Word F1':>9} {'Char F1':>9} {'Delta':>9}")
print("-" * 73)
for result in largest_losses:
    print(
        f"{result['intent']:<35} "
        f"{result['support']:>7} "
        f"{result['word_f1']:>9.4f} "
        f"{result['char_f1']:>9.4f} "
        f"{result['delta']:>+9.4f}"
    )

general_quirky = next(
    result for result in per_class_results if result["intent"] == "general_quirky"
)
print("\ngeneral_quirky")
print(
    f"Support: {general_quirky['support']}, "
    f"Word F1: {general_quirky['word_f1']:.4f}, "
    f"Character F1: {general_quirky['char_f1']:.4f}, "
    f"Delta: {general_quirky['delta']:+.4f}"
)

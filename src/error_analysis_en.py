from collections import Counter

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


data_files = {
    "train": "hf://datasets/AmazonScience/massive@~parquet/en-US/train/0000.parquet",
    "validation": "hf://datasets/AmazonScience/massive@~parquet/en-US/validation/0000.parquet",
}

dataset = load_dataset("parquet", data_files=data_files)

X_train = dataset["train"]["utt"]
y_train = dataset["train"]["intent"]
X_val = dataset["validation"]["utt"]
y_val = dataset["validation"]["intent"]

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_val_pred = model.predict(X_val_tfidf)

accuracy = accuracy_score(y_val, y_val_pred)
macro_f1 = f1_score(y_val, y_val_pred, average="macro")

print("Validation results")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")

intent_names = dataset["train"].features["intent"].names
errors = [
    {
        "utterance": utterance,
        "true_intent": intent_names[true_label],
        "predicted_intent": intent_names[predicted_label],
    }
    for utterance, true_label, predicted_label in zip(
        X_val,
        y_val,
        y_val_pred,
    )
    if true_label != predicted_label
]

confusion_counts = Counter(
    (error["true_intent"], error["predicted_intent"])
    for error in errors
)
top_confusions = confusion_counts.most_common(15)

print("\nTop 15 confusion pairs")
for (true_intent, predicted_intent), count in top_confusions:
    print(f"{true_intent} -> {predicted_intent} -> {count}")

print("\nExamples for the top 5 confusion pairs")
for (true_intent, predicted_intent), _ in top_confusions[:5]:
    examples = [
        error["utterance"]
        for error in errors
        if error["true_intent"] == true_intent
        and error["predicted_intent"] == predicted_intent
    ][:3]

    print(f"\nTrue intent: {true_intent}")
    print(f"Predicted intent: {predicted_intent}")
    print("Examples:")
    for utterance in examples:
        print(f"- {utterance}")

general_quirky_id = intent_names.index("general_quirky")
general_quirky_predictions = [
    predicted_label
    for true_label, predicted_label in zip(y_val, y_val_pred)
    if true_label == general_quirky_id
]
general_quirky_errors = [
    error for error in errors if error["true_intent"] == "general_quirky"
]

support = len(general_quirky_predictions)
correct = sum(
    predicted_label == general_quirky_id
    for predicted_label in general_quirky_predictions
)
incorrect = support - correct
misclassified_as = Counter(
    error["predicted_intent"] for error in general_quirky_errors
)

print("\ngeneral_quirky analysis")
print(f"Support: {support}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {incorrect}")

print("\nTop predicted intents when general_quirky is misclassified")
for predicted_intent, count in misclassified_as.most_common(10):
    print(f"{predicted_intent} -> {count}")

print("\nExample general_quirky errors")
for error in general_quirky_errors[:5]:
    print(
        f"- {error['utterance']} "
        f"-> {error['predicted_intent']}"
    )

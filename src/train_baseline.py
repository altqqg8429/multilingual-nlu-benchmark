import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


data_files = {
    "train": "hf://datasets/AmazonScience/massive@~parquet/en-US/train/0000.parquet",
    "validation": "hf://datasets/AmazonScience/massive@~parquet/en-US/validation/0000.parquet",
    "test": "hf://datasets/AmazonScience/massive@~parquet/en-US/test/0000.parquet",
}

dataset = load_dataset("parquet", data_files=data_files)


# Prepare training and validation data
X_train = dataset["train"]["utt"]
y_train = dataset["train"]["intent"]

X_val = dataset["validation"]["utt"]
y_val = dataset["validation"]["intent"]


# Convert text into TF-IDF features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

print("Number of training examples:", len(X_train))
print("TF-IDF matrix shape:", X_train_tfidf.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))


# Train the baseline classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)


# Predict validation labels
y_val_pred = model.predict(X_val_tfidf)

print("\nNumber of validation examples:", len(y_val))
print("Number of predictions:", len(y_val_pred))


# Evaluate overall performance
accuracy = accuracy_score(y_val, y_val_pred)

macro_f1 = f1_score(
    y_val,
    y_val_pred,
    average="macro",
)

print("\nValidation results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {macro_f1:.4f}")


# Prepare human-readable intent names
intent_names = dataset["train"].features["intent"].names

all_labels = set(range(len(intent_names)))
validation_labels = set(y_val)

missing_validation_labels = sorted(all_labels - validation_labels)

print("\nIntents absent from the validation set:")
for label_id in missing_validation_labels:
    print(f"{label_id}: {intent_names[label_id]}")


# Evaluate performance for each intent
present_labels = sorted(set(y_val) | set(y_val_pred))
present_intent_names = [intent_names[label_id] for label_id in present_labels]

report = classification_report(
    y_val,
    y_val_pred,
    labels=present_labels,
    target_names=present_intent_names,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()

class_report = report_df.loc[present_intent_names]

# Only rank classes that actually have validation examples
class_report = class_report[class_report["support"] > 0]

worst_intents = class_report.sort_values("f1-score").head(10)

print("\n10 intents with the lowest F1:")
print(
    worst_intents[
        ["precision", "recall", "f1-score", "support"]
    ].round(3)
)


val_results = pd.DataFrame({
    "utterance": X_val,
    "true_label": y_val,
    "predicted_label": y_val_pred,
})

val_results["true_intent"] = val_results["true_label"].map(
    lambda label_id: intent_names[label_id]
)

val_results["predicted_intent"] = val_results["predicted_label"].map(
    lambda label_id: intent_names[label_id]
)

errors = val_results[
    val_results["true_label"] != val_results["predicted_label"]
]

quirky_errors = errors[
    errors["true_intent"] == "general_quirky"
]

print("\nMisclassified general_quirky examples:")
print(
    quirky_errors[
        ["utterance", "true_intent", "predicted_intent"]
    ].head(15).to_string(index=False)
)

quirky_false_positives = errors[
    errors["predicted_intent"] == "general_quirky"
]

print("\nExamples incorrectly predicted as general_quirky:")
print(
    quirky_false_positives[
        ["utterance", "true_intent", "predicted_intent"]
    ].head(15).to_string(index=False)
)

print("\nTrue intents most often confused with general_quirky:")
print(
    quirky_false_positives["true_intent"]
    .value_counts()
    .head(10)
)


quirky_true_count = (
    val_results["true_intent"] == "general_quirky"
).sum()

quirky_predicted_count = (
    val_results["predicted_intent"] == "general_quirky"
).sum()

print("\nGeneral_quirky counts:")
print("True examples:", quirky_true_count)
print("Predicted examples:", quirky_predicted_count)
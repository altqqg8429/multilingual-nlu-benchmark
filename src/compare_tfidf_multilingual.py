from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


languages = [
    ("English", "en-US"),
    ("German", "de-DE"),
    ("Russian", "ru-RU"),
]


def evaluate_representation(vectorizer, X_train, y_train, X_val, y_val):
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    y_val_pred = model.predict(X_val_tfidf)

    return {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "macro_f1": f1_score(y_val, y_val_pred, average="macro"),
    }


results = []

print("Dataset sanity checks")
for language, locale in languages:
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

    X_train = dataset["train"]["utt"]
    y_train = dataset["train"]["intent"]
    X_val = dataset["validation"]["utt"]
    y_val = dataset["validation"]["intent"]

    print(
        f"{language} ({locale}): "
        f"train={len(X_train)}, validation={len(X_val)}"
    )

    representations = [
        ("Word", TfidfVectorizer()),
        (
            "Character",
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)),
        ),
    ]

    for representation, vectorizer in representations:
        metrics = evaluate_representation(
            vectorizer,
            X_train,
            y_train,
            X_val,
            y_val,
        )
        results.append(
            {
                "language": language,
                "representation": representation,
                **metrics,
            }
        )

print("\nValidation results")
print(f"{'Language':<10} {'Representation':<15} {'Accuracy':>10} {'Macro-F1':>10}")
print("-" * 49)
for result in results:
    print(
        f"{result['language']:<10} "
        f"{result['representation']:<15} "
        f"{result['accuracy']:>10.4f} "
        f"{result['macro_f1']:>10.4f}"
    )

print("\nCharacter - Word")
for language, _ in languages:
    language_results = [
        result for result in results if result["language"] == language
    ]
    word_result, char_result = language_results
    accuracy_delta = char_result["accuracy"] - word_result["accuracy"]
    macro_f1_delta = char_result["macro_f1"] - word_result["macro_f1"]

    print(
        f"{language}: "
        f"Accuracy {accuracy_delta:+.4f}, "
        f"Macro-F1 {macro_f1_delta:+.4f}"
    )

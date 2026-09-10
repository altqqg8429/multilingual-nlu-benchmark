import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"
with (results_dir / "tfidf_baseline_results.csv").open(
    newline="", encoding="utf-8"
) as file:
    tfidf_results = {
        (row["language"], row["representation"]): float(row["macro_f1"])
        for row in csv.DictReader(file)
    }
with (results_dir / "xlmr_finetuning_results.csv").open(
    newline="", encoding="utf-8"
) as file:
    xlmr_results = {
        row["language"]: float(row["macro_f1"])
        for row in csv.DictReader(file)
    }

languages = ["English", "German", "Russian"]
series = [
    ("Word TF-IDF", [tfidf_results[(language, "Word")] for language in languages]),
    (
        "Character TF-IDF",
        [tfidf_results[(language, "Character")] for language in languages],
    ),
    ("XLM-R", [xlmr_results[language] for language in languages]),
]

fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
bar_width = 0.24
colors = ["#4477AA", "#EEAA33", "#228833"]
for index, (label, scores) in enumerate(series):
    positions = [position + (index - 1) * bar_width for position in range(3)]
    bars = ax.bar(
        positions, scores, width=bar_width, label=label, color=colors[index],
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

ax.set_xticks(range(len(languages)), languages)
ax.set_ylabel("Macro-F1")
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_title("Macro-F1 by Model and Language", pad=12)
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper center", ncol=3, frameon=False)

figures_dir = project_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
output_path = figures_dir / "overall_performance.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {output_path}")

with (results_dir / "xlmr_cross_lingual_results.csv").open(
    newline="", encoding="utf-8"
) as file:
    cross_lingual_results = {
        row["evaluation_language"]: float(row["macro_f1"])
        for row in csv.DictReader(file)
        if row["train_language"] == "English"
    }

languages = ["German", "Russian"]
series = [
    ("Target-language fine-tuning", [xlmr_results[language] for language in languages]),
    (
        "English-only zero-shot",
        [cross_lingual_results[language] for language in languages],
    ),
]

fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
for index, (label, scores) in enumerate(series):
    positions = [
        position + (index - 0.5) * bar_width
        for position in range(len(languages))
    ]
    bars = ax.bar(
        positions, scores, width=bar_width, label=label, color=colors[index],
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

ax.set_xticks(range(len(languages)), languages)
ax.set_ylabel("Macro-F1")
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_title("Target-Language Fine-Tuning vs Zero-Shot Transfer", pad=12)
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper center", ncol=2, frameon=False)

output_path = figures_dir / "cross_lingual_transfer.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {output_path}")

with (results_dir / "xlmr_predictions_English.csv").open(
    newline="", encoding="utf-8"
) as file:
    confusion_counts = Counter(
        (row["true_intent"], row["predicted_intent"])
        for row in csv.DictReader(file)
        if int(row["true_id"]) != int(row["predicted_id"])
    )
top_confusions = confusion_counts.most_common(12)
labels = [
    f"{true_intent} → {predicted_intent}"
    for (true_intent, predicted_intent), count in top_confusions
]
counts = [count for pair, count in top_confusions]

fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
bars = ax.barh(labels, counts, color=colors[0])
ax.invert_yaxis()
ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
ax.set_xlabel("Count")
ax.set_title("Most Frequent English XLM-R Confusions", pad=12)
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.set_xlim(left=0)
ax.margins(x=0.12)
ax.set_axisbelow(True)
ax.grid(axis="x", alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)

output_path = figures_dir / "xlmr_top_confusions_en.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {output_path}")

with (results_dir / "xlmr_per_class_f1_English.csv").open(
    newline="", encoding="utf-8"
) as file:
    per_class_results = [
        {
            "intent": row["intent"],
            "support": int(row["support"]),
            "tfidf_f1": float(row["tfidf_f1"]),
            "xlmr_f1": float(row["xlmr_f1"]),
            "delta": float(row["delta"]),
        }
        for row in csv.DictReader(file)
    ]
largest_gains = sorted(
    [result for result in per_class_results if result["delta"] > 0],
    key=lambda result: result["delta"], reverse=True,
)[:5]
largest_losses = sorted(
    [result for result in per_class_results if result["delta"] < 0],
    key=lambda result: result["delta"],
)[:5]
selected_classes = sorted(
    largest_gains + largest_losses, key=lambda result: result["delta"],
)

fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
deltas = [result["delta"] for result in selected_classes]
bars = ax.barh(
    [result["intent"] for result in selected_classes], deltas,
    color=[colors[2] if delta > 0 else colors[0] for delta in deltas],
)
ax.bar_label(bars, fmt="%+.3f", padding=3, fontsize=9)
ax.axvline(0, color="#444444", linewidth=1)
ax.set_xlabel("F1 change (XLM-R - TF-IDF)")
ax.set_title("Largest Per-Class F1 Changes: TF-IDF → XLM-R", pad=12)
ax.margins(x=0.18)
ax.set_axisbelow(True)
ax.grid(axis="x", alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)

output_path = figures_dir / "per_class_f1_change_en.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {output_path}")

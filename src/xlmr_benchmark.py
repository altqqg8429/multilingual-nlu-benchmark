from statistics import median

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "FacebookAI/xlm-roberta-base"
MAX_LENGTH = 64
BATCH_SIZE = 4


print("English XLM-R pipeline smoke test ")

data_files = {
    "train": (
        "hf://datasets/AmazonScience/massive@~parquet/"
        "en-US/train/0000.parquet"
    ),
    "validation": (
        "hf://datasets/AmazonScience/massive@~parquet/"
        "en-US/validation/0000.parquet"
    ),
}

dataset = load_dataset("parquet", data_files=data_files)

# Keep the same MASSIVE fields used in the classical experiments.
# Convert Dataset columns to plain Python lists because the tokenizer
# expects list[str] rather than a Hugging Face Dataset Column object.
X_train = list(dataset["train"]["utt"])
y_train = list(dataset["train"]["intent"])
X_val = list(dataset["validation"]["utt"])
y_val = list(dataset["validation"]["intent"])

print("\nDataset sanity checks")
print(f"English (en-US): train={len(X_train)}, validation={len(X_val)}")

if len(X_train) != 11514 or len(X_val) != 2033:
    raise ValueError("Unexpected English MASSIVE split sizes")


# MASSIVE already stores intent labels as integer ClassLabel IDs.
# Reuse that original mapping instead of creating a new label encoding.
intent_feature = dataset["train"].features["intent"]
validation_intent_feature = dataset["validation"].features["intent"]

if intent_feature.num_classes != 60:
    raise ValueError(
        f"Expected 60 intents, got {intent_feature.num_classes}"
    )

if validation_intent_feature.names != intent_feature.names:
    raise ValueError("Train and validation intent schemas do not match")

unknown_validation_labels = set(y_val) - set(y_train)
if unknown_validation_labels:
    raise ValueError(
        f"Validation contains unknown intent IDs: "
        f"{sorted(unknown_validation_labels)}"
    )

id2label = {
    intent_id: intent_feature.int2str(intent_id)
    for intent_id in range(intent_feature.num_classes)
}

label2id = {
    intent_name: intent_id
    for intent_id, intent_name in id2label.items()
}

print("Input: utt")
print("Target: intent")
print(f"Intent classes: {intent_feature.num_classes}")
print("Validation intent schema matches training schema")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Check whether max_length=64 is large enough using training data only.
full_encoding = tokenizer(
    X_train,
    truncation=False,
    padding=False,
    return_attention_mask=False,
)

lengths = [len(input_ids) for input_ids in full_encoding["input_ids"]]
over_limit = sum(length > MAX_LENGTH for length in lengths)

print("\nTrain-only token lengths (including special tokens)")
print(f"Maximum: {max(lengths)}")
print(f"Median: {median(lengths)}")
print(
    f"Exceeding {MAX_LENGTH}: "
    f"{over_limit}/{len(lengths)} "
    f"({over_limit / len(lengths):.2%})"
)


utterances = X_train[:BATCH_SIZE]

inputs = tokenizer(
    utterances,
    max_length=MAX_LENGTH,
    truncation=True,
    padding=True,
    return_tensors="pt",
)

print("\nTokenizer examples")

for index, utterance in enumerate(utterances[:2]):
    input_ids = inputs["input_ids"][index].tolist()
    intent_id = y_train[index]

    print(f"\nRaw utterance: {utterance}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids)}")
    print(f"Token IDs: {input_ids}")
    print(
        f"Attention mask: "
        f"{inputs['attention_mask'][index].tolist()}"
    )
    print(
        f"Intent: {id2label[intent_id]} "
        f"-> model label ID {intent_id}"
    )


print("\nGPU check")
print(f"torch version: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this checkpoint; GPU smoke test failed"
    )

device = torch.device("cuda:0")
print(f"CUDA device name: {torch.cuda.get_device_name(device)}")


# XLM-R is pretrained, but the 60-class classification head is new.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=intent_feature.num_classes,
    label2id=label2id,
    id2label=id2label,
)

model = model.to(device)
model.eval()

inputs = {
    name: tensor.to(device)
    for name, tensor in inputs.items()
}

labels = torch.tensor(
    y_train[:BATCH_SIZE],
    dtype=torch.long,
    device=device,
)

if model.config.num_labels != 60:
    raise RuntimeError("Model must have exactly 60 output labels")

print("\nForward-pass check")
print(
    f"Model: {MODEL_NAME}; "
    f"labels={model.config.num_labels}; "
    f"device={model.device}"
)
print(f"Batch size: {BATCH_SIZE}")
print(f"model.training: {model.training}")

for name, tensor in {**inputs, "labels": labels}.items():
    print(
        f"{name}: "
        f"shape={list(tensor.shape)}, "
        f"device={tensor.device}"
    )


with torch.inference_mode():
    outputs = model(**inputs, labels=labels)


if outputs.logits.shape != (BATCH_SIZE, 60):
    raise RuntimeError(
        f"Unexpected logits shape: {outputs.logits.shape}"
    )

if outputs.loss is None or not torch.isfinite(outputs.loss).item():
    raise RuntimeError("Expected a finite loss")

if not torch.isfinite(outputs.logits).all().item():
    raise RuntimeError("Non-finite logits")


print(
    f"Logits: shape={list(outputs.logits.shape)}, "
    f"device={outputs.logits.device}"
)
print(
    f"Loss: {outputs.loss.item():.6f}; "
    f"finite=True; "
    f"device={outputs.loss.device}"
)

print(
    "\nPASS: CUDA forward pass only. "
    "Untrained-head loss has no benchmark meaning."
)
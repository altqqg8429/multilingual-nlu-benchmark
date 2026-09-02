from datasets import load_dataset

data_files = {
    "train": "hf://datasets/AmazonScience/massive@~parquet/en-US/train/0000.parquet",
    "validation": "hf://datasets/AmazonScience/massive@~parquet/en-US/validation/0000.parquet",
    "test": "hf://datasets/AmazonScience/massive@~parquet/en-US/test/0000.parquet",
}

dataset = load_dataset("parquet", data_files=data_files)

print(dataset)


train_df = dataset["train"].to_pandas()

print("\nTraining set shape:")
print(train_df.shape)

print("\nColumns:")
print(train_df.columns.tolist())

print("\nFirst five examples:")
print(train_df[["utt", "intent", "scenario"]].head())


print("\nBasic checks:")
print("Rows:", len(train_df))
print("Locales:", train_df["locale"].nunique())
print("Scenarios:", train_df["scenario"].nunique())
print("Intents:", train_df["intent"].nunique())

print("\nMissing values in key columns:")
print(train_df[["utt", "intent", "scenario"]].isna().sum())

print("\nDuplicate utterances:")
print(train_df["utt"].duplicated().sum())


intent_counts = train_df["intent"].value_counts()

print("\nIntent frequency summary:")
print(intent_counts.describe())

print("\nMost common intent IDs:")
print(intent_counts.head(10))

intent_feature = dataset["train"].features["intent"]
scenario_feature = dataset["train"].features["scenario"]

print("\nIntent feature:")
print(intent_feature)

print("\nScenario feature:")
print(scenario_feature)


train_df["intent_name"] = train_df["intent"].map(intent_feature.int2str)
train_df["scenario_name"] = train_df["scenario"].map(scenario_feature.int2str)

print("\nExamples with readable labels:")
print(
    train_df[
        ["utt", "scenario_name", "intent_name"]
    ].head()
)


intent_counts = train_df["intent_name"].value_counts()

print("\nMost common intents:")
print(intent_counts.head(10))

print("\nLeast common intents:")
print(intent_counts.tail(10))
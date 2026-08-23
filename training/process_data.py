import argparse
import pandas as pd
import json
import os

parser = argparse.ArgumentParser(
    description="Rename NLI CSV columns (sent0/sent1/hard_neg) to anchor/positive/negative "
                "and emit the triplet JSON consumed by train_finetuning.py."
)
parser.add_argument("--input_file", required=True, help="Source CSV, e.g. nli_for_simcse.csv")
parser.add_argument("--output_path", default="./data/all_nli_triplets.json")
args = parser.parse_args()

input_file = args.input_file

print(f"Loading {input_file}...")
try:
    df = pd.read_csv(input_file)
except FileNotFoundError:
    # If you are debugging and just have the dataframe in memory, ignore this
    print("File not found. Please ensure the path is correct.")
    exit()

# 2. Rename columns to match the training script's expectations
# sent0 -> anchor
# sent1 -> positive
# hard_neg -> negative
print("Renaming columns...")
df = df.rename(columns={
    "sent0": "anchor",
    "sent1": "positive",
    "hard_neg": "negative"
})

# 3. Data Cleaning (Crucial for stability)
# Drop rows where any of the text fields are missing (NaN)
initial_len = len(df)
df = df.dropna(subset=["anchor", "positive", "negative"])
cleaned_len = len(df)
if initial_len != cleaned_len:
    print(f"Dropped {initial_len - cleaned_len} rows containing empty/NaN values.")

# Ensure all data is string type (sometimes CSVs read numbers as int)
df["anchor"] = df["anchor"].astype(str)
df["positive"] = df["positive"].astype(str)
df["negative"] = df["negative"].astype(str)

# 4. Convert to list of dictionaries
print("Converting to JSON format...")
json_data = df.to_dict(orient="records")

# 5. Save to JSON
output_path = args.output_path
if os.path.dirname(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f"Saving {len(json_data)} samples to {output_path}...")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print("Done! You can now run the training script.")
import json
import csv
import random

# === FILES ===
INPUT_JSON = "aggregated_durations.json"
CSV_FILE = "ai_benchmark_data.csv"
OUTPUT_JSON = "final_dataset_with_full_model_info.json"

# === Step 1: Load JSON data
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

# === Step 2: Load full rows from CSV
csv_rows = []
with open(CSV_FILE, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Clean up whitespace
        clean_row = {key.strip(): value.strip() for key, value in row.items()}
        csv_rows.append(clean_row)

# === Step 3: Assign a random row to each trace
for entry in data:
    entry["model"] = random.choice(csv_rows)

# === Step 4: Save to output JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved {len(data)} enriched entries to {OUTPUT_JSON}")

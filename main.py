import csv, json
import pandas as pd

data = pd.read_csv("data (1).csv")

columns = data.columns
with open("data (1).csv") as f, open("faq_alpaca.jsonl", "w") as out:
    reader = csv.DictReader(f)
    for row in reader:
        entry = {
            "instruction": row[columns[0]],
            "input": "",
            "output": row[columns[1]],
        }
        out.write(json.dumps(entry) + "\n")

print("Done! Check faq_alpaca.jsonl")
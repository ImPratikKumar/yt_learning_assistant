import pickle
import json
from src.evaluation_dataset import build_dataset


with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Number of chunks: {len(chunks)}")

dataset = build_dataset(chunks)

## Save evaluation dataset
with open("eval_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)
import os
import json
from config import DATA_PATH, DOMAINS

def load_dataset():
    dataset = {}
    for domain in DOMAINS:
        path = os.path.join(DATA_PATH, domain)
        dataset[domain] = []
        if os.path.exists(path):
            for f in os.listdir(path):
                if f.endswith(".json"):
                    with open(path, "r", encoding="utf-8") as fin:
                        dataset[domain].append(json.load(fin))
    return dataset

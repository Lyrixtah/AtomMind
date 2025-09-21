"""Load scientific datasets for each domain."""

import os
import json
from config import DATA_PATH, DOMAINS

def load_dataset():
    """
    Loads JSON datasets from each domain folder specified in DOMAINS.

    Returns:
        dict: A dictionary mapping domain names to lists of JSON objects.
    """
    dataset = {}

    for domain in DOMAINS:
        domain_path = os.path.join(DATA_PATH, domain)
        dataset[domain] = []

        if os.path.exists(domain_path):
            for filename in os.listdir(domain_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(domain_path, filename)
                    with open(file_path, "r", encoding="utf-8") as fin:
                        dataset[domain].append(json.load(fin))

    return dataset

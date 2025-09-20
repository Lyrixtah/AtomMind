import os
import requests
from dotenv import load_dotenv

load_dotenv()

class ExecutorAgent:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate_candidates(self, prompt, num_candidates=3):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        messages = [{"role": "system", "content": "You are a scientific assistant."},
                    {"role": "user", "content": prompt}]
        params = {
            "model": "openai/gpt-4",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 150
        }
        response = requests.post(self.base_url, json=params, headers=headers, timeout=10)
        response.raise_for_status()
        choices = response.json().get("choices", [])
        candidates = [choice["message"]["content"] for choice in choices[:num_candidates]]
        return candidates

"""
Provides tokenization utilities for the SmallScientificLLM model.
Initializes a GPT-2 tokenizer and adds domain-specific special tokens.
"""

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
special_tokens = ["<eq>", "<mol>", "<bio>", "<chem>"]
tokenizer.add_tokens(special_tokens)

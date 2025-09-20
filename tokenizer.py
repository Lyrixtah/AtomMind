from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
special_tokens = ["<eq>", "<mol>", "<bio>", "<chem>"]
tokenizer.add_tokens(special_tokens)

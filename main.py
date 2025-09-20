import torch
from config import DEVICE, EPOCHS, MAX_SEQ_LEN, HIDDEN_SIZE, DOMAINS, SAVE_PATH
from tokenizer import tokenizer
from models.slm import SmallScientificLLM
from memory.memory import MemorySystem
from agents.executor import ExecutorAgent
from agents.critic import CriticAgent
from agents.consensus import consensus
from agents.trainer import TrainerAgent
from utils.logger import Logger
from torch.nn import Embedding
import os

# Initialize modules
logger = Logger()
slm = SmallScientificLLM().to(DEVICE)
memory = MemorySystem()
executor = ExecutorAgent()
critic = CriticAgent()
trainer = TrainerAgent()
embedding_layer = Embedding(tokenizer.vocab_size, HIDDEN_SIZE).to(DEVICE)

optimizer = torch.optim.AdamW(slm.parameters(), lr=1e-4)

# Toggle chat training
train_chat = True

# Ensure save path exists
os.makedirs(SAVE_PATH, exist_ok=True)

for epoch in range(EPOCHS):
    logger.log(f"=== Epoch {epoch + 1} ===")

    # ---- Automatic Science Tasks ----
    try:
        auto_tasks = executor.generate_candidates(
            prompt="Generate a scientific task/problem in math, physics, chemistry, or biology.", 
            num_candidates=5
        )
    except Exception as e:
        logger.log(f"Executor error: {e}")
        continue

    # Optional: Generate chat examples
    if train_chat:
        try:
            chat_examples = executor.generate_candidates(
                prompt="Generate casual daily conversation examples between two people talking about hobbies, greetings, or daily life.",
                num_candidates=5
            )
        except Exception as e:
            logger.log(f"Executor chat generation error: {e}")
            chat_examples = []

    for i, task_prompt in enumerate(auto_tasks):
        # ---- Candidate generation & evaluation ----
        try:
            candidates = executor.generate_candidates(task_prompt)
        except Exception as e:
            logger.log(f"Executor error on task: {e}")
            continue

        scores = critic.evaluate(candidates)
        top_candidate = consensus(candidates, scores)

        logger.log(task_prompt)
        logger.log(f"Top Candidate - {top_candidate}")
        memory.store(top_candidate, max(scores))

        # ---- Tokenize science output ----
        tokens = tokenizer.encode(top_candidate, return_tensors="pt").to(DEVICE)
        if tokens.size(1) > MAX_SEQ_LEN:
            tokens = tokens[:, :MAX_SEQ_LEN]
        elif tokens.size(1) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - tokens.size(1)
            tokens = torch.cat([tokens, torch.zeros(1, pad_len, dtype=torch.long).to(DEVICE)], dim=1)

        input_tensor = embedding_layer(tokens)
        x_dict = {domain: input_tensor for domain in DOMAINS}

        # ---- Tokenize chat output ----
        chat_tensor = None
        if train_chat and chat_examples:
            chat_text = chat_examples[i % len(chat_examples)]
            chat_tokens = tokenizer.encode(chat_text, return_tensors="pt").to(DEVICE)
            if chat_tokens.size(1) > MAX_SEQ_LEN:
                chat_tokens = chat_tokens[:, :MAX_SEQ_LEN]
            elif chat_tokens.size(1) < MAX_SEQ_LEN:
                pad_len = MAX_SEQ_LEN - chat_tokens.size(1)
                chat_tokens = torch.cat([chat_tokens, torch.zeros(1, pad_len, dtype=torch.long).to(DEVICE)], dim=1)
            chat_tensor = embedding_layer(chat_tokens)

        # ---- Forward pass ----
        output = slm(x_dict, chat_tensor)

        # ---- Loss computation ----
        loss = ((output[:, :input_tensor.size(1), :] - input_tensor) ** 2).mean()

        # ---- Backprop ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log(f"Loss: {loss.item():.4f}\n")

    # ---- Save pretrain model after each epoch ----
    save_file = os.path.join(SAVE_PATH, f"slm_pretrain_epoch{epoch+1}.pt")
    torch.save(slm.state_dict(), save_file)
    logger.log(f"Saved pretrained SLM model to: {save_file}\n")

if train_chat:
    logger.log("Automatic Science + Chat training completed!")
else:
    logger.log("Automatic Science training completed!")

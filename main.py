import os

import torch
from torch.nn import Embedding

from config import DEVICE, EPOCHS, MAX_SEQ_LEN, HIDDEN_SIZE, DOMAINS, SAVE_PATH
from tokenizer import tokenizer
from models.slm import SmallScientificLLM
from memory.memory import MemorySystem
from agents.executor import ExecutorAgent
from agents.critic import CriticAgent
from agents.consensus import consensus
from agents.trainer import TrainerAgent
from utils.logger import Logger


# Initialize modules
logger = Logger()
slm = SmallScientificLLM().to(DEVICE)
memory = MemorySystem()
executor = ExecutorAgent()
critic = CriticAgent()
trainer = TrainerAgent()
embedding_layer = Embedding(tokenizer.vocab_size, HIDDEN_SIZE).to(DEVICE)

optimizer = torch.optim.AdamW(slm.parameters(), lr=1e-4)

# Features (Customizable)
train_chat = True # Set False if just need to gather knowledge
stable_start = False # Set True if model training is complete and starting to fine-tuning
warmup_epochs = 3
max_grad_norm = 1.0

for epoch in range(EPOCHS):
    logger.log(f"\n=== Epoch {epoch + 1} ===")

    if stable_start and epoch < warmup_epochs:
        lr = 1e-5
    else:
        lr = 1e-4
    for g in optimizer.param_groups:
        g['lr'] = lr

    # Automatic Science Tasks
    try:
        auto_tasks = executor.generate_candidates(
            prompt="Generate a scientific task/problem in math, physics, chemistry, or biology.",
            num_candidates=5
        )
    except Exception as e:
        logger.log(f"Executor error: {e}")
        continue

    # Optional chat examples
    chat_examples = []
    if train_chat:
        try:
            chat_examples = executor.generate_candidates(
                prompt="Generate casual daily conversation examples between two people talking about hobbies, greetings, or daily life.",
                num_candidates=5
            )
        except Exception as e:
            logger.log(f"Executor chat generation error: {e}")

    for i, task_prompt in enumerate(auto_tasks):
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

        # Tokenize science output
        tokens = tokenizer.encode(top_candidate, return_tensors="pt").to(DEVICE)
        if tokens.size(1) > MAX_SEQ_LEN:
            tokens = tokens[:, :MAX_SEQ_LEN]
        elif tokens.size(1) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - tokens.size(1)
            tokens = torch.cat([tokens, torch.zeros(1, pad_len, dtype=torch.long).to(DEVICE)], dim=1)
        input_tensor = embedding_layer(tokens)
        x_dict = {domain: input_tensor for domain in DOMAINS}

        # Tokenize chat output
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

        output = slm(x_dict, chat_tensor)
        loss_mse = ((output[:, :input_tensor.size(1), :] - input_tensor) ** 2).mean()

        if stable_start and epoch < warmup_epochs:
            loss = loss_mse
        else:
            metrics = critic.evaluate_metrics(top_candidate)  # get contradictions, hallucinations, reasoning_quality
            contradiction_penalty = metrics["contradictions"] * 2.0
            hallucination_penalty = metrics["hallucinations"] * 2.0
            reasoning_bonus = (1.0 - metrics["reasoning_quality"]) * 1.5
            loss = loss_mse + contradiction_penalty + hallucination_penalty + reasoning_bonus

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if stable_start:
            torch.nn.utils.clip_grad_norm_(slm.parameters(), max_grad_norm)

        optimizer.step()
        logger.log(f"Loss: {loss.item():.4f}\n")

    # Save pretrain model after each epoch
    save_file = os.path.join(SAVE_PATH, f"slm_pretrain_epoch{epoch+1}.pt")
    torch.save(slm.state_dict(), save_file)
    logger.log(f"Saved pretrained SLM model to: {save_file}\n")

if train_chat:
    logger.log("Automatic Science + Chat training completed!")
else:
    logger.log("Automatic Science training completed!")

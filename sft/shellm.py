import verifiers as vf
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

load_dotenv()

"""
accelerate launch --config-file sft/zero3.yaml sft/shellm.py
"""

# convenience function for FA2 initialization
model, tokenizer = vf.get_model_and_tokenizer("Qwen/Qwen3-8B", use_liger=False)
dataset = load_dataset('deathbyknowledge/V3-shell-format', split='train')
dataset = dataset.select(range(200)) # type: ignore

tok_counts = []
for row in dataset:
    # count tokens in (prompt, completion)
    messages = row['prompt'] + row['completion'] # type: ignore
    toks = tokenizer.apply_chat_template( 
        messages,
        tokenize=True
    )
    tok_counts.append(len(toks))

# tok count stats
print(f"Dataset size: {len(tok_counts)}")
print(f"Min tokens: {min(tok_counts)}")
print(f"Max tokens: {max(tok_counts)}")
print(f"Mean tokens: {sum(tok_counts) / len(tok_counts)}")
print(f"Median tokens: {sorted(tok_counts)[len(tok_counts) // 2]}")

args = SFTConfig(
    max_length=8192,
    output_dir="qwen3-8b-shell-sft",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    bf16=True,
    learning_rate=2e-5,
    num_train_epochs=2,
    weight_decay=0.01,
    max_grad_norm=1.0,
    report_to="wandb",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=1,
    save_only_model=True,
    log_on_each_node=True,
    push_to_hub=True,
    hub_model_id="Qwen3-8B-Shell-SFT",
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset # type: ignore
)
trainer.train()

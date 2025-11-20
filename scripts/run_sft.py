import argparse
import os
from typing import Dict

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer


def build_instruction(example: Dict[str, str], prompt_field: str, target_field: str) -> str:
    """Simple math/logic prompt template so we can keep SFT lightweight."""
    prompt = example[prompt_field].strip()
    target = example[target_field].strip()
    return f"Question:\n{prompt}\n\nAnswer:\n{target}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT bootstrap for Gemma 3 12B")
    parser.add_argument("--model-name", default="google/gemma-3-12b-it", help="HF model id")
    parser.add_argument("--dataset", default="gsm8k", help="HF dataset id (math/logic friendly)")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--prompt-field", default="question", help="Field that holds the question/puzzle")
    parser.add_argument("--target-field", default="answer", help="Field that holds the reference answer")
    parser.add_argument("--max-samples", type=int, default=2000, help="Optional cap to keep runs cheap")
    parser.add_argument("--output-dir", default="checkpoints/sft-gemma3-12b", help="Where to save LoRA adapters")
    parser.add_argument("--per-device-batch", type=int, default=1, help="Per-GPU batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Grad accumulation steps (keeps VRAM low)")
    parser.add_argument("--max-steps", type=int, default=500, help="Training steps (small by default)")
    parser.add_argument("--save-steps", type=int, default=50, help="Checkpoint every N optimizer steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="LoRA learning rate")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--resume-from", default=None, help="Path to resume_from_checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    raw = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        raw = raw.select(range(min(args.max_samples, len(raw))))

    def tokenize(batch):
        prompts = batch[args.prompt_field]
        targets = batch[args.target_field]
        texts = [
            build_instruction({args.prompt_field: p, args.target_field: t}, args.prompt_field, args.target_field)
            for p, t in zip(prompts, targets)
        ]
        return tokenizer(texts, truncation=True, max_length=4096, padding="longest")

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
    tokenized.set_format(type="torch")

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized,
        peft_config=lora_cfg,
        data_collator=collator,
        args=training_args,
        packing=False,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

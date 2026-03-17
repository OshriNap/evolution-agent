"""Train a QLoRA adapter for the evolution mutator model.

Fine-tunes qwen2.5-coder:7b on successful mutations from evolution runs.
The LoRA learns general mutation skills (format compliance, guidance following,
producing valid improvements) NOT problem-specific knowledge.

Uses unsloth for 4-bit QLoRA training on consumer GPUs (8GB VRAM).

Usage:
    # First extract training data:
    python scripts/extract_training_data.py --format sft --min-reward 0.1 --output training_data_sft.jsonl

    # Then train:
    python scripts/train_lora.py --data training_data_sft.jsonl --output lora_mutator

    # Merge and export to Ollama:
    python scripts/train_lora.py --merge --output lora_mutator
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_training_data(data_path: Path) -> list[dict]:
    """Load SFT training data from JSONL."""
    examples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def format_for_training(examples: list[dict]) -> list[dict]:
    """Convert to the chat format unsloth expects.

    Each example should have 'messages' key with system/user/assistant roles.
    We also add a reward-aware system prompt to teach the model to respond
    to evolution prompts generally.
    """
    formatted = []
    for ex in examples:
        if "messages" in ex:
            # Already in chat format (SFT output)
            formatted.append({"messages": ex["messages"]})
        elif "system" in ex and "user" in ex and "response" in ex:
            # Raw format
            formatted.append({
                "messages": [
                    {"role": "system", "content": ex["system"]},
                    {"role": "user", "content": ex["user"]},
                    {"role": "assistant", "content": ex["response"]},
                ]
            })
    return formatted


def train(
    data_path: Path,
    output_dir: Path,
    base_model: str = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
    max_steps: int = 100,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    batch_size: int = 2,
    gradient_accumulation: int = 4,
    max_seq_length: int = 4096,
):
    """Train QLoRA adapter using unsloth."""
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel

    print(f"Loading base model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load and format data
    print(f"Loading training data from {data_path}")
    raw_data = load_training_data(data_path)
    formatted = format_for_training(raw_data)
    print(f"Training examples: {len(formatted)}")

    if not formatted:
        print("No training data! Run extract_training_data.py first.")
        return

    # Create dataset
    dataset = Dataset.from_list(formatted)

    # Training config
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(5, max_steps // 10),
        fp16=True,
        logging_steps=5,
        save_steps=max_steps,
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print(f"Starting training: {max_steps} steps, lr={learning_rate}, rank={lora_rank}")
    trainer.train()

    # Save LoRA adapter
    model.save_pretrained(str(output_dir / "lora"))
    tokenizer.save_pretrained(str(output_dir / "lora"))
    print(f"LoRA adapter saved to {output_dir / 'lora'}")

    return model, tokenizer


def merge_and_export(
    output_dir: Path,
    base_model: str = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
    export_format: str = "gguf",  # "gguf" for Ollama, "hf" for HuggingFace
    quantization: str = "q4_k_m",
    max_seq_length: int = 4096,
):
    """Merge LoRA adapter with base model and export for Ollama."""
    from unsloth import FastLanguageModel

    lora_path = output_dir / "lora"
    if not lora_path.exists():
        print(f"No LoRA adapter found at {lora_path}")
        return

    print(f"Loading base model + LoRA from {lora_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(lora_path),
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if export_format == "gguf":
        print(f"Exporting to GGUF ({quantization})...")
        gguf_path = output_dir / "gguf"
        model.save_pretrained_gguf(
            str(gguf_path),
            tokenizer,
            quantization_method=quantization,
        )
        print(f"GGUF exported to {gguf_path}")

        # Create Ollama Modelfile
        gguf_file = list(gguf_path.glob("*.gguf"))[0]
        modelfile = gguf_path / "Modelfile"
        modelfile.write_text(
            f'FROM {gguf_file.name}\n'
            f'PARAMETER temperature 0.7\n'
            f'PARAMETER num_predict 4096\n'
            f'SYSTEM "You are a code mutation engine. You make improvements to Python functions."\n'
        )
        print(f"Modelfile created at {modelfile}")
        print(f"\nTo import into Ollama:")
        print(f"  cd {gguf_path}")
        print(f"  ollama create evol-mutator -f Modelfile")
    else:
        hf_path = output_dir / "merged_hf"
        model.save_pretrained_merged(str(hf_path), tokenizer)
        print(f"Merged model saved to {hf_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LoRA mutator model")
    parser.add_argument("--data", type=Path, default=Path("training_data_sft.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("lora_mutator"))
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--merge", action="store_true", help="Merge LoRA and export to GGUF")
    parser.add_argument("--quantization", default="q4_k_m")
    args = parser.parse_args()

    if args.merge:
        merge_and_export(args.output, args.base_model, quantization=args.quantization)
    else:
        train(
            args.data, args.output,
            base_model=args.base_model,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            lora_rank=args.rank,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()

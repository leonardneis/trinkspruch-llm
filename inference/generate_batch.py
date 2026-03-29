import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
ADAPTER_PATH = "models/lora"


def extract_response(text: str) -> str:
    if "### Response:" in text:
        text = text.split("### Response:", 1)[1]
    if "### Instruction:" in text:
        text = text.split("### Instruction:", 1)[0]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Gib mir einen Trinkspruch")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", type=str, default="data/generated_300.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    template = "### Instruction:\n{prompt}\n\n### Response:\n"

    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        while written < args.num_samples:
            current_batch = min(args.batch_size, args.num_samples - written)
            prompts = [template.format(prompt=args.prompt)] * current_batch

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    max_new_tokens=60,
                    pad_token_id=tokenizer.eos_token_id,
                )

            texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for text in texts:
                response = extract_response(text)
                item = {
                    "instruction": args.prompt,
                    "output": response,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1

            if written % 50 == 0 or written == args.num_samples:
                print(f"generated={written}/{args.num_samples}")

    print(f"saved={output_path}")


if __name__ == "__main__":
    main()

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
ADAPTER_PATH = "models/lora"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--prompt",
    default="Gib mir einen kurzen, lustigen Trinkspruch auf Deutsch:"
    )
    args = parser.parse_args()

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

    prompt = f"### Instruction:\n{args.prompt}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=60,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in generated:
        generated = generated.split("### Response:", 1)[1].strip()
    print(generated)


if __name__ == "__main__":
    main()

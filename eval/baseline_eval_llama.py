#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GSM8K evaluation on LLaMA-style prompts (few-shot vs direct).

What this script does:
- Builds LLaMA chat-format prompts (with <|start_header_id|>, <|eot_id|>, etc.).
- Evaluates a Hugging Face causal LM on the GSM8K test set.
- Compares two modes: "fewshot" (with a fixed few-shot prefix) vs "direct" (no prefix).
- Extracts numeric final answers from model outputs.
- Writes per-example predictions (JSONL) and a summary (JSON).

Requirements:
- PyTorch, transformers, tqdm.
- A GSM8K test file in JSONL with fields: {"question": ..., "answer": ...}.
- A local or remote HF model path for `AutoModelForCausalLM` and `AutoTokenizer`.

Example:
python eval_llama_gsm8k.py \
  --model_path ../models/Meta-Llama-3___1-8B-Instruct \
  --gsm8k_test_path ./data/gsm8k_test.jsonl \
  --dtype bfloat16 --gen_max_new_tokens 256 --gen_temperature 0.0
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================ Few-shot template (LLaMA) ============================
FEWSHOT_SYSTEM_TEXT = (
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8"
    "<|eot_id|>"
)

# =============== Question block (user + assistant headers; assistant is where model continues) ==========
QUESTION_BLOCK_TEMPLATE = (
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: {instruction}\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)
def build_question_block(instruction: str) -> str:
    return QUESTION_BLOCK_TEMPLATE.format(instruction=instruction.strip())


# ================================ Answer parsing ===================================
ANS_TRIGGERS = ['The final answer is', 'The answer is:', 'The answer is', 'the answer is', '####']

def extract_pred_answer(text: str) -> str:
    s = text.strip()
    for trig in ANS_TRIGGERS[:-1]:
        if trig in s:
            tail = s.split(trig, 1)[-1]
            m = re.findall(r'[-+]?\d+(\.\d+)?', tail)
            if m:
                m2 = re.findall(r'[-+]?\d+(?:\.\d+)?', tail)
                return m2[-1]
    m3 = re.findall(r'[-+]?\d+(?:\.\d+)?', s)
    if m3: return m3[-1]
    return ""

def parse_gsm8k_groundtruth(answer_field: str) -> str:
    m = re.search(r'####\s*([-+]?\d+(?:\.\d+)?)', answer_field)
    if m: return m.group(1)
    m2 = re.findall(r'[-+]?\d+(?:\.\d+)?', answer_field)
    if m2: return m2[-1]
    return ""


# ================================ Utilities ========================================
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_gsm8k(path: Path):
    """Load GSM8K JSONL (each line has 'question' and 'answer')."""
    data = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            q = obj.get("question", "")
            a = obj.get("answer", "")
            data.append({"question": q, "answer": a})
    return data

def ensure_bos(text: str, tokenizer) -> str:
    """Ensure the prompt starts with BOS if the tokenizer expects it."""
    bos = tokenizer.bos_token
    if bos and not text.lstrip().startswith(bos):
        return bos + text
    return text


# ================================ Core evaluation ==================================
def build_full_prompt(question: str, mode: str) -> str:
    """
    mode: 'fewshot' or 'direct'
    - fewshot: FEWSHOT_SYSTEM_TEXT + question block
    - direct : question block only
    """
    qblock = build_question_block(question)
    if mode == "fewshot":
        return FEWSHOT_SYSTEM_TEXT + qblock
    elif mode == "direct":
        return qblock
    else:
        raise ValueError(f"Unknown mode: {mode}")

@torch.no_grad()
def generate_answer(model, tokenizer, prompt_text: str, max_new_tokens: int, temperature: float) -> str:
    # Make sure BOS is present if needed
    prompt_text = ensure_bos(prompt_text, tokenizer)

    # Encode (we already inserted special tokens manually, so disable add_special_tokens)
    enc = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt')
    input_ids = enc.input_ids.to(model.device)
    attn = enc.attention_mask.to(model.device) if enc.get("attention_mask") is not None else None

    # Allow <|eot_id|> to serve as an additional stop token (besides EOS)
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id != -1:
        eos_ids.append(eot_id)
    if not eos_ids:
        eos_ids = None  # fallback to model default

    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature if temperature > 0 else 1.0,
        top_p=1.0,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode only the newly generated continuation (avoid few-shot numbers leaking)
    new_tokens = gen_ids[0][input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text

def evaluate_mode(model, tokenizer, dataset, mode: str, max_new_tokens: int, temperature: float):
    """Run evaluation for a single prompting mode and compute accuracy."""
    correct, total = 0, 0
    details = []
    for ex in tqdm(dataset, desc=f"[Eval] mode={mode}"):
        q = ex["question"]
        gold = parse_gsm8k_groundtruth(ex["answer"])

        prompt = build_full_prompt(q, mode=mode)
        out_text = generate_answer(model, tokenizer, prompt, max_new_tokens, temperature)

        pred = extract_pred_answer(out_text)
        hit = (pred == gold)
        correct += int(hit); total += 1

        details.append({
            "question": q,
            "gold": gold,
            "pred": pred,
            "hit": hit,
            "mode": mode,
            "raw_output": out_text
        })
    acc = correct / max(1, total)
    return {"accuracy": acc, "num": total, "correct": correct, "details": details}


# ================================ CLI entrypoint ===================================
def main():
    p = argparse.ArgumentParser(description="LLaMA-format GSM8K eval (few-shot vs direct) without memory compression")
    p.add_argument('--model_path', type=str, required=True,
                   help='e.g., ../models/Meta-Llama-3___1-8B-Instruct')
    p.add_argument('--tokenizer_path', type=str, default=None)
    p.add_argument('--gsm8k_test_path', type=str, default='./data/gsm8k_test.jsonl')
    p.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'float16', 'bfloat16'])
    p.add_argument('--gen_max_new_tokens', type=int, default=256)
    p.add_argument('--gen_temperature', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--preds_out', type=str, default='./out_llama_baseline_gsm8k_preds.jsonl')
    p.add_argument('--summary_out', type=str, default='./out_llama_baseline_gsm8k_summary.json')
    p.add_argument('--modes', type=str, default='fewshot,direct', help="comma-separated: fewshot,direct")
    args = p.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    tok_path = args.tokenizer_path or args.model_path
    print(f"⏳ Loading tokenizer from {tok_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"⏳ Loading model from {args.model_path} ...")
    # Prefer `attn_implementation="flash_attention_2"` when available; fall back to `use_flash_attention_2`.
    extra_load_kwargs = {}
    try:
        extra_load_kwargs["attn_implementation"] = "flash_attention_2" if torch_dtype != torch.float32 else "eager"
    except Exception:
        pass

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            **extra_load_kwargs
        )
    except TypeError:
        # If `attn_implementation` is unsupported, try the legacy `use_flash_attention_2` flag.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            use_flash_attention_2=True if torch_dtype != torch.float32 else False
        )

    model = model.to(device)
    model.eval()

    dataset = load_gsm8k(Path(args.gsm8k_test_path))

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_details = []
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": args.model_path,
        "test_file": args.gsm8k_test_path,
        "dtype": args.dtype,
        "gen_max_new_tokens": args.gen_max_new_tokens,
        "gen_temperature": args.gen_temperature,
        "results": {}
    }

    for mode in modes:
        res = evaluate_mode(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            mode=mode,
            max_new_tokens=args.gen_max_new_tokens,
            temperature=args.gen_temperature
        )
        summary["results"][mode] = {
            "accuracy": res["accuracy"],
            "num": res["num"],
            "correct": res["correct"]
        }
        all_details.extend(res["details"])
        print(f"✅ {mode} | GSM8K Acc = {res['accuracy']:.4f} ({res['correct']}/{res['num']})")

    # Write per-example predictions
    preds_out = Path(args.preds_out)
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    with preds_out.open('w', encoding='utf-8') as f:
        for d in all_details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Write summary
    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n==== Baseline Summary (LLaMA format) ====")
    for mode in modes:
        r = summary["results"][mode]
        print(f"{mode:8s}: acc={r['accuracy']:.4f}  correct={r['correct']}/{r['num']}")
    print(f"\nPredictions written to: {preds_out}")
    print(f"Summary written to    : {summary_out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
RoleLLM Win-Rate Evaluation (A/B/Tie) using a local Qwen judge.

What this does:
- Compares your model's responses (the "new" side) against one or more reference sets.
- Uses a local Qwen model as an automatic judge to decide A/B/Tie per instruction.
- Randomizes which side is A vs B to avoid position bias.
- Writes per-comparison logs (jsonl) and a text summary with win/tie/loss counts and win-rate.

Inputs you set:
- NUM_SAMPLES: how many of your model responses to evaluate (-1 = all)
- LLM_ANSWERS_PATH: path to your model's answers (jsonl)
- STANDARD_ANSWERS_PATHS: list of reference answer files (jsonl)
- OUTPUT_DIR: directory for logs & summaries
- QWEN_LOCAL_MODEL_PATH: your local Qwen path

Outputs:
- evaluation_summary__<ref_name>.txt in OUTPUT_DIR
- qwen_evaluations__<ref_name>.jsonl in OUTPUT_DIR

Assumptions:
- Your jsonl files contain "role", "question", and (for your side) "model_answer".
- Reference files may store the canonical answer under keys like "generated", "answer",
  "response", "output", "target", or "gold".
"""

import json
import os
import random
import time
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =========================
# 1) Config
# =========================

# Number of samples to evaluate; -1 means evaluate all
NUM_SAMPLES = -1  # e.g., set to 10 to only evaluate the first 10

# Path to your model's answers (the "new" side to be evaluated)
LLM_ANSWERS_PATH = './1boutputs-kdst852500-1-l/1boutputs-kdst852500-1-l.jsonl'

# Paths to one or more reference answer sets (the "baseline" side)
STANDARD_ANSWERS_PATHS = [
    # './data/RoleBench/rolebench-eng/instruction-generalization/role_specific/test.jsonl',
    './data/RoleBench/rolebench-eng/instruction-generalization/role_specific/rolegpt_baseline.jsonl',
]

# Output directory (per reference set, will write a summary and a detailed log)
OUTPUT_DIR = './1boutputs-kdst852500-1-l/'

# Local Qwen judge model path
QWEN_LOCAL_MODEL_PATH = '../models/Qwen/Qwen3-30B-A3B-Instruct-2507'

# Judge prompt (forces a single-token decision: A, B, or Tie)
PROMPT_TEMPLATE = """You are asked to compare two model responses to a given instruction.
Your goal is to decide which response is better.

[Instruction]
{instruction}

[Response A]
{response_a}

[Response B]
{response_b}

Please evaluate the two responses according to the following criteria:
1. Correctness: Is the content factually correct and relevant to the instruction?
2. Role alignment: If the instruction requires role-playing, does the response
   reflect the style, tone, and knowledge consistent with the role?
3. Completeness: Does the response address the instruction fully?
4. Fluency: Is the response fluent and natural English?

After considering these factors, output your judgment in the following format:
- "A" if Response A is better
- "B" if Response B is better
- "Tie" if both are equally good

IMPORTANT: Reply with ONLY one token among A, B, or Tie. Do not include any explanation.
"""

# Qwen generation settings (short, deterministic outputs)
GENERATION_KW = dict(
    max_new_tokens=8,
    do_sample=False,
    temperature=0.0,
)


# =========================
# 2) Utilities
# =========================

def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def load_jsonl(filepath: str) -> Optional[List[dict]]:
    """Load a .jsonl file into a list of dicts; return None if missing."""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def _try_get_standard_answer(item: dict) -> Optional[str]:
    """
    Robustly extract a reference answer from a variety of schemas.
    Priority:
      1) 'generated' (list or str)
      2) any of: 'answer' / 'response' / 'output' / 'target' / 'gold'
    """
    if "generated" in item:
        gen = item["generated"]
        if isinstance(gen, list) and len(gen) > 0:
            return gen[0]
        if isinstance(gen, str) and gen.strip():
            return gen

    for k in ["answer", "response", "output", "target", "gold"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
            return v[0]
    return None

def create_standard_answers_map(standard_answers_list: List[dict]) -> Dict[Tuple[str, str], str]:
    """Build a {(role, question): reference_answer} map from a list of reference items."""
    answers_map = {}
    for item in standard_answers_list:
        role = item.get("role")
        question = item.get("question")
        if role is None or question is None:
            continue
        std_ans = _try_get_standard_answer(item)
        if std_ans:
            key = (role, question)
            if key not in answers_map:
                answers_map[key] = std_ans
    return answers_map

def derive_suffix_from_path(path: str) -> str:
    """Derive a suffix (e.g. 'test' or 'rolegpt_baseline') from the reference filename."""
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name

def parse_qwen_judgment(text: str) -> str:
    """Parse Qwen raw output into 'A' / 'B' / 'TIE' / 'PARSE_ERROR'."""
    if not isinstance(text, str):
        return "PARSE_ERROR"
    s = text.strip().upper().replace('"', '').replace("'", "")
    if s.startswith("A"):
        return "A"
    if s.startswith("B"):
        return "B"
    if s.startswith("TIE"):
        return "TIE"
    # Loose fallback on the head
    head = s[:10]
    if head == "A":
        return "A"
    if head == "B":
        return "B"
    if "TIE" in head:
        return "TIE"
    return "PARSE_ERROR"


# =========================
# 3) Qwen judge wrapper
# =========================

class QwenJudge:
    """Use a local Qwen model as an automatic judge returning A/B/Tie."""

    def __init__(self, model_path: str):
        print(f"Loading local judge model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False
        )
        # Safety: align pad_token to eos if missing
        if self.pipe.tokenizer.pad_token_id is None and self.pipe.tokenizer.eos_token_id is not None:
            self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id

    def _build_messages(self, prompt_text: str) -> List[dict]:
        """Construct chat messages compatible with Qwen's chat template."""
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict and concise judge for pairwise comparison. "
                    "You MUST return exactly one token: A, B, or Tie."
                ),
            },
            {"role": "user", "content": prompt_text},
        ]

    def _apply_chat_template(self, messages: List[dict]) -> str:
        """Render messages into a single prompt string via the tokenizer's chat template."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def judge(self, instruction: str, response_a: str, response_b: str) -> dict:
        """
        Run a single A/B/Tie judgment.
        Returns a dict with raw output, parsed label, prompt text, timing, token usage, etc.
        """
        prompt = PROMPT_TEMPLATE.format(
            instruction=instruction,
            response_a=response_a,
            response_b=response_b
        )

        messages = self._build_messages(prompt)
        prompt_text = self._apply_chat_template(messages)

        # Count input tokens
        input_ids = self.pipe.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        input_token_count = int(input_ids.shape[-1])

        t0 = time.time()
        outputs = self.pipe(
            prompt_text,
            eos_token_id=self.pipe.tokenizer.eos_token_id,
            **GENERATION_KW
        )
        t1 = time.time()
        gen_text = outputs[0]["generated_text"]
        parsed = parse_qwen_judgment(gen_text)

        # Rough output token count (judge output only)
        out_token_count = len(self.pipe.tokenizer(gen_text, add_special_tokens=False).input_ids)

        return {
            "raw_output": gen_text,
            "parsed": parsed,
            "prompt_text": prompt_text,
            "timing": {
                "start": t0,
                "end": t1,
                "duration_sec": round(t1 - t0, 4),
            },
            "token_usage": {
                "prompt_tokens": input_token_count,
                "output_tokens": out_token_count,
            },
            "generation_kwargs": GENERATION_KW,
            "model_path": QWEN_LOCAL_MODEL_PATH,
        }


# =========================
# 4) Evaluate against one reference set
# =========================

def evaluate_against_standard(
    qwen: QwenJudge,
    llm_answers_path: str,
    standard_answers_path: str,
    output_dir: str,
    num_samples: int = -1,
    seed: int = 42
):
    random.seed(seed)
    ensure_dir(output_dir)

    suffix = derive_suffix_from_path(standard_answers_path)
    summary_file = os.path.join(output_dir, f"evaluation_summary__{suffix}.txt")
    log_file = os.path.join(output_dir, f"qwen_evaluations__{suffix}.jsonl")

    print(f"\n=== Start evaluation: {os.path.basename(llm_answers_path)}  VS  {os.path.basename(standard_answers_path)} ===")
    print(f"Num samples setting: {'ALL' if num_samples == -1 else num_samples}")

    # Load reference answers
    std_list = load_jsonl(standard_answers_path)
    if std_list is None:
        print("Failed to load reference answers, skipping.")
        return
    std_map = create_standard_answers_map(std_list)
    print(f"Loaded reference answers (deduped): {len(std_map)}")

    # Load our model answers
    llm_list = load_jsonl(llm_answers_path)
    if llm_list is None:
        print("Failed to load model answers, skipping.")
        return

    # Truncate if needed
    if num_samples != -1 and num_samples > 0:
        if len(llm_list) > num_samples:
            print(f"Detected {len(llm_list)} model answers, evaluating only the first {num_samples}.")
            llm_list = llm_list[:num_samples]
        else:
            print(f"Warning: requested {num_samples} samples, but only {len(llm_list)} available. Evaluating all found data.")

    print(f"Prepared to evaluate {len(llm_list)} model answers.")

    # Counters
    wins = 0
    ties = 0
    losses = 0
    errors = 0
    total_comparisons = 0
    evaluation_details = []

    with open(log_file, 'w', encoding='utf-8') as fout:
        for idx, item in enumerate(llm_list):
            role = item.get("role")
            question = item.get("question")
            model_answer = item.get("model_answer")

            if role is None or question is None or model_answer is None:
                print(f"[{idx}] Missing required fields, skip.")
                errors += 1
                evaluation_details.append({"index": idx, "result": "MISSING_FIELDS"})
                continue

            # Match reference answer
            std_answer = std_map.get((role, question))
            if std_answer is None:
                q_preview = str(question)[:50]
                print(f"[{idx}] No reference found for role='{role}', question starts with '{q_preview}...'; skip.")
                errors += 1
                evaluation_details.append({"index": idx, "result": "STD_NOT_FOUND"})
                continue

            total_comparisons += 1
            print(f"Evaluating item {total_comparisons}/{len(llm_list)} ...")

            # Randomly assign A/B to avoid position bias
            our_model_is_A = random.choice([True, False])
            if our_model_is_A:
                response_a = model_answer
                response_b = std_answer
            else:
                response_a = std_answer
                response_b = model_answer

            # Ask Qwen to judge
            judge_ret = qwen.judge(question, response_a, response_b)
            raw_output = judge_ret["raw_output"]
            parsed = judge_ret["parsed"]

            # Tally result
            if parsed == "A":
                if our_model_is_A:
                    wins += 1
                    cat = "win"
                else:
                    losses += 1
                    cat = "loss"
            elif parsed == "B":
                if our_model_is_A:
                    losses += 1
                    cat = "loss"
                else:
                    wins += 1
                    cat = "win"
            elif parsed == "TIE":
                ties += 1
                cat = "tie"
            else:
                errors += 1
                cat = "error"

            log_entry = {
                "role": role,
                "question": question,
                "model_answer": model_answer,
                "standard_answer": std_answer,
                "our_model_assigned_to": "A" if our_model_is_A else "B",
                "qwen_raw_output": raw_output,
                "qwen_parsed_judgment": parsed,
                "result_category": cat,
                "qwen_prompt_text": judge_ret.get("prompt_text", ""),
                "qwen_timing": judge_ret.get("timing", {}),
                "qwen_token_usage": judge_ret.get("token_usage", {}),
                "qwen_generation_kwargs": judge_ret.get("generation_kwargs", {}),
                "qwen_model_path": judge_ret.get("model_path", ""),
                "index": idx,
            }
            fout.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            evaluation_details.append({"index": idx, "result": cat})

    # Compute win-rate
    if total_comparisons > 0:
        win_rate = (wins + 0.5 * ties) / total_comparisons
    else:
        win_rate = 0.0

    # Write summary
    summary_content = (
        f"--- Evaluation Summary (Reference: {os.path.basename(standard_answers_path)}) ---\n\n"
        f"Requested samples: {'ALL' if num_samples == -1 else num_samples} (executed: {total_comparisons})\n\n"
        f"Total comparisons: {total_comparisons}\n"
        f"Wins: {wins}\n"
        f"Losses: {losses}\n"
        f"Ties: {ties}\n"
        f"Errors/Unparsed: {errors}\n\n"
        f"Final Win-Rate (wins + 0.5 * ties) / total: {win_rate:.4f}\n\n"
        f"--- Per-item results ---\n"
    )
    for detail in evaluation_details:
        summary_content += f"Item {detail['index']}: {detail['result']}\n"

    with open(summary_file, 'w', encoding='utf-8') as sf:
        sf.write(summary_content)

    print("\n--- Evaluation finished ---")
    print(f"Summary written to: {summary_file}")
    print(f"Qwen detailed log written to: {log_file}")
    print(f"Final win-rate: {win_rate:.4f}")


# =========================
# 5) Entry point: evaluate across all reference sets
# =========================

def main():
    ensure_dir(OUTPUT_DIR)

    # Load local Qwen judge
    qwen = QwenJudge(QWEN_LOCAL_MODEL_PATH)

    # Evaluate for each reference set and write separate outputs
    for std_path in STANDARD_ANSWERS_PATHS:
        evaluate_against_standard(
            qwen=qwen,
            llm_answers_path=LLM_ANSWERS_PATH,
            standard_answers_path=std_path,
            output_dir=OUTPUT_DIR,
            num_samples=NUM_SAMPLES,
            seed=42
        )

if __name__ == "__main__":
    main()

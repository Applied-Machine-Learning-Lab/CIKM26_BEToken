#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Few-shot Role-Play Inference on RoleLLM/RoleBench-style data.

Key additions in this version:
- Completed `format_prompt` for three styles: short / llama / qwen3.
- Added `--prompt_style` CLI flag and propagate both `style` and `user_name`.
- English comments and docstrings for clarity.

Prompt construction strategy:
1) Build a compressed "system segment" (system instruction + few-shot pairs).
2) For each test question, append a style-specific user turn produced by `format_prompt(...)`.
3) Generate with the base CausalLM.

Notes:
- The system segment is plain text by design (compressed header + few-shot). We then switch into a chat-style
  user/assistant block only for the real question, which works reliably with common instruction-tuned LLMs.
- If you prefer fully strict chat formatting (including a system role token), you can extend the code to wrap
  `system_prompt` into model-specific system blocks; this script keeps it simple and model-agnostic.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import random

import torch
from torch.utils.data import Dataset, DataLoader  # unused, kept for future extension
from tqdm.auto import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================ Prompt style support ============================
# Plain, one-shot prompt template (no chat tokens)
SHORT_PROMPT_TEMPLATE = "\nQuestion:\n{instruction}\nAnswer:\n"

# ---- Llama 3 chat-style special tokens (meta headers) ----
LLAMA3_USER_BLOCK = "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
LLAMA3_ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_RESPONSE_SUFFIX = "<|eot_id|>"

# ---- Qwen3 (ChatML-style) special tokens ----
QWEN3_USER_BLOCK = "<|im_start|>user\n{instruction}<|im_end|>\n"
QWEN3_ASSISTANT_PREFIX = "<|im_start|>assistant\n"
QWEN3_RESPONSE_SUFFIX = "<|im_end|>"

def format_prompt(
    instruction: str,
    style: str = "short",
    user_name: Optional[str] = None,
) -> str:
    """
    Build the *user-turn + assistant-prefix* portion according to a given style.

    Parameters
    ----------
    instruction : str
        The user question/content for the current sample.
    style : str
        One of {"short", "llama", "qwen3"}.
    user_name : Optional[str]
        A visible speaker tag to include inside the user content (purely textual).
        This does *not* change the special token role; it helps traceability in logs.

    Returns
    -------
    str
        The formatted text that, when appended after the system+fewshot segment,
        makes the model ready to generate the assistant's answer.

    Contract per style
    ------------------
    - "short":
        Question/Answer format, e.g.
            Question:
            <text>
            Answer:
    - "llama":
        Emits a user block ending with <|eot_id|>, followed by an assistant header.
    - "qwen3":
        Emits a user block ending with <|im_end|>, followed by an assistant header.

    Rationale
    ---------
    We keep the "system segment" as already-built plain text (compressed header + few-shot examples)
    and only format the *live user turn* via chat tokens. This approach avoids mixing special tokens
    inside the few-shot text while still giving the generation model a clear assistant entry point.
    """
    # Optionally inject the speaker name into the *content*, not the role tag.
    payload = instruction if not user_name else f"{user_name}: {instruction}"

    if style == "short":
        # Classic instruct format, no chat tokens.
        return SHORT_PROMPT_TEMPLATE.format(instruction=payload)

    elif style == "llama":
        # Llama 3 chat-style: user block + assistant header (no body yet).
        return f"{LLAMA3_USER_BLOCK.format(instruction=payload)}{LLAMA3_ASSISTANT_PREFIX}"

    elif style == "qwen3":
        # Qwen 3 (ChatML-like): user block + assistant header (no body yet).
        return f"{QWEN3_USER_BLOCK.format(instruction=payload)}{QWEN3_ASSISTANT_PREFIX}"

    else:
        raise ValueError(f"Unknown prompt_style: {style}")

def format_response(answer: str, style: str = "short") -> str:
    """
    Optionally wrap/terminate a generated answer with model-specific suffix tokens.
    Not strictly required when decoding with `skip_special_tokens=True`, but useful
    if you ever need to *manually* prepare the target text for supervised fine-tuning.
    """
    if style == "short":
        return answer
    elif style == "llama":
        return f"{answer}{LLAMA3_RESPONSE_SUFFIX}"
    elif style == "qwen3":
        return f"{answer}{QWEN3_RESPONSE_SUFFIX}"
    else:
        raise ValueError(f"Unknown prompt_style: {style}")


# ====================== System prompt builder (Few-shot) ======================
FEWSHOT_SYSTEM_HEADER_TEMPLATE = (
    "System Instruction:\n"
    "You are {role_name}, your description is: {role_description_and_catchphrases}.\n"
    "Now please answer some questions to accurately show your personality traits!\n"
    "Your speaking style should fully imitate the personality role assigned to you!\n"
    "Please do not expose that you are an artificial intelligence model or a language model,\n"
    "you must always remember that you are only assigned one personality role.\n"
    "Don't be verbose or too formal or polite when speaking.\n\n"
)

def safe_read_json(fp: Path) -> Any:
    with fp.open('r', encoding='utf-8') as f:
        return json.load(f)

def ensure_text(x) -> str:
    if isinstance(x, str): return x
    if isinstance(x, list): return " ".join(str(e) for e in x)
    if isinstance(x, dict): return json.dumps(x, ensure_ascii=False)
    return str(x) if x is not None else ""

def resolve_role_and_desc(desc_json_path: Path, role_name: str) -> Tuple[str, str]:
    """
    Resolve a role's description by (case-insensitive) key lookup.
    Returns (role_desc, resolved_role_name).
    """
    desc = safe_read_json(desc_json_path)
    if role_name not in desc:
        keys_ci = {k.lower(): k for k in desc.keys()}
        if role_name.lower() in keys_ci:
            key = keys_ci[role_name.lower()]
        else:
            key = list(desc.keys())[0]
            print(f"[WARN] Role '{role_name}' not found in desc.json. Falling back to '{key}'.")
        role_name = key
    role_desc = ensure_text(desc[role_name])
    return role_desc, role_name

def build_system_prompt_with_fewshot(role_name: str, role_desc: str,
                                     fewshot_pairs: List[Tuple[str, str]]) -> str:
    """
    Build the *compressed* system segment:
    - A system instruction block tailored to `role_name` and `role_desc`.
    - Then K in-context examples rendered as: "User Prompt:\n... \nAssistant Prompt:\n...\n\n"
      (We do not inject chat tokens here on purpose; it's kept as plain text.)

    The real user question for each test sample will be appended later via `format_prompt(...)`.
    """
    parts = [FEWSHOT_SYSTEM_HEADER_TEMPLATE.format(
        role_name=role_name,
        role_description_and_catchphrases=role_desc
    )]
    for i, (q, a) in enumerate(fewshot_pairs, 1):
        q_txt = ensure_text(q).strip()
        a_txt = ensure_text(a).strip()
        parts.append(f"User Prompt:\n{q_txt}\nAssistant Prompt:\n{a_txt}\n\n")
    return "".join(parts)

def guess_role_file(instructions_dir: Path, role_name: str) -> Optional[Path]:
    """
    Try to locate a role-specific training file under `instructions_dir`.
    Several normalized variants of the file name are probed.
    """
    candidates = [
        instructions_dir / f"role-specific-{role_name}.jsonl",
        instructions_dir / f"role-specific-{role_name.replace(' ', '_')}.jsonl",
        instructions_dir / f"role-specific-{role_name.replace(' ', '-')}.jsonl",
        instructions_dir / f"role-specific-{re.sub(r'[^A-Za-z0-9_-]+','', role_name)}.jsonl",
    ]
    for c in candidates:
        if c.exists(): return c
    for p in instructions_dir.glob("role-specific-*.jsonl"):
        if role_name.lower().replace(' ', '') in p.stem.lower().replace(' ', ''):
            return p
    return None


# =============================== Dataset / IO ================================
def load_jsonl(fp: Path) -> List[Dict[str, Any]]:
    data = []
    with fp.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: data.append(json.loads(line))
    return data

def collect_all_roles_from_test_jsonl(fp: Path) -> List[str]:
    """
    Scan the test JSONL to collect unique role names (case-insensitive uniqueness).
    """
    roles, seen, uniq = [], set(), []
    with fp.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            r = obj.get("role") or obj.get("role_name") or obj.get("character")
            if r: roles.append(str(r).strip())
    for r in roles:
        key = r.lower().strip()
        if key not in seen:
            seen.add(key); uniq.append(r)
    return uniq


# =================================== Utils ==================================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ============================ Inference (Few-shot) ===========================
def generate_for_role(
    model,
    tokenizer,
    args,
    role_name: str,
    outputs_root: Path,
    role_outputs_jsonl_path: Path,
) -> Dict[str, Any]:

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # ---------- Resolve role description (for the system instruction) ----------
    desc_json_path = Path(args.desc_json)
    role_desc, role_resolved = resolve_role_and_desc(desc_json_path, role_name)

    # ---------- Load role-specific training pairs (first K used as few-shot) ----------
    instr_dir = Path(args.instructions_dir)
    role_file = guess_role_file(instr_dir, role_resolved)
    if role_file is None:
        print(f"[WARN] Could not find role-specific instructions file for role '{role_resolved}' in {instr_dir}")
        role_train = []
    else:
        role_train = load_jsonl(role_file)

    fewshot_k = max(0, int(args.fewshot_k))
    fewshot_pairs = []
    for r in role_train[:fewshot_k]:
        fewshot_pairs.append((ensure_text(r.get("instruction","")), ensure_text(r.get("answer",""))))

    # ---------- Build the compressed system header (system + few-shot) ----------
    system_prompt = build_system_prompt_with_fewshot(
        role_name=role_resolved,
        role_desc=role_desc,
        fewshot_pairs=fewshot_pairs
    )

    print(f"\n🎭 Role => {role_resolved}")
    print("🧾 SYSTEM_PROMPT (first 400 chars):")
    print(system_prompt[:400] + ("..." if len(system_prompt) > 400 else ""))

    # ---------- Load test set and filter samples belonging to this role ----------
    all_test = load_jsonl(Path(args.test_jsonl))
    test_samples = []
    for r in all_test:
        r_role = r.get("role") or r.get("role_name") or r.get("character") or ""
        if str(r_role).strip().lower() == role_resolved.strip().lower():
            item = r.copy()
            # Common field fallback for question text.
            item["question"]  = ensure_text(item.get("question", "") or item.get("instruction", "") or item.get("prompt", ""))
            test_samples.append(item)

    if len(test_samples) == 0:
        print(f"[WARN] No test samples found for role '{role_resolved}'.")
        out_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "role_name": role_resolved,
            "num_samples": 0,
            "artifacts": {
                "system_prompt_path": str(outputs_root / f"{role_resolved}_system_prompt.txt"),
                "outputs_path": str(role_outputs_jsonl_path),
            },
        }
        return out_record

    # Save the system header for reproducibility/debugging.
    outputs_root.mkdir(parents=True, exist_ok=True)
    sys_path = outputs_root / f"{role_resolved}_system_prompt.txt"
    with sys_path.open('w', encoding='utf-8') as f:
        f.write(system_prompt)

    # ---------- Inference ----------
    model.eval()
    total = 0

    for idx, item in enumerate(tqdm(test_samples, desc=f"[{role_resolved}] Generating", leave=False)):
        question = ensure_text(item.get("question", "")).strip()
        if not question:
            continue

        # Compose final prompt: system header + user turn (style-aware)
        prompt_text = system_prompt + format_prompt(
            question,
            style=args.prompt_style,
            user_name=args.user_name
        )

        inputs = tokenizer(prompt_text, return_tensors='pt').to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.no_sample,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Only keep the newly generated tail beyond the prompt length
        full_out = gen_ids[0]
        new_tokens = full_out[inputs["input_ids"].shape[1]:]
        pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        out_line = {
            "role": role_resolved,
            "question": question,
            "model_answer": pred
        }
        with role_outputs_jsonl_path.open('a', encoding='utf-8') as wf:
            wf.write(json.dumps(out_line, ensure_ascii=False) + "\n")

        total += 1

        # Optional: print a couple of previews
        if idx < 2:
            print("-"*80)
            print(f"[{idx+1}/{len(test_samples)}] Role: {role_resolved}")
            print(f"Q: {question}")
            print(f"PRED: {pred}")

    print("\n" + "="*80)
    print(f"FINAL: Saved {total} model outputs for role '{role_resolved}'.")
    print("="*80 + "\n")

    out_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "role_name": role_resolved,
        "num_samples": total,
        "artifacts": {
            "system_prompt_path": str(sys_path),
            "outputs_path": str(role_outputs_jsonl_path),
        },
    }
    return out_record


# ================================== Main ====================================
def parse_arguments():
    p = argparse.ArgumentParser(
        description=(
            "Few-shot Role-Play Inference: Build system prompt (header + few-shot) "
            "and generate answers. No memory tokens/KD/AE."
        )
    )
    # Model / Tokenizer
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--tokenizer_path', type=str, default=None)

    # RoleBench paths
    p.add_argument('--desc_json', type=str, default="data/RoleBench/profiles-eng/desc.json")
    p.add_argument('--instructions_dir', type=str, default="data/RoleBench/instructions-eng")
    p.add_argument('--test_jsonl', type=str, default="data/RoleBench/rolebench-eng/instruction-generalization/role_specific/test.jsonl")

    # Few-shot settings
    p.add_argument('--fewshot_k', type=int, default=5, help='Number of few-shot example pairs (K) per role.')
    p.add_argument('--user_name', type=str, default='User', help='A visible tag added in the user content for traceability.')

    # Prompt style
    p.add_argument('--prompt_style', type=str, default='short', choices=['short', 'llama', 'qwen3'],
                   help='Choose how the *user turn* is formatted.')

    # Inference config
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--repetition_penalty', type=float, default=1.0)
    p.add_argument('--no_sample', action='store_true', help='Disable sampling (use deterministic decoding).')

    # Output
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--role_specific_outputs_jsonl', type=str, default=None)

    # Others
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'float16', 'bfloat16'])
    p.add_argument('--num_roles', type=int, default=0, help='If >0, only evaluate the first N roles in test.jsonl order.')

    return p.parse_args()

def main():
    args = parse_arguments()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Device / dtype
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    desired_dtype = getattr(torch, args.dtype)
    if device == 'cpu' and desired_dtype != torch.float32:
        print("[INFO] Forcing float32 on CPU.")
        desired_dtype = torch.float32

    # Tokenizer
    tok_path = args.tokenizer_path or args.model_path
    print(f"⏳ Loading tokenizer from {tok_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    print(f"⏳ Loading base model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=desired_dtype,
        use_flash_attention_2=True if desired_dtype != torch.float32 else False
    ).to(device)
    model.eval()

    outputs_root = Path(args.output_dir)
    role_outputs_jsonl_path = Path(args.role_specific_outputs_jsonl) if args.role_specific_outputs_jsonl \
        else outputs_root / 'role_specific_outputs.jsonl'
    role_outputs_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Roles to evaluate
    test_path = Path(args.test_jsonl)
    all_roles = collect_all_roles_from_test_jsonl(test_path)
    if args.num_roles and args.num_roles > 0:
        all_roles = all_roles[:args.num_roles]
    print(f"🚀 Selected {len(all_roles)} roles.")

    total_samples = 0
    per_role_counts = []

    for role_name in all_roles:
        print("\n" + "#"*100)
        print(f" Start Role => {role_name} | K={args.fewshot_k} | style={args.prompt_style}")
        print("#"*100 + "\n")
        try:
            rec = generate_for_role(
                model=model,
                tokenizer=tokenizer,
                args=args,
                role_name=role_name,
                outputs_root=outputs_root,
                role_outputs_jsonl_path=role_outputs_jsonl_path,
            )
            if rec.get("num_samples", 0) > 0:
                total_samples += rec["num_samples"]
                per_role_counts.append((rec["role_name"], int(rec["num_samples"])))
        except Exception as e:
            print(f"[ERROR] Role '{role_name}' failed: {e}")

    print("\n" + "="*80)
    print("OVERALL RESULTS (inference only)")
    print("="*80)
    print(f"Total evaluated roles  : {len(per_role_counts)}")
    print(f"Total generated samples: {total_samples}")
    print("="*80 + "\n")

    # Optional summary artifact
    summary_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "num_roles_evaluated": len(per_role_counts),
            "num_samples": total_samples,
        },
        "outputs_path": str(role_outputs_jsonl_path),
    }
    with (outputs_root / 'summary.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

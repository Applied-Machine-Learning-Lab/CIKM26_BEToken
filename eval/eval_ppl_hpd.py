#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HPD PPL evaluation using BE token only (no baseline system prompt).

What this script does
---------------------
- Builds a user/assistant-style prompt (no system block) for each HPD sample.
- Prepends learned BE memory embeddings to the model input during inference.
- Computes NLL **only** on gold answer tokens (instruction/prompt tokens are ignored).
- Optionally forces the model to predict an EOT token after the gold answer.
- Reports two aggregations:
    1) token-weighted PPL (sum over tokens / total target tokens)
    2) sample-averaged PPL (mean loss over samples)

Inputs you need
---------------
- --model_path: path to HF causal LM
- --tokenizer_path: optional (defaults to model_path)
- --test_json: HPD test set JSON (e.g., hpd/dataset/en_test_set.json)
- --mem_vector_path: saved BE tensor (shape [N_mem_tokens, hidden_dim])
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================ Prompt style support ============================
# A short plain Q/A prompt (no chat headers)
SHORT_PROMPT_TEMPLATE = "\nQuestion:\n{instruction}\nAnswer:\n"

# Llama3-style chat blocks (we only use user + assistant prefix; no system block)
LLAMA3_USER_BLOCK = "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
LLAMA3_ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"

# Qwen2/3-style chat blocks (we only use user + assistant prefix; no system block)
QWEN3_USER_BLOCK = "<|im_start|>user\n{instruction}<|im_end|>\n"
QWEN3_ASSISTANT_PREFIX = "<|im_start|>assistant\n"

def format_prompt(instruction: str, style: str = "llama") -> str:
    """Return the chat-like prompt (no system block) for a given style."""
    if style == "short":
        return SHORT_PROMPT_TEMPLATE.format(instruction=instruction)
    elif style == "llama":
        return f"{LLAMA3_USER_BLOCK.format(instruction=instruction)}{LLAMA3_ASSISTANT_PREFIX}"
    elif style == "qwen3":
        return f"{QWEN3_USER_BLOCK.format(instruction=instruction)}{QWEN3_ASSISTANT_PREFIX}"
    else:
        raise ValueError(f"Unknown prompt_style: {style}")

# ================================= MemoryCell =================================
class MemoryCell(torch.nn.Module):
    """
    Inference wrapper that prepends BE memory embeddings before the token embeddings.
    Model weights remain frozen; only 'memory' is provided (pretrained/saved).
    """
    def __init__(self, base_model, num_mem_tokens: int, memory_dim: int):
        super().__init__()
        self.model = base_model
        self.num_mem_tokens = num_mem_tokens
        self.memory_dim = memory_dim
        for _, p in self.model.named_parameters():
            p.requires_grad = False
        emb = self.model.get_input_embeddings()
        self.register_parameter(
            "memory",
            torch.nn.Parameter(
                torch.zeros((self.num_mem_tokens, self.memory_dim),
                            device=emb.weight.device, dtype=emb.weight.dtype),
                requires_grad=False
            )
        )

    def set_memory(self, input_shape):
        return self.memory.repeat(input_shape[0], 1, 1)

    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        mem_mask = torch.ones(shape[0], self.num_mem_tokens,
                              dtype=attention_mask.dtype, device=attention_mask.device)
        return torch.cat([mem_mask, attention_mask], dim=1)

    def forward(self, input_ids=None, memory_state=None, **kwargs):
        # Prepare memory for batch size inferred from inputs
        if memory_state is None:
            if input_ids is not None:
                memory_state = self.set_memory(input_ids.shape)
            else:
                inputs_embeds = kwargs.get("inputs_embeds")
                if inputs_embeds is None:
                    raise ValueError("Either input_ids/inputs_embeds or memory_state must be provided.")
                fake_ids = torch.empty((inputs_embeds.shape[0], 1),
                                       dtype=torch.long, device=inputs_embeds.device)
                memory_state = self.set_memory(fake_ids.shape)

        # Build inputs_embeds = [BE_memory; token_embeds]
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None:
            if input_ids is None:
                full_inputs_embeds = memory_state
            else:
                tok_emb = self.model.get_input_embeddings()(input_ids)
                full_inputs_embeds = torch.cat([memory_state, tok_emb], dim=1)
        else:
            full_inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)

        # Attention mask with BE prefix
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(full_inputs_embeds.shape[:2],
                                        dtype=torch.long, device=full_inputs_embeds.device)
        else:
            attention_mask = self.pad_attention_mask(attention_mask, full_inputs_embeds.shape)

        labels = kwargs.get("labels", None)

        return self.model(
            input_ids=None,
            inputs_embeds=full_inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        ), memory_state

# =============================== HPD Dataset =================================
def build_hpd_instruction(sample: Dict[str, Any]) -> Tuple[str, str]:
    """
    Turn an HPD JSON sample into a single instruction block and its gold answer.
    The instruction includes scene/meta info + dialogue; the target is the gold response.
    """
    scene = sample.get("scene", "")
    position = sample.get("position", "")
    speakers = sample.get("speakers", [])
    speakers_str = ", ".join(map(str, speakers)) if isinstance(speakers, (list, tuple)) else str(speakers)
    attributes = sample.get("attributes", {})
    harry_attr = attributes.get("Harry", attributes.get("harry", attributes))
    rel = sample.get("relations with Harry", sample.get("relations_with_Harry", sample.get("relations_with_harry", {})))
    dialogue_lines = sample.get("dialogue", [])
    dialogue_text = "\n".join(dialogue_lines) if isinstance(dialogue_lines, list) else str(dialogue_lines)

    instruction = (
        f"Scene: {scene}\n"
        f"Dialogue Position: {position}\n"
        f"Speakers: {speakers_str}\n\n"
        f"Harry’s attributes: {json.dumps(harry_attr, ensure_ascii=False)}\n\n"
        f"Speakers relations with Harry: {json.dumps(rel, ensure_ascii=False)}\n"
        f"Dialogue:\n{dialogue_text}\n"
        "Harry's Response:\n"
    )
    answer = sample.get("positive_response", sample.get("answer", "")) or ""
    return instruction, str(answer)

# =============================== Eval helpers ================================
def load_hpd_test_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        rows = list(data.values())
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError("Unknown JSON structure for HPD test set.")
    return rows

def maybe_append_eot(ans_ids: List[int], eos_id: int | None, enable: bool = True) -> List[int]:
    """If enabled and eos_id exists, ensure the final target token is eos."""
    if not enable or eos_id is None:
        return ans_ids
    if len(ans_ids) == 0 or ans_ids[-1] != eos_id:
        return ans_ids + [eos_id]
    return ans_ids

# =================================== Main ====================================
def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate PPL on HPD EN-TEST with [BE] (no baseline)")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, default=None)
    ap.add_argument("--test_json", type=str, required=True, help="Path to hpd/dataset/en_test_set.json")
    ap.add_argument("--mem_vector_path", type=str, required=True, help="Path to saved [BE] tensor")
    ap.add_argument("--N_mem_tokens", type=int, default=1)
    ap.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="bfloat16")
    ap.add_argument("--prompt_style", type=str, choices=["llama", "qwen3", "short"], default="llama")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--predict_eot", action="store_true",
                    help="If set, force the model to predict eos_token_id after gold answer.")
    ap.add_argument("--output_json", type=str, default=None, help="Optional path to write summary json")
    return ap.parse_args()

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    tok_path = args.tokenizer_path or args.model_path
    print(f"⏳ Loading tokenizer from {tok_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"⏳ Loading model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2"
    )
    model.to(device).eval()

    # EOT id (used if --predict_eot)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if args.predict_eot and eos_id is None:
        print("⚠️ tokenizer.eos_token_id is None; will skip EOT prediction.")

    # Build the BE wrapper and load the saved BE tensor
    print(f"⏳ Loading [BE] vector from {args.mem_vector_path} ...")
    mem_tensor = torch.load(args.mem_vector_path, map_location=device)
    if mem_tensor.dim() == 3:
        mem_tensor = mem_tensor[0]
    cfg = model.config
    memory_dim = getattr(cfg, "word_embed_proj_dim", getattr(cfg, "hidden_size"))
    mem_wrapper = MemoryCell(model, num_mem_tokens=args.N_mem_tokens, memory_dim=memory_dim).to(device)
    if mem_tensor.shape != mem_wrapper.memory.data.shape:
        raise ValueError(f"[BE] shape mismatch: loaded {tuple(mem_tensor.shape)} vs wrapper {tuple(mem_wrapper.memory.data.shape)}")
    with torch.no_grad():
        mem_wrapper.memory.data.copy_(mem_tensor.to(dtype))
    mem_wrapper.eval()

    rows = load_hpd_test_json(args.test_json)
    if args.max_samples and args.max_samples > 0:
        rows = rows[:args.max_samples]
    print(f"📦 Loaded {len(rows)} HPD test samples")

    # Aggregations
    mem_sum_loss = 0.0
    mem_sum_tokens = 0
    mem_losses: List[float] = []

    pbar = tqdm(rows, desc="Evaluating (BE only)")
    for sample in pbar:
        instruction, answer = build_hpd_instruction(sample)
        if not answer.strip():
            continue

        # Prompt: user + assistant prefix (no system block)
        prompt_text = format_prompt(instruction, style=args.prompt_style)
        pr_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        ans_ids = tokenizer(answer, add_special_tokens=False).input_ids
        ans_ids = maybe_append_eot(ans_ids, eos_id, enable=args.predict_eot)

        if len(ans_ids) == 0:
            continue

        # Student input contains [prompt tokens + answer tokens]; BE is injected via wrapper
        student_ids = torch.tensor([pr_ids + ans_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(student_ids, dtype=torch.long)

        # Labels: ignore BE prefix and prompt tokens; compute loss only on answer tokens
        labels = torch.cat([
            torch.full((args.N_mem_tokens,), -100, dtype=torch.long),
            torch.full((len(pr_ids),), -100, dtype=torch.long),
            torch.tensor(ans_ids, dtype=torch.long)
        ], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            out, _ = mem_wrapper(input_ids=student_ids, attention_mask=attention_mask, labels=labels)
            loss = float(out.loss.detach().cpu())

        # token-weighted
        mem_sum_loss += loss * len(ans_ids)
        mem_sum_tokens += len(ans_ids)
        # sample-averaged
        mem_losses.append(loss)

    # ----- Metrics -----
    def safe_ppl_from_sum(sum_loss: float, sum_tok: int) -> float:
        if sum_tok == 0:
            return float("nan")
        avg_nll = sum_loss / sum_tok
        return math.exp(avg_nll)

    def safe_ppl_from_list(losses: List[float]) -> float:
        if len(losses) == 0:
            return float("nan")
        return math.exp(sum(losses) / len(losses))

    summary = {
        "mem": {
            "target_tokens": mem_sum_tokens,
            "avg_nll_token_weighted": (mem_sum_loss / mem_sum_tokens) if mem_sum_tokens > 0 else None,
            "ppl_token_weighted": safe_ppl_from_sum(mem_sum_loss, mem_sum_tokens),
            "ppl_sample_avg": safe_ppl_from_list(mem_losses),
            "num_samples": len(mem_losses),
        }
    }

    print("\n==== Compressed ([BE] only) ====")
    if mem_sum_tokens > 0:
        print(f"Total target tokens: {mem_sum_tokens}")
        print(f"PPL (token-weighted): {summary['mem']['ppl_token_weighted']:.4f}")
        print(f"PPL (sample-avg)   : {summary['mem']['ppl_sample_avg']:.4f}")
    else:
        print("No target tokens (skipped).")

    print("\n==== Summary (JSON) ====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n📝 Saved summary JSON to: {out_path}")

if __name__ == "__main__":
    main()

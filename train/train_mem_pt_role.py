#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-Token Training (Ablation Study) — prompt-style support, PT (prompt tuning).
This script trains and evaluates a memory-token "cell" that prepends learnable
memory embeddings to a frozen base LLM, targeting ablation on memory-token style
(w/o [AE] + prompt tuning in the broader study context).

Features:
  • Prompt style templates: short / llama / qwen3 (ChatML-style).
  • Simple ROUGE-L(F1) metric implemented via LCS.
  • Grid search over (#mem tokens, LM loss weight) and milestone eval.
  • Per-role artifacts and JSONL outputs.

Expected data (RoleBench):
  • profiles-eng/desc.json                       — role descriptions
  • instructions-eng/role-specific-<Role>.jsonl — training pairs: {"instruction","answer"}
  • rolebench-eng/instruction-generalization/role_specific/test.jsonl — test set

CLI notes:
  • --model_path / --tokenizer_path: HF paths.
  • --ae_vector_path: a pre-computed vector used in Phase 1 (AE term).
  • --prompt_style: {short, llama, qwen3}.
  • --hparam_search + search_* flags to enable grid search & epoch milestones.

This file only removes Chinese comments and adds short English ones; core logic is unchanged.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

# ============================ Prompt style support ============================
SHORT_PROMPT_TEMPLATE = "\nQuestion:\n{instruction}\nAnswer:\n"

# Llama 3 (chat format fragments)
LLAMA3_USER_BLOCK = "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
LLAMA3_ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_RESPONSE_SUFFIX = "<|eot_id|>"

# Qwen3 (ChatML-like format)
QWEN3_USER_BLOCK = "<|im_start|>user\n{instruction}<|im_end|>\n"
QWEN3_ASSISTANT_PREFIX = "<|im_start|>assistant\n"
QWEN3_RESPONSE_SUFFIX = "<|im_end|>"


def format_prompt(instruction: str, style: str = "short") -> str:
    """Build a prompt according to the chosen style."""
    if style == "short":
        return SHORT_PROMPT_TEMPLATE.format(instruction=instruction)
    elif style == "llama":
        # Up to assistant prefix; does not include the answer body
        return f"{LLAMA3_USER_BLOCK.format(instruction=instruction)}{LLAMA3_ASSISTANT_PREFIX}"
    elif style == "qwen3":
        # ChatML: one user turn + assistant prefix
        return f"{QWEN3_USER_BLOCK.format(instruction=instruction)}{QWEN3_ASSISTANT_PREFIX}"
    else:
        raise ValueError(f"Unknown prompt_style: {style}")


def format_response(answer: str, style: str = "short") -> str:
    """Wrap a response according to the chosen style."""
    if style == "short":
        return answer
    elif style == "llama":
        # Llama 3 requires <|eot_id|> at the end
        return f"{answer}{LLAMA3_RESPONSE_SUFFIX}"
    elif style == "qwen3":
        # Qwen ChatML uses <|im_end|> terminator
        return f"{answer}{QWEN3_RESPONSE_SUFFIX}"
    else:
        raise ValueError(f"Unknown prompt_style: {style}")


# ================================ Memory Cell ================================
class MemoryCell(torch.nn.Module):
    """
    A wrapper that prepends learnable memory embeddings to the base model inputs.
    The base model is frozen; only memory parameters are optimized.
    """
    def __init__(self, base_model, num_mem_tokens, memory_dim):
        super().__init__()
        self.model = base_model
        self.memory_dim = memory_dim
        self.num_mem_tokens = num_mem_tokens

        # Freeze base model parameters
        for _, p in self.model.named_parameters():
            p.requires_grad = False

        self.create_memory()

    def create_memory(self):
        """Initialize memory params using input embedding std for scaling."""
        embeddings = self.model.get_input_embeddings()
        device = self.model.device
        dtype = self.model.dtype
        memory_params = torch.randn(
            (self.num_mem_tokens, self.memory_dim),
            device=device,
            dtype=dtype
        ) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_params, requires_grad=True))

    def set_memory(self, input_shape):
        """Repeat memory to batch size. input_shape: (batch, seq_len)."""
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids=None, memory_state=None, **kwargs):
        """Forward pass through base model with memory prepended."""
        if memory_state is None:
            if input_ids is None:
                raise ValueError("Either input_ids or memory_state must be provided.")
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, **kwargs)
        out = self.model(**seg_kwargs)
        return out, memory_state

    def generate(self, inputs_embeds, memory_state, attention_mask, **generate_kwargs):
        """Generation path using inputs_embeds + memory."""
        if memory_state is None:
            raise ValueError("Memory state must be provided for generation.")
        full_inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)
        full_attention_mask = self.pad_attention_mask(attention_mask, full_inputs_embeds.shape)
        out = self.model.generate(
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_attention_mask,
            **generate_kwargs
        )
        return out

    def process_input(self, input_ids, memory_state, **kwargs):
        """Build kwargs for base model call where inputs_embeds = [MEM || embeds]."""
        mem_kwargs = dict(**kwargs)
        inputs_embeds = kwargs.get('inputs_embeds')

        if inputs_embeds is None:
            if input_ids is not None:
                inputs_embeds = self.model.get_input_embeddings()(input_ids)
                inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)
            else:
                inputs_embeds = memory_state
        else:
            inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)

        mem_kwargs['input_ids'] = None
        mem_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            mem_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape)
        else:
            mem_kwargs['attention_mask'] = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device)
        return mem_kwargs

    def pad_attention_mask(self, attention_mask, shape):
        """Pad attention mask to account for memory tokens on the left."""
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mem_mask = torch.ones(shape[0], self.num_mem_tokens, dtype=torch.long, device=attention_mask.device)
            return torch.cat([mem_mask, attention_mask], dim=1)


# ===================== Utilities: paths & system prompt ======================
ROLE_SYSTEM_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n You are {role_name}, your description is: {role_description_and_catchphrases}. "
    "Now please answer some questions to accurately show your personality traits! "
    "Your speaking style should fully imitate the personality role assigned to you! "
    "Please do not expose that you are an artificial intelligence model or a language model, "
    "you must always remember that you are only assigned one personality role. "
    "Don't be verbose or too formal or polite when speaking.<|eot_id|>"
)

def safe_read_json(fp: Path) -> Any:
    with fp.open('r', encoding='utf-8') as f:
        return json.load(f)

def read_desc_and_build_system_prompt(desc_json_path: Path, role_name: str) -> Tuple[str, str]:
    """Fetch role description and render a role-specific system prompt."""
    desc = safe_read_json(desc_json_path)
    if role_name not in desc:
        keys_ci = {k.lower(): k for k in desc.keys()}
        if role_name.lower() in keys_ci:
            key = keys_ci[role_name.lower()]
        else:
            key = list(desc.keys())[0]
            print(f"[WARN] Role '{role_name}' not found in desc.json. Falling back to '{key}'.")
        role_name = key
    role_desc = desc[role_name]
    system_prompt = ROLE_SYSTEM_TEMPLATE.format(
        role_name=role_name,
        role_description_and_catchphrases=role_desc
    )
    return system_prompt, role_name

def guess_role_file(instructions_dir: Path, role_name: str) -> Optional[Path]:
    """Find role-specific JSONL file with multiple name patterns; fallback to fuzzy match."""
    candidates = [
        instructions_dir / f"role-specific-{role_name}.jsonl",
        instructions_dir / f"role-specific-{role_name.replace(' ', '_')}.jsonl",
        instructions_dir / f"role-specific-{role_name.replace(' ', '-')}.jsonl",
        instructions_dir / f"role-specific-{re.sub(r'[^A-Za-z0-9_-]+','', role_name)}.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    for p in instructions_dir.glob("role-specific-*.jsonl"):
        if role_name.lower().replace(' ', '') in p.stem.lower().replace(' ', ''):
            return p
    return None


# ================================ Dataset ===================================
class RoleSpecificDataset(Dataset):
    """
    RoleBench format:
      data/RoleBench/instructions-eng/role-specific-<Role>.jsonl
      each line: {"instruction": "...", "answer": "..."}
    """
    def __init__(self, data: List[Dict[str, Any]], tokenizer, prompt_style: str = "short"):
        self.data = data
        self.tok = tokenizer
        self.prompt_style = prompt_style

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]
        instr = r.get("instruction", "")
        ans = r.get("answer", "")

        # Build style-formatted prompt/response
        prompt_text = format_prompt(instr, self.prompt_style)
        response_text = format_response(ans, self.prompt_style)

        prompt_ids = self.tok.encode(prompt_text, add_special_tokens=False)
        response_ids = self.tok.encode(response_text, add_special_tokens=False)
        full_ids = prompt_ids + response_ids + [self.tok.eos_token_id]

        # Mask out prompt tokens from loss
        labels = [-100] * len(prompt_ids) + response_ids + [self.tok.eos_token_id]

        return {
            "full_input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_text": prompt_text
        }

def collate_single(batch):
    return batch[0]

def load_jsonl(fp: Path) -> List[Dict[str, Any]]:
    data = []
    with fp.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def ensure_text(x) -> str:
    """Coerce various types into a string."""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(str(e) for e in x)
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x) if x is not None else ""

def normalize_text(s) -> str:
    """Lowercase, trim, and collapse whitespace for ROUGE tokenization."""
    s = ensure_text(s)
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s


# ============================== ROUGE-L (F1) ================================
def _lcs_length(a_tokens: List[str], b_tokens: List[str]) -> int:
    n, m = len(a_tokens), len(b_tokens)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        ai = a_tokens[i]
        row = dp[i]
        row_next = dp[i+1]
        for j in range(m):
            if ai == b_tokens[j]:
                row_next[j+1] = row[j] + 1
            else:
                row_next[j+1] = max(row_next[j], row[j+1])
    return dp[n][m]

def rouge_l_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    lcs = _lcs_length(pred_toks, gold_toks)
    prec = lcs / max(1, len(pred_toks))
    rec  = lcs / max(1, len(gold_toks))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec + 1e-12)


# ============================== Experiment utils ============================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def param_tag(role: str, n_mem: int, lm_w: float, iters: int, epochs: int, ae_path: str) -> str:
    """Build a unique tag for artifact naming."""
    ae_name = Path(ae_path).stem
    role_s = re.sub(r'[^A-Za-z0-9_-]+', '_', role)
    tag = f"role={role_s}__N={n_mem}__lmw={str(lm_w).replace('.','_')}__iters={iters}__epochs={epochs}__ae={ae_name}"
    return tag

def save_phase_losses_plot(losses: List[float], title: str, out_path: Path, xlabel: str = "Iteration"):
    """Save a simple loss curve plot."""
    plt.figure()
    plt.plot(losses, label=title)
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def collect_all_roles_from_test_jsonl(fp: Path) -> List[str]:
    """Extract distinct roles from test JSONL while preserving order."""
    roles = []
    with fp.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            r = obj.get("role") or obj.get("role_name") or obj.get("character")
            if r:
                roles.append(str(r).strip())
    seen = set()
    uniq = []
    for r in roles:
        key = r.lower().strip()
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


# =========================== Single-role pipeline ===========================
def run_single_role(  # adds eval_epochs and early stopping
    args,
    model,
    tokenizer,
    role_name: str,
    N_mem_tokens: int,
    lm_loss_weight: float,
    initial_max_iterations: int,
    finetune_epochs: int,
    ae_vector_path: str,
    outputs_root: Path,
    results_jsonl_path: Path,
    role_outputs_jsonl_path: Path,
    eval_epochs: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Train/eval for one role; returns final record for this role.
    If eval_epochs is set, will run Phase 3 at those milestones.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = getattr(torch, args.dtype)

    # System prompt
    desc_json_path = Path(args.desc_json)
    system_prompt, role_resolved = read_desc_and_build_system_prompt(desc_json_path, role_name)
    print(f"\n🎭 Role => {role_resolved}")
    print(f"🧾 SYSTEM_PROMPT (first 200 chars): {system_prompt[:200]}...")

    # AE vector (used in Phase 1)
    print(f"⏳ Loading AE vector: {ae_vector_path}")
    ae_vector = torch.load(ae_vector_path, map_location=device).to(dtype)
    if ae_vector.dim() == 1:
        ae_vector = ae_vector.unsqueeze(0)
    ae_vector_batch = ae_vector.unsqueeze(0).to(dtype)  # (1, 1, D)

    # Wrap base model with memory cell
    config = model.config
    memory_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size'))
    model_with_memory = MemoryCell(
        base_model=model,
        num_mem_tokens=N_mem_tokens,
        memory_dim=memory_dim
    ).to(device)

    # ================================ Phase 1 =================================
    print("\n" + "="*80)
    print(" PHASE 1: Initial Memory Training on Role SYSTEM_PROMPT (AE + LM Loss) with Early Stopping")
    print("="*80 + "\n")

    opt = AdamW(model_with_memory.parameters(), lr=args.initial_lr)
    label_ids = tokenizer(system_prompt, return_tensors='pt').input_ids.to(device)

    # Teacher logits for KL term
    with torch.no_grad(), torch.amp.autocast(device_type='cuda' if device=='cuda' else 'cpu', dtype=dtype):
        teacher_inputs = tokenizer(system_prompt, return_tensors='pt').to(device)
        teacher_outputs = model(**teacher_inputs)
        teacher_logits = teacher_outputs.logits[:, -1, :].clone()

    ce_loss = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

    ae_losses, lm_losses, total_losses = [], [], []

    # Early stopping state
    patience = args.early_stopping_patience
    best_loss = float('inf')
    best_state = None
    best_iter = -1
    no_improve = 0

    pbar1 = tqdm(range(initial_max_iterations), desc=f"[{role_resolved}] Phase 1")
    for it in pbar1:
        with torch.amp.autocast(device_type='cuda' if device=='cuda' else 'cpu', dtype=dtype):
            memory_state = model_with_memory.set_memory(label_ids.shape)

            # AE loss on (memory || AE-vector || label-embeds)
            label_embeds = model.get_input_embeddings()(label_ids)
            full_ae_embeds = torch.cat([memory_state, ae_vector_batch, label_embeds], dim=1)
            ae_outputs = model(inputs_embeds=full_ae_embeds)
            logits_for_ae = ae_outputs.logits[:, N_mem_tokens:N_mem_tokens + label_ids.shape[1], :]
            targets_for_ae = label_ids
            loss_ae = ce_loss(logits_for_ae.reshape(-1, logits_for_ae.size(-1)), targets_for_ae.reshape(-1))

            # LM distillation on memory-only forward
            lm_outputs, _ = model_with_memory(memory_state=memory_state)
            student_lm_logits = lm_outputs.logits[:, -1, :]
            loss_lm = kl_loss(F.log_softmax(student_lm_logits, dim=-1), F.softmax(teacher_logits, dim=-1))

            loss_total = (1 - lm_loss_weight) * loss_ae + lm_loss_weight * loss_lm

        loss_total.backward()
        opt.step()
        opt.zero_grad()

        lt = float(loss_total.detach().cpu())
        ae_losses.append(float(loss_ae.detach().cpu()))
        lm_losses.append(float(loss_lm.detach().cpu()))
        total_losses.append(lt)

        # Early stopping bookkeeping
        if lt < best_loss - 1e-8:
            best_loss = lt
            best_state = model_with_memory.memory.data.detach().clone()
            best_iter = it
            no_improve = 0
        else:
            no_improve += 1

        pbar1.set_postfix({"loss": lt, "AE": ae_losses[-1], "LM": lm_losses[-1], "best": best_loss, "pat": no_improve})

        if patience is not None and patience > 0 and no_improve >= patience:
            print(f"⏹️ Early stopping at iter={it}, best_iter={best_iter}, best_loss={best_loss:.6f}")
            break
    pbar1.close()

    # Roll back memory to best checkpoint
    if best_state is not None:
        model_with_memory.memory.data.copy_(best_state)
    print("✅ Phase 1 complete. Best loss:", best_loss, " at iter:", best_iter)

    # Save Phase 1 curves
    tag = param_tag(role_resolved, N_mem_tokens, lm_loss_weight, initial_max_iterations, finetune_epochs, ae_vector_path)
    out_dir = outputs_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(ae_losses, label="AE Loss")
    plt.plot(lm_losses, label="LM Loss")
    plt.plot(total_losses, label="Total Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Phase 1 Losses ({tag})")
    plt.legend()
    phase1_plot = out_dir / f"phase1_losses_{tag}.png"
    plt.savefig(phase1_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved Phase 1 loss curve to: {phase1_plot}")

    # ================================ Phase 2 =================================
    print("\n" + "="*80)
    print(" PHASE 2: Role-Specific Fine-tuning on RoleBench instructions-eng (with mid-epoch eval)")
    print("="*80 + "\n")

    instr_dir = Path(args.instructions_dir)
    role_file = guess_role_file(instr_dir, role_resolved)
    if role_file is None:
        raise FileNotFoundError(f"Could not find role-specific instructions file for role '{role_resolved}' in {instr_dir}")

    role_train = load_jsonl(role_file)
    if len(role_train) == 0:
        raise ValueError(f"No data loaded from {role_file}")

    role_dataset = RoleSpecificDataset(role_train, tokenizer, prompt_style=args.prompt_style)
    role_loader = DataLoader(role_dataset, batch_size=1, shuffle=True, collate_fn=collate_single)

    opt = AdamW(model_with_memory.parameters(), lr=args.finetune_lr)
    ft_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Milestones for evaluation (sanitize and clamp to [1, finetune_epochs])
    if eval_epochs is None:
        eval_epochs = [finetune_epochs]
    eval_epochs = sorted(set([e for e in eval_epochs if 1 <= e <= finetune_epochs]))

    model_with_memory.train()
    phase2_losses_all_iters: List[float] = []
    for epoch in range(finetune_epochs):
        print(f"--- Epoch {epoch + 1}/{finetune_epochs} ---")
        pbar2 = tqdm(role_loader, desc=f"[{role_resolved}] Phase 2")
        total_loss_epoch = 0.0
        for i, batch in enumerate(pbar2):
            input_ids = batch['full_input_ids'].unsqueeze(0).to(device)
            labels = batch['labels'].unsqueeze(0).to(device)

            with torch.amp.autocast(device_type='cuda' if device=='cuda' else 'cpu', dtype=dtype):
                memory_state = model_with_memory.set_memory(input_ids.shape)
                outputs, _ = model_with_memory(input_ids=input_ids, memory_state=memory_state)
                logits = outputs.logits  # (1, M+L, V)

                # Align to predict each label token (shift by one)
                logits_for_loss = logits[:, N_mem_tokens-1:-1, :].contiguous()
                labels_for_loss = labels.contiguous()
                loss = ft_loss_fn(logits_for_loss.view(-1, logits.size(-1)), labels_for_loss.view(-1))

            loss.backward()
            opt.step()
            opt.zero_grad()

            loss_val = float(loss.detach().cpu())
            total_loss_epoch += loss_val
            phase2_losses_all_iters.append(loss_val)
            pbar2.set_postfix({"loss": loss_val, "avg_loss": total_loss_epoch / (i + 1)})
        print(f"Epoch {epoch+1} avg loss: {total_loss_epoch / max(1, i+1):.6f}")

        # Mid-epoch evaluation
        if (epoch + 1) in eval_epochs:
            print(f"🧪 Evaluating at epoch {epoch+1} ...")
            rec = evaluate_role(
                args, tokenizer, model, model_with_memory, role_resolved,
                N_mem_tokens, lm_loss_weight, initial_max_iterations,
                finetune_epochs, ae_vector_path, out_dir, results_jsonl_path,
                phase1_plot, phase2_losses_all_iters, eval_epoch=(epoch+1),
                role_outputs_jsonl_path=role_outputs_jsonl_path
            )
            last_rec = rec  # keep last milestone

    print("✅ Phase 2 complete.")

    # Save Phase 2 curve
    phase2_plot = out_dir / f"phase2_losses_{tag}.png"
    save_phase_losses_plot(phase2_losses_all_iters, f"Phase 2 Losses ({tag})", phase2_plot)
    print(f"📈 Saved Phase 2 loss curve to: {phase2_plot}")

    # Save final memory tensor
    mem_out_path = out_dir / 'system_prompt_memory_role.pt'
    torch.save(model_with_memory.memory.data.clone(), mem_out_path)
    print(f"💾 Saved final [MEM] vector to: {mem_out_path}")

    # Return last record if available, else a placeholder
    return last_rec if 'last_rec' in locals() else {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "role_name": role_resolved,
        "N_mem_tokens": N_mem_tokens,
        "lm_loss_weight": lm_loss_weight,
        "initial_max_iterations": initial_max_iterations,
        "finetune_epochs": finetune_epochs,
        "ae_vector_path": str(ae_vector_path),
        "rouge_l": None,
        "num_samples": 0,
        "sum_rouge_l": 0.0,
        "artifacts": {
            "phase1_plot": str(phase1_plot),
        },
        "eval_epoch": None,
        "tag": tag,
    }


# =========================== Phase 3: evaluation ============================
def evaluate_role(
    args, tokenizer, base_model, model_with_memory, role_resolved,
    N_mem_tokens, lm_loss_weight, initial_max_iterations, finetune_epochs,
    ae_vector_path, out_dir: Path, results_jsonl_path: Path,
    phase1_plot: Path, phase2_losses_all_iters: List[float], eval_epoch: int,
    role_outputs_jsonl_path: Path,
) -> Dict[str, Any]:
    """Run inference on test JSONL for the given role and compute ROUGE-L(F1)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = getattr(torch, args.dtype)

    print("\n" + "="*80)
    print(f" PHASE 3: Inference & Evaluation on RoleBench test  (epoch={eval_epoch})")
    print("="*80 + "\n")

    # Load and filter test samples for this role
    all_test = load_jsonl(Path(args.test_jsonl))
    test_samples = []
    for r in all_test:
        r_role = r.get("role") or r.get("role_name") or r.get("character") or ""
        if str(r_role).strip().lower() == role_resolved.strip().lower():
            item = r.copy()
            item["question"] = ensure_text(item.get("question", ""))
            item["generated"] = ensure_text(item.get("generated", ""))
            test_samples.append(item)

    tag = param_tag(role_resolved, N_mem_tokens, lm_loss_weight, initial_max_iterations, finetune_epochs, ae_vector_path)

    if len(test_samples) == 0:
        print(f"[WARN] No test samples found for role '{role_resolved}'.")
        out_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "role_name": role_resolved,
            "N_mem_tokens": N_mem_tokens,
            "lm_loss_weight": lm_loss_weight,
            "initial_max_iterations": initial_max_iterations,
            "finetune_epochs": finetune_epochs,
            "ae_vector_path": str(ae_vector_path),
            "rouge_l": None,
            "num_samples": 0,
            "sum_rouge_l": 0.0,
            "artifacts": {
                "phase1_plot": str(phase1_plot),
                "phase2_plot": str(out_dir / f"phase2_losses_{tag}.png"),
                "memory_path": str(out_dir / 'system_prompt_memory_role.pt')
            },
            "eval_epoch": eval_epoch,
            "tag": tag,
        }
        with results_jsonl_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
        return out_record

    # Ensure per-sample output path exists
    role_outputs_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    model_with_memory.eval()
    final_memory_tensor = model_with_memory.memory.data.clone().unsqueeze(0)

    rouge_list = []

    for idx, item in enumerate(test_samples):
        question = ensure_text(item.get("question", ""))
        ref = ensure_text(item.get("generated", ""))

        # Use the same prompt-style as training
        prompt_text = format_prompt(question, args.prompt_style)

        # Inference path uses inputs_embeds + memory
        prompt_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(device)
        prompt_embeds = base_model.get_input_embeddings()(prompt_ids)
        attention_mask = torch.ones_like(prompt_ids)

        with torch.no_grad(), torch.amp.autocast(device_type='cuda' if device=='cuda' else 'cpu', dtype=dtype):
            gen_ids = model_with_memory.generate(
                inputs_embeds=prompt_embeds,
                memory_state=final_memory_tensor,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        score = rouge_l_f1(pred, ref)
        rouge_list.append(score)

        # Write per-sample output (role, question, model answer)
        out_line = {
            "role": role_resolved,
            "question": question,
            "model_answer": pred
        }
        with role_outputs_jsonl_path.open('a', encoding='utf-8') as wf:
            wf.write(json.dumps(out_line, ensure_ascii=False) + "\n")

        # Console preview
        print("-" * 80)
        print(f"[{idx+1}/{len(test_samples)}] Role: {role_resolved}")
        print(f"Q: {question}")
        print(f"PRED: {pred}")
        print(f"REF : {ref}")
        print(f"ROUGE-L(F1)={score:.3f}")

    mean_rouge = float(np.mean(rouge_list))
    print("\n" + "="*80)
    print(f"FINAL METRIC (epoch={eval_epoch}) for role '{role_resolved}': ROUGE-L(F1)={mean_rouge:.4f}")
    print("="*80 + "\n")

    out_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "role_name": role_resolved,
        "N_mem_tokens": N_mem_tokens,
        "lm_loss_weight": lm_loss_weight,
        "initial_max_iterations": initial_max_iterations,
        "finetune_epochs": finetune_epochs,
        "ae_vector_path": str(ae_vector_path),
        "rouge_l": mean_rouge,
        "num_samples": len(rouge_list),
        "sum_rouge_l": float(np.sum(rouge_list)),
        "artifacts": {
            "phase1_plot": str(phase1_plot),
            "phase2_plot": str(out_dir / f"phase2_losses_{param_tag(role_resolved, N_mem_tokens, lm_loss_weight, initial_max_iterations, finetune_epochs, ae_vector_path)}.png"),
            "memory_path": str(out_dir / 'system_prompt_memory_role.pt')
        },
        "eval_epoch": eval_epoch,
        "tag": tag,
    }
    with results_jsonl_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
    return out_record


# ======================= Driver: hparam search & summary =====================
def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in re.split(r'[,\s]+', s.strip()) if x != '']

def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in re.split(r'[,\s]+', s.strip()) if x != '']

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Auto-train/fine-tune [MEM] with ROUGE-L metric, early stopping, and hyperparameter search."
    )
    # Model / Tokenizer / AE
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--tokenizer_path', type=str, default=None, help="If None, use model_path.")
    parser.add_argument('--ae_vector_path', type=str, default="./results/ae_token_tuning2/ae_vector.pt")

    # RoleBench paths
    parser.add_argument('--desc_json', type=str, default="data/RoleBench/profiles-eng/desc.json")
    parser.add_argument('--instructions_dir', type=str, default="data/RoleBench/instructions-eng")
    parser.add_argument('--test_jsonl', type=str, default="data/RoleBench/rolebench-eng/instruction-generalization/role_specific/test.jsonl")

    # Training & losses
    parser.add_argument('--N_mem_tokens', type=int)
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--initial_lr', type=float, default=1e-2)
    parser.add_argument('--initial_max_iterations', type=int)
    parser.add_argument('--lm_loss_weight', type=float, default=0.5)
    parser.add_argument('--finetune_epochs', type=int)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument('--early_stopping_patience', type=int)

    # Output
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--results_jsonl', type=str)

    # Per-sample generation outputs (JSONL)
    parser.add_argument('--role_specific_outputs_jsonl', type=str, default=None)

    parser.add_argument('--seed', type=int, default=42)

    # Grid search & milestone eval
    parser.add_argument('--hparam_search', action='store_true', help='Enable grid search over N_mem_tokens and lm_loss_weight.')
    parser.add_argument('--search_mem_tokens', type=str)
    parser.add_argument('--search_lm_loss_weights', type=str)
    parser.add_argument('--search_eval_epochs', type=str, help='Evaluate at these epoch milestones during Phase 2.')

    # Run only first n roles (deterministic order)
    parser.add_argument('--num_roles', type=int, default=0, help='If >0, only run the first n roles (deterministic order).')

    # Prompt formatting
    parser.add_argument(
        '--prompt_style',
        type=str,
        default='short',
        choices=['short', 'llama', 'qwen3'],
        help="Prompt formatting: 'short' (old) or 'llama' (Llama-3 chat format) or 'qwen3' (Qwen ChatML)."
    )

    return parser.parse_args()

def main():
    args = parse_arguments()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = getattr(torch, args.dtype)

    # Load tokenizer/model
    tok_path = args.tokenizer_path or args.model_path
    print(f"⏳ Loading tokenizer from {tok_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"⏳ Loading base model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        use_flash_attention_2=True if dtype != torch.float32 else False
    ).to(device)
    model.eval()

    outputs_root = Path(args.output_dir)
    results_jsonl_path = Path(args.results_jsonl)
    results_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-sample output path (default under output_dir)
    if args.role_specific_outputs_jsonl is not None:
        role_outputs_jsonl_path = Path(args.role_specific_outputs_jsonl)
    else:
        role_outputs_jsonl_path = outputs_root / 'role_specific_outputs.jsonl'
    role_outputs_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect roles (preserve order from test.jsonl)
    test_path = Path(args.test_jsonl)
    all_roles = collect_all_roles_from_test_jsonl(test_path)
    if args.num_roles and args.num_roles > 0:
        all_roles = all_roles[:args.num_roles]
    print(f"🚀 Selected {len(all_roles)} roles.")

    # Global aggregation over ROUGE-L
    total_samples = 0
    sum_rouge = 0.0
    per_role_means = []  # [(role, rouge_l, num_samples), ...]

    # Grid search settings
    if args.hparam_search:
        mem_list = _parse_int_list(args.search_mem_tokens)
        lm_list = _parse_float_list(args.search_lm_loss_weights)
        eval_epochs = _parse_int_list(args.search_eval_epochs)
        max_epoch = max(eval_epochs) if len(eval_epochs) > 0 else args.finetune_epochs
        print(f"🔎 HParam search: N_mem_tokens={mem_list}, lm_loss_weight={lm_list}, eval_epochs={eval_epochs} (train up to {max_epoch})")
    else:
        mem_list = [args.N_mem_tokens]
        lm_list = [args.lm_loss_weight]
        eval_epochs = [args.finetune_epochs]
        max_epoch = args.finetune_epochs

    # Iterate over hparams and roles
    for N_mem_tokens in mem_list:
        for lm_w in lm_list:
            for role_name in all_roles:
                print("\n" + "#" * 100)
                print(f" Start Role => {role_name} | N={N_mem_tokens} | lm_w={lm_w}")
                print("#" * 100 + "\n")
                try:
                    rec = run_single_role(
                        args=argparse.Namespace(**{**vars(args), "lm_loss_weight": lm_w, "N_mem_tokens": N_mem_tokens, "finetune_epochs": max_epoch}),
                        model=model,
                        tokenizer=tokenizer,
                        role_name=role_name,
                        N_mem_tokens=N_mem_tokens,
                        lm_loss_weight=lm_w,
                        initial_max_iterations=args.initial_max_iterations,
                        finetune_epochs=max_epoch,
                        ae_vector_path=args.ae_vector_path,
                        outputs_root=outputs_root,
                        results_jsonl_path=results_jsonl_path,
                        role_outputs_jsonl_path=role_outputs_jsonl_path,
                        eval_epochs=eval_epochs
                    )
                    # Update micro/macro only when we had valid evaluation
                    if rec.get("num_samples", 0) > 0 and rec.get("rouge_l") is not None:
                        total_samples += rec["num_samples"]
                        sum_rouge += rec["sum_rouge_l"]
                        per_role_means.append((
                            rec["role_name"],
                            float(rec["rouge_l"]),
                            int(rec["num_samples"])
                        ))
                except Exception as e:
                    err_record = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "role_name": role_name,
                        "N_mem_tokens": N_mem_tokens,
                        "lm_loss_weight": lm_w,
                        "error": repr(e)
                    }
                    with results_jsonl_path.open('a', encoding='utf-8') as f:
                        f.write(json.dumps(err_record, ensure_ascii=False) + "\n")
                    print(f"[ERROR] Role '{role_name}' failed: {e}")

    # Overall summary (friendly console output; authoritative results are in JSONL)
    if total_samples > 0:
        micro_rouge = sum_rouge / total_samples
    else:
        micro_rouge = float('nan')

    if len(per_role_means) > 0:
        macro_rouge = float(np.mean([x[1] for x in per_role_means]))
    else:
        macro_rouge = float('nan')

    print("\n" + "="*80)
    print("OVERALL RESULTS (ROUGE-L F1)")
    print("="*80)
    print(f"Total evaluated role-means: {len(per_role_means)}")
    print(f"Total evaluated samples    : {total_samples}")
    print(f"Micro ROUGE-L(F1)          : {micro_rouge:.4f}")
    print(f"Macro ROUGE-L(F1)          : {macro_rouge:.4f}")
    print("="*80 + "\n")

    summary_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "num_roles_evaluated": len(per_role_means),
            "num_samples": total_samples,
            "micro": {"ROUGE-L(F1)": micro_rouge},
            "macro": {"ROUGE-L(F1)": macro_rouge}
        }
    }
    with results_jsonl_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

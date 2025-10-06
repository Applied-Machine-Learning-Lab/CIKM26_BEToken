# -*- coding: utf-8 -*-
"""
(NO-AE variant) for RoleLLM:
- Freeze base model; train only [MEM] (memory) embeddings.
- AE-style reconstruction: input is [MEM] + system tokens (teacher forcing).
  Slice logits starting at (N_mem_tokens - 1) and compute CE against system tokens.
- KD: match teacher on the first T assistant tokens (temperature KL).
  Student replaces the system text with [MEM] while teacher sees the real system.

Constraints:
- Do not change loss signs, weights, or early-stopping logic.
- Do not change data path semantics or evaluation protocol.

Notes:
- “NO-AE” here means we reconstruct the system prompt without adding a special AE token;
  prediction for y1 uses logits at index N_mem_tokens - 1 due to memory prepend.
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================ Prompt style support ============================
SHORT_PROMPT_TEMPLATE = "\nQuestion:\n{instruction}\nAnswer:\n"

LLAMA3_USER_BLOCK = "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
LLAMA3_ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_RESPONSE_SUFFIX = "<|eot_id|>"

# ---- Qwen3 (ChatML-style) special tokens ----
QWEN3_USER_BLOCK = "<|im_start|>user\n{instruction}<|im_end|>\n"
QWEN3_ASSISTANT_PREFIX = "<|im_start|>assistant\n"
QWEN3_RESPONSE_SUFFIX = "<|im_end|>"

def format_prompt(instruction: str, style: str = "short") -> str:
    if style == "short":
        return SHORT_PROMPT_TEMPLATE.format(instruction=instruction)
    elif style == "llama":
        # user + assistant prefix (without answer body)
        return f"{LLAMA3_USER_BLOCK.format(instruction=instruction)}{LLAMA3_ASSISTANT_PREFIX}"
    elif style == "qwen3":
        return f"{QWEN3_USER_BLOCK.format(instruction=instruction)}{QWEN3_ASSISTANT_PREFIX}"
    else:
        raise ValueError(f"Unknown prompt_style: {style}")

def format_response(answer: str, style: str = "short") -> str:
    if style == "short":
        return answer
    elif style == "llama":
        return f"{answer}{LLAMA3_RESPONSE_SUFFIX}"
    elif style == "qwen3":
        return f"{answer}{QWEN3_RESPONSE_SUFFIX}"
    else:
        raise ValueError(f"Unknown prompt_style: {style}")

# ================================ MemoryCell =================================
class MemoryCell(torch.nn.Module):
    """
    Wraps a causal LM and prepends a learnable [MEM] token block to the inputs.
    Base model params are frozen; only memory embeddings are trained.
    """
    def __init__(self, base_model, num_mem_tokens, memory_dim):
        super().__init__()
        self.model = base_model
        self.memory_dim = memory_dim
        self.num_mem_tokens = num_mem_tokens
        for _, p in self.model.named_parameters():
            p.requires_grad = False
        self.create_memory()

    def create_memory(self):
        embeddings = self.model.get_input_embeddings()
        device = self.model.device
        dtype = self.model.dtype
        memory_params = torch.randn(
            (self.num_mem_tokens, self.memory_dim),
            device=device, dtype=dtype
        ) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_params, requires_grad=True))

    def set_memory(self, input_shape):
        return self.memory.repeat(input_shape[0], 1, 1)

    def forward(self, input_ids=None, memory_state=None, **kwargs):
        if memory_state is None:
            if input_ids is None:
                raise ValueError("Either input_ids or memory_state must be provided.")
            memory_state = self.set_memory(input_ids.shape)
        seg_kwargs = self.process_input(input_ids, memory_state, **kwargs)
        out = self.model(**seg_kwargs)
        return out, memory_state

    def generate(self, inputs_embeds, memory_state, attention_mask, **generate_kwargs):
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
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        mem_mask = torch.ones(shape[0], self.num_mem_tokens, dtype=torch.long, device=attention_mask.device)
        return torch.cat([mem_mask, attention_mask], dim=1)

# ====================== System prompt builder (RoleBench) =====================
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
class RoleSpecificDataset(Dataset):
    """
    RoleBench format:
      data/RoleBench/instructions-eng/role-specific-<Role>.jsonl
    Each line: {"instruction": "...", "answer": "..."}
    """
    def __init__(self, data: List[Dict[str, Any]], tokenizer, prompt_style: str = "short"):
        self.data = data
        self.tok = tokenizer
        self.prompt_style = prompt_style

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]
        instr = r.get("instruction", "")
        ans = r.get("answer", "")
        prompt_text = format_prompt(instr, self.prompt_style)
        response_text = format_response(ans, self.prompt_style)
        prompt_ids = self.tok.encode(prompt_text, add_special_tokens=False)
        response_ids = self.tok.encode(response_text, add_special_tokens=False)
        full_ids = prompt_ids + response_ids + [self.tok.eos_token_id]
        labels = [-100] * len(prompt_ids) + response_ids + [self.tok.eos_token_id]
        return {
            "full_input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_text": prompt_text,
            "response_text": response_text,        # used by KD
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "response_ids": torch.tensor(response_ids, dtype=torch.long),
        }

def collate_single(batch): return batch[0]

def load_jsonl(fp: Path) -> List[Dict[str, Any]]:
    data = []
    with fp.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: data.append(json.loads(line))
    return data

def ensure_text(x) -> str:
    if isinstance(x, str): return x
    if isinstance(x, list): return " ".join(str(e) for e in x)
    if isinstance(x, dict): return json.dumps(x, ensure_ascii=False)
    return str(x) if x is not None else ""

# =================================== Utils ===================================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def param_tag(role: str, n_mem: int, lm_w: float, iters: int, epochs: int, ae_path: str) -> str:
    """Keep original filename compatibility; 'noae' can be a placeholder for ae_path."""
    ae_name = "noae" if (ae_path is None or str(ae_path).lower() in {"", "none", "noae"}) else Path(ae_path).stem
    role_s = re.sub(r'[^A-Za-z0-9_-]+', '_', role)
    return f"role={role_s}__N={n_mem}__lmw={str(lm_w).replace('.','_')}__iters={iters}__epochs={epochs}__ae={ae_name}"

def save_curve(vals: List[float], title: str, out_path: Path, xlabel: str = "Iteration"):
    plt.figure()
    plt.plot(vals, label=title)
    plt.xlabel(xlabel); plt.ylabel("Loss"); plt.title(title)
    plt.legend(); out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()

def collect_all_roles_from_test_jsonl(fp: Path) -> List[str]:
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

# ========================== KD helper (matches original) ======================
def kd_loss_on_assistant_prefix(
    tokenizer,
    teacher_model,                # base model (with system)
    student_with_mem: MemoryCell, # memory wrapper (no system)
    system_prompt: str,
    prompt_text: str,             # user + assistant BOS (from format_prompt)
    response_text: str,           # ground-truth answer (can be empty)
    N_mem_tokens: int,
    kd_T_tokens: int = 32,
    kd_temperature: float = 2.0,
    use_teacher_gen: bool = False,   # True: always use teacher generation; False: use dataset answer when available
) -> Tuple[torch.Tensor, int]:
    device = next(student_with_mem.parameters()).device

    # Encode system and prompt
    sys_ids = tokenizer(system_prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(device)

    # Target sequence for KD
    if use_teacher_gen:
        with torch.no_grad():
            teacher_prefix = torch.cat([sys_ids[0], prompt_ids[0]], dim=0).unsqueeze(0)
            gen_ids = teacher_model.generate(
                input_ids=teacher_prefix,
                max_new_tokens=kd_T_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )[0]
            ans_ids_full = gen_ids[teacher_prefix.shape[1]:]
    else:
        if response_text and response_text.strip():
            ans_ids_full = tokenizer(response_text, add_special_tokens=False, return_tensors='pt').input_ids.to(device)[0]
        else:
            return torch.tensor(0.0, device=device), 0

    Tprime = int(min(kd_T_tokens, int(ans_ids_full.numel())))
    if Tprime <= 0:
        return torch.tensor(0.0, device=device), 0

    # Teacher-forcing input: system + prompt + answer[:T'-1]
    teacher_input_ids = torch.cat([sys_ids[0], prompt_ids[0], ans_ids_full[:max(0, Tprime-1)]], dim=0).unsqueeze(0)
    teacher_outputs = teacher_model(input_ids=teacher_input_ids)

    prefix_len_teacher = sys_ids.shape[1] + prompt_ids.shape[1]
    teacher_logits_slice = teacher_outputs.logits[:, prefix_len_teacher-1 : prefix_len_teacher-1+Tprime, :]

    # Student-forcing input: prompt + answer[:T'-1]; [MEM] is prepended by wrapper
    student_input_ids = torch.cat([prompt_ids[0], ans_ids_full[:max(0, Tprime-1)]], dim=0).unsqueeze(0)
    student_outputs, _ = student_with_mem(input_ids=student_input_ids)
    prefix_len_student = N_mem_tokens + prompt_ids.shape[1]
    student_logits_slice = student_outputs.logits[:, prefix_len_student-1 : prefix_len_student-1+Tprime, :]

    # Temperature-scaled KL (teacher q, student p)
    T = kd_temperature
    log_p = F.log_softmax(student_logits_slice / T, dim=-1)
    with torch.no_grad():
        q = F.softmax(teacher_logits_slice / T, dim=-1)
    kd = -(q * log_p).sum(dim=-1).mean() * (T * T)
    return kd, Tprime

# ======================= Phase-3: Evaluation (no ROUGE) =======================
def evaluate_role(
    args, tokenizer, base_model, model_with_memory, role_resolved,
    N_mem_tokens, lm_loss_weight, initial_max_iterations, finetune_epochs,
    ae_vector_path, out_dir: Path, results_jsonl_path: Path,
    phase1_plot: Path, phase2_losses_all_iters: List[float], eval_epoch: int,
    role_outputs_jsonl_path: Path,
) -> Dict[str, Any]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = getattr(torch, args.dtype)

    print("\n" + "="*80)
    print(f" PHASE 3: Inference & Save Outputs on RoleBench test  (epoch={eval_epoch})")
    print("="*80 + "\n")

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
            "num_samples": 0,
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

    role_outputs_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    model_with_memory.eval()
    final_memory_tensor = model_with_memory.memory.data.clone().unsqueeze(0)

    for idx, item in enumerate(test_samples):
        question = ensure_text(item.get("question", ""))

        prompt_text = format_prompt(question, args.prompt_style)
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

        out_line = {"role": role_resolved, "question": question, "model_answer": pred}
        with role_outputs_jsonl_path.open('a', encoding='utf-8') as wf:
            wf.write(json.dumps(out_line, ensure_ascii=False) + "\n")

        print("-"*80)
        print(f"[{idx+1}/{len(test_samples)}] Role: {role_resolved}")
        print(f"Q: {question}")
        print(f"PRED: {pred}")

    print("\n" + "="*80)
    print(f"FINAL: Saved {len(test_samples)} model outputs for role '{role_resolved}'.")
    print("="*80 + "\n")

    out_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "role_name": role_resolved,
        "N_mem_tokens": N_mem_tokens,
        "lm_loss_weight": lm_loss_weight,
        "initial_max_iterations": initial_max_iterations,
        "finetune_epochs": finetune_epochs,
        "ae_vector_path": str(ae_vector_path),
        "num_samples": len(test_samples),
        "artifacts": {
                "phase1_plot": str(phase1_plot),
                "phase2_plot": str(out_dir / f"phase2_losses_{param_tag(role_resolved, N_mem_tokens, lm_loss_weight, initial_max_iterations, finetune_epochs, ae_vector_path)}.png"),
                "memory_path": str(out_dir / 'system_prompt_memory_role.pt')
        },
        "eval_epoch": eval_epoch,
        "tag": param_tag(role_resolved, N_mem_tokens, lm_loss_weight, initial_max_iterations, finetune_epochs, ae_vector_path),
    }
    with results_jsonl_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
    return out_record

# ============ Single role: Phase-1 (NO-AE: system reconstruction + KD) =========
def run_single_role(
    args,
    model,
    tokenizer,
    role_name: str,
    N_mem_tokens: int,
    lm_loss_weight: float,
    initial_max_iterations: int,
    outputs_root: Path,
    results_jsonl_path: Path,
    role_outputs_jsonl_path: Path,
) -> Dict[str, Any]:
    """
    Phase-1: AE-style reconstruction (without an AE token) + KD with early stopping.
    Save curves and [MEM], then run Phase-3 inference (save outputs, no ROUGE).
    """
    assert N_mem_tokens >= 1, "NO-AE variant requires N_mem_tokens >= 1"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = getattr(torch, args.dtype)

    # ---------- Build system prompt ----------
    desc_json_path = Path(args.desc_json)
    system_prompt, role_resolved = read_desc_and_build_system_prompt(desc_json_path, role_name)
    print(f"\n🎭 Role => {role_resolved}")
    print(f"🧾 SYSTEM_PROMPT (first 200 chars): {system_prompt[:200]}...")

    # ---------- Memory wrapper ----------
    config = model.config
    memory_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size'))
    model_with_memory = MemoryCell(base_model=model, num_mem_tokens=N_mem_tokens, memory_dim=memory_dim).to(device)

    # ---------- KD training data (RoleBench instructions used for KD only) ----------
    instr_dir = Path(args.instructions_dir)
    role_file = guess_role_file(instr_dir, role_resolved)
    if role_file is None:
        raise FileNotFoundError(f"Could not find role-specific instructions file for role '{role_resolved}' in {instr_dir}")
    role_train = load_jsonl(role_file)
    if len(role_train) == 0:
        raise ValueError(f"No data loaded from {role_file}")
    role_dataset = RoleSpecificDataset(role_train, tokenizer, prompt_style=args.prompt_style)
    role_loader = DataLoader(role_dataset, batch_size=1, shuffle=True, collate_fn=collate_single)

    # ============================ PHASE 1 ============================
    print("\n" + "="*80)
    print(" PHASE 1 (NO-AE): [MEM] reconstruct system + KD (assistant prefix)")
    print("="*80 + "\n")

    opt = AdamW(model_with_memory.parameters(), lr=args.initial_lr)
    ce_loss = torch.nn.CrossEntropyLoss()
    ae_losses, kd_losses, total_losses = [], [], []

    # Teacher system text is only used for AE; KD is driven by instruction samples
    label_ids = tokenizer(system_prompt, return_tensors='pt').input_ids.to(device)

    # Early stopping state
    patience = args.early_stopping_patience
    best_loss = float('inf'); best_state = None; best_iter = -1; no_improve = 0

    # Each iter: sample 1 KD batch and perform 1 AE step
    loader_iter = iter(role_loader)
    pbar1 = tqdm(range(initial_max_iterations), desc=f"[{role_resolved}] Phase 1 (NO-AE)")
    for it in pbar1:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(role_loader)
            batch = next(loader_iter)

        with torch.amp.autocast(device_type='cuda' if device=='cuda' else 'cpu', dtype=dtype):
            # ----- AE loss (NO-AE): reconstruct system ([MEM] + system_tokens) -----
            memory_state = model_with_memory.set_memory(label_ids.shape)        # (1, N, D)
            label_embeds = model.get_input_embeddings()(label_ids)             # (1, L, D)
            full_ae_embeds = torch.cat([memory_state, label_embeds], dim=1)    # (1, N+L, D)
            ae_outputs = model(inputs_embeds=full_ae_embeds)

            # Important: in NO-AE, y1 uses logits at index = N_mem_tokens - 1
            start = N_mem_tokens - 1
            end = start + label_ids.shape[1]
            logits_for_ae = ae_outputs.logits[:, start:end, :]                 # (1, L, V)
            loss_ae = ce_loss(logits_for_ae.reshape(-1, logits_for_ae.size(-1)), label_ids.reshape(-1))

            # ----- KD loss: first T tokens after assistant prefix -----
            kd_loss, used_T = kd_loss_on_assistant_prefix(
                tokenizer=tokenizer,
                teacher_model=model,
                student_with_mem=model_with_memory,
                system_prompt=system_prompt,
                prompt_text=batch["prompt_text"],
                response_text=batch["response_text"],    # ground-truth answer
                N_mem_tokens=N_mem_tokens,
                kd_T_tokens=args.kd_T_tokens,
                kd_temperature=args.kd_temperature,
                use_teacher_gen=args.kd_use_teacher_gen,
            )

            # Combine losses
            if used_T > 0:
                loss_total = (1 - lm_loss_weight) * loss_ae + lm_loss_weight * kd_loss
            else:
                loss_total = loss_ae

        loss_total.backward()
        opt.step(); opt.zero_grad()

        lt = float(loss_total.detach().cpu())
        la = float(loss_ae.detach().cpu())
        lk = float(kd_loss.detach().cpu()) if isinstance(kd_loss, torch.Tensor) else 0.0

        ae_losses.append(la); kd_losses.append(lk); total_losses.append(lt)
        if lt < best_loss - 1e-8:
            best_loss = lt; best_state = model_with_memory.memory.data.detach().clone()
            best_iter = it; no_improve = 0
        else:
            no_improve += 1
        pbar1.set_postfix({"total": lt, "AE(noae)": la, "KD": lk, "best": best_loss, "pat": no_improve})

        if patience and patience > 0 and no_improve >= patience:
            print(f"⏹️ Early stopping at iter={it}, best_iter={best_iter}, best_loss={best_loss:.6f}")
            break
    pbar1.close()

    # Restore best memory
    if best_state is not None:
        model_with_memory.memory.data.copy_(best_state)
    print("✅ Phase 1 (NO-AE) complete. Best total loss:", best_loss, " at iter:", best_iter)

    # Save curves and [MEM]
    tag = param_tag(role_resolved, N_mem_tokens, lm_loss_weight, initial_max_iterations, 0, "noae")
    out_dir = outputs_root / tag; out_dir.mkdir(parents=True, exist_ok=True)

    phase1_plot = out_dir / f"phase1_losses_{tag}.png"
    plt.figure()
    plt.plot(ae_losses, label="AE Loss (NO-AE)")
    plt.plot(kd_losses, label="KD Loss")
    plt.plot(total_losses, label="Total Loss")
    plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.title(f"Phase 1 (NO-AE) Losses ({tag})"); plt.legend()
    plt.savefig(phase1_plot, dpi=150, bbox_inches='tight'); plt.close()
    print(f"📈 Saved Phase 1 loss curve to: {phase1_plot}")

    mem_out_path = out_dir / 'system_prompt_memory_role.pt'
    torch.save(model_with_memory.memory.data.clone(), mem_out_path)
    print(f"💾 Saved final [MEM] vector to: {mem_out_path}")

    # Immediate evaluation (epoch=0 marker): save model outputs only, no ROUGE
    rec = evaluate_role(
        args, tokenizer, model, model_with_memory, role_resolved,
        N_mem_tokens, lm_loss_weight, initial_max_iterations,
        0, "noae", out_dir, results_jsonl_path,
        phase1_plot, [], eval_epoch=0,
        role_outputs_jsonl_path=role_outputs_jsonl_path
    )
    return rec

# ================================== Main ====================================
def parse_arguments():
    p = argparse.ArgumentParser(description="Phase-1 (NO-AE): [MEM] reconstruction + KD (assistant prefix) and evaluation (no ROUGE)")
    # Model / Tokenizer
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--tokenizer_path', type=str, default=None)

    # RoleBench paths
    p.add_argument('--desc_json', type=str, default="data/RoleBench/profiles-eng/desc.json")
    p.add_argument('--instructions_dir', type=str, default="data/RoleBench/instructions-eng")
    p.add_argument('--test_jsonl', type=str, default="data/RoleBench/rolebench-eng/instruction-generalization/role_specific/test.jsonl")

    # Training
    p.add_argument('--N_mem_tokens', type=int, required=True)
    p.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'float16', 'bfloat16'])
    p.add_argument('--initial_lr', type=float, default=1e-2)
    p.add_argument('--initial_max_iterations', type=int, required=True)
    p.add_argument('--lm_loss_weight', type=float, default=0.5)
    p.add_argument('--early_stopping_patience', type=int, default=50)

    # KD specifics
    p.add_argument('--kd_T_tokens', type=int, default=32, help='Number of assistant prefix tokens T to align')
    p.add_argument('--kd_temperature', type=float, default=2.0, help='KD temperature τ')
    p.add_argument('--kd_use_teacher_gen', action='store_true',
                   help='If set, ignore dataset answer and always use teacher-generated first T tokens as the forced sequence for KD. '
                        'If not set, use dataset answer; if empty, skip KD.')

    # Output
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--results_jsonl', type=str, required=True)
    p.add_argument('--role_specific_outputs_jsonl', type=str, default=None)

    # Others
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_roles', type=int, default=0, help='If >0, only run the first n roles (by test.jsonl order)')
    p.add_argument('--prompt_style', type=str, default='short', choices=['short', 'llama', 'qwen3'])
    return p.parse_args()

def main():
    args = parse_arguments()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = getattr(torch, args.dtype)

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
    results_jsonl_path = Path(args.results_jsonl); results_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    role_outputs_jsonl_path = Path(args.role_specific_outputs_jsonl) if args.role_specific_outputs_jsonl \
        else outputs_root / 'role_specific_outputs.jsonl'
    role_outputs_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Role list
    test_path = Path(args.test_jsonl)
    all_roles = collect_all_roles_from_test_jsonl(test_path)
    if args.num_roles and args.num_roles > 0:
        all_roles = all_roles[:args.num_roles]
    print(f"🚀 Selected {len(all_roles)} roles.")

    total_samples = 0
    per_role_counts = []

    for role_name in all_roles:
        print("\n" + "#"*100)
        print(f" Start Role => {role_name} | N={args.N_mem_tokens} | lm_w={args.lm_loss_weight}")
        print("#"*100 + "\n")
        try:
            rec = run_single_role(
                args=args,
                model=model,
                tokenizer=tokenizer,
                role_name=role_name,
                N_mem_tokens=args.N_mem_tokens,
                lm_loss_weight=args.lm_loss_weight,
                initial_max_iterations=args.initial_max_iterations,
                outputs_root=outputs_root,
                results_jsonl_path=results_jsonl_path,
                role_outputs_jsonl_path=role_outputs_jsonl_path,
            )
            if rec.get("num_samples", 0) > 0:
                total_samples += rec["num_samples"]
                per_role_counts.append((rec["role_name"], int(rec["num_samples"])))
        except Exception as e:
            err_record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "role_name": role_name,
                "N_mem_tokens": args.N_mem_tokens,
                "lm_loss_weight": args.lm_loss_weight,
                "error": repr(e)
            }
            with results_jsonl_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(err_record, ensure_ascii=False) + "\n")
            print(f"[ERROR] Role '{role_name}' failed: {e}")

    print("\n" + "="*80)
    print("OVERALL RESULTS (no ROUGE)")
    print("="*80)
    print(f"Total evaluated roles  : {len(per_role_counts)}")
    print(f"Total evaluated samples: {total_samples}")
    print("="*80 + "\n")

    summary_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "num_roles_evaluated": len(per_role_counts),
            "num_samples": total_samples,
        }
    }
    with results_jsonl_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

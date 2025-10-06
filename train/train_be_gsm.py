import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import random
import itertools

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
# ============================ Few-shot (作为“system 文本”) ============================
FEWSHOT_SYSTEM_TEXT = (
    "<|im_start|>user\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|im_end|>\n<|im_start|>assistant\n"
    "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6"
    "<|im_end|>\n<|im_start|>user\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|im_end|>\n<|im_start|>assistant\n"
    "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5"
    "<|im_end|>\n<|im_start|>user\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|im_end|>\n<|im_start|>assistant\n"
    "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39"
    "<|im_end|>\n<|im_start|>user\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|im_end|>\n<|im_start|>assistant\n"
    "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8"
    "<|im_end|>\n<|im_start|>user\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|im_end|>\n<|im_start|>assistant\n"
    "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9"
    "<|im_end|>\n<|im_start|>user\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|im_end|>\n<|im_start|>assistant\n"
    "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29"
    "<|im_end|>\n<|im_start|>user\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|im_end|>\n<|im_start|>assistant\n"
    "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33"
    "<|im_end|>\n<|im_start|>user\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|im_end|>\n<|im_start|>assistant\n"
    "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8"
    "<|im_end|>"
)

# =============== 问题 Block（要求格式）: user block + assistant 开头 =================
QUESTION_BLOCK_TEMPLATE = (
    "<|im_start|>user\n"
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: {instruction}\n"
    "Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
    "<|im_end|>\n<|im_start|>assistant\n"
)
def build_question_block(instruction: str) -> str:
    return QUESTION_BLOCK_TEMPLATE.format(instruction=instruction.strip())

# ================================ MemoryCell =================================
class MemoryCell(torch.nn.Module):
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
        std = embeddings.weight.data.std()
        memory_params = torch.randn((self.num_mem_tokens, self.memory_dim), device=device, dtype=dtype) * std
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

# ================================ IO / Utils =================================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_text(x) -> str:
    if isinstance(x, str): return x
    if isinstance(x, list): return " ".join(str(e) for e in x)
    if isinstance(x, dict): return json.dumps(x, ensure_ascii=False)
    return str(x) if x is not None else ""

def normalize_text(s) -> str:
    s = ensure_text(s).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def save_curve(vals: List[float], labels: List[str], title: str, out_path: Path, xlabel: str = "Iteration"):
    plt.figure()
    for v, lab in zip(vals, labels):
        plt.plot(v, label=lab)
    plt.xlabel(xlabel); plt.ylabel("Loss"); plt.title(title)
    plt.legend(); out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()

def tag_from_grid(lr: float, kdT: int, lmw: float, use_tgen: bool, n_mem: int, iters: int):
    tgen = "tgenT" if use_tgen else "gtT"
    return f"N{n_mem}_lr{lr}_T{kdT}_lmw{str(lmw).replace('.','_')}_{tgen}_iters{iters}"

# =============================== 数据集（Train） ==============================
class GSMKDTrainDataset(Dataset):
    """
    从 MathInstruct.json 中筛选 source == "data/CoT/gsm_train.json"
    每条样本：提供 prompt_text（问题 block），以及 response_text（ground-truth answer；可为空）
    """
    def __init__(self, math_instruct_path: Path):
        self.samples = []
        with math_instruct_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if item.get("source") == "data/CoT/gsm_train.json":
                instr = ensure_text(item.get("instruction", ""))
                out = ensure_text(item.get("output", "")).strip()
                # Ground-truth answer文本可直接作为 teacher 强制序列；也可以后续改为 teacher 生成
                self.samples.append({
                    "prompt_text": build_question_block(instr),  # user+assistantBOS
                    "response_text": out
                })
        if len(self.samples) == 0:
            raise ValueError("No training samples found for source='data/CoT/gsm_train.json'")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
def collate_single(batch): return batch[0]

# ================================ KD 核心（修改） ==============================
def kd_loss_on_assistant_prefix(
    tokenizer,
    teacher_model,                # base model (冻结)
    student_with_mem: MemoryCell, # Memory wrapper（仅 [MEM] 可训练）
    system_prompt_text: str,      # few-shot 文本
    prompt_text: str,             # user + assistant_BOS（问题 block）
    response_text: str,           # 训练数据中的原始 answer 文本（可空）
    N_mem_tokens: int,
    kd_T_tokens: int = 32,
    kd_temperature: float = 2.0,
    kd_use_teacher_gen_if_empty: bool = True,
    use_teacher_gen: bool = False,   # 新增：True=强制使用 teacher 生成的 T token；False=优先用原始 answer
) -> Tuple[torch.Tensor, int]:
    """
    返回 (KD loss, 实际使用 token 数 T')
    Teacher 输入： system_text + prompt_text + (answer[:T'-1])
    Student 输入：          prompt_text + (answer[:T'-1])，由 [MEM] 预置
    """
    device = next(student_with_mem.parameters()).device

    sys_ids = tokenizer(system_prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(device)

    # 目标序列：三种情况
    ans_ids_full = torch.tensor([], dtype=torch.long, device=device)
    if use_teacher_gen:
        # 强制使用 teacher 生成
        with torch.no_grad():
            teacher_prefix = torch.cat([sys_ids[0], prompt_ids[0]], dim=0).unsqueeze(0)
            gen_ids = teacher_model.generate(
                input_ids=teacher_prefix,
                max_new_tokens=kd_T_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )[0]
            ans_ids_full = gen_ids[teacher_prefix.shape[1]:]  # 新增部分
    else:
        # 尝试使用数据集 ground-truth answer
        if response_text and response_text.strip():
            ans_ids_full = tokenizer(response_text, add_special_tokens=False, return_tensors='pt').input_ids.to(device)[0]
        elif kd_use_teacher_gen_if_empty:
            with torch.no_grad():
                teacher_prefix = torch.cat([sys_ids[0], prompt_ids[0]], dim=0).unsqueeze(0)
                gen_ids = teacher_model.generate(
                    input_ids=teacher_prefix,
                    max_new_tokens=kd_T_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )[0]
                ans_ids_full = gen_ids[teacher_prefix.shape[1]:]

    Tprime = int(min(kd_T_tokens, int(ans_ids_full.numel())))
    if Tprime <= 0:
        return torch.tensor(0.0, device=device), 0

    # Teacher-Forcing：system + prompt + answer[:T'-1]
    teacher_input_ids = torch.cat([sys_ids[0], prompt_ids[0], ans_ids_full[:max(0, Tprime-1)]], dim=0).unsqueeze(0)
    teacher_outputs = teacher_model(input_ids=teacher_input_ids)
    prefix_len_teacher = sys_ids.shape[1] + prompt_ids.shape[1]
    teacher_logits_slice = teacher_outputs.logits[:, prefix_len_teacher-1 : prefix_len_teacher-1+Tprime, :]

    # Student-Forcing：prompt + answer[:T'-1]，由 [MEM] 预置
    student_input_ids = torch.cat([prompt_ids[0], ans_ids_full[:max(0, Tprime-1)]], dim=0).unsqueeze(0)
    student_outputs, _ = student_with_mem(input_ids=student_input_ids)
    prefix_len_student = N_mem_tokens + prompt_ids.shape[1]
    student_logits_slice = student_outputs.logits[:, prefix_len_student-1 : prefix_len_student-1+Tprime, :]

    T = kd_temperature
    log_p = F.log_softmax(student_logits_slice / T, dim=-1)
    with torch.no_grad():
        q = F.softmax(teacher_logits_slice / T, dim=-1)

    kd = -(q * log_p).sum(dim=-1).mean() * (T * T)
    return kd, Tprime

# ================================ 评测（GSM8K） ===============================
ANS_TRIGGERS = ['The final answer is', 'The answer is:', 'The answer is', 'the answer is', '####']

def extract_pred_answer(text: str) -> str:
    s = text.strip()
    # 优先从 "The final answer is xxx" / "The answer is xxx"
    for trig in ANS_TRIGGERS[:-1]:
        if trig in s:
            tail = s.split(trig, 1)[-1]
            # 找到最后一个数（支持负数/小数）
            m = re.findall(r'[-+]?\d+(\.\d+)?', tail)
            if m:
                # m 是捕获组列表，这里需要用另一个正则抓全串
                m2 = re.findall(r'[-+]?\d+(?:\.\d+)?', tail)
                return m2[-1]
    # 回退：找整段最后一个数字
    m3 = re.findall(r'[-+]?\d+(?:\.\d+)?', s)
    if m3: return m3[-1]
    return ""

def parse_gsm8k_groundtruth(answer_field: str) -> str:
    # gsm8k: "... #### 42" -> 抓 #### 后的最后一个数字
    m = re.search(r'####\s*([-+]?\d+(?:\.\d+)?)', answer_field)
    if m: return m.group(1)
    # 回退：抓最后数字
    m2 = re.findall(r'[-+]?\d+(?:\.\d+)?', answer_field)
    if m2: return m2[-1]
    return ""

def evaluate_gsm8k_accuracy(
    tokenizer, base_model, model_with_memory, memory_tensor,
    test_jsonl_path: Path, max_new_tokens: int = 256, temperature: float = 0.0
) -> Dict[str, Any]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    correct = 0; total = 0
    preds = []

    with test_jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            q = ensure_text(obj.get("question", ""))
            gt = ensure_text(obj.get("answer", ""))

            prompt_text = build_question_block(q)
            prompt_ids = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)
            prompt_embeds = base_model.get_input_embeddings()(prompt_ids)
            attention_mask = torch.ones_like(prompt_ids)

            with torch.no_grad():
                gen_ids = model_with_memory.generate(
                    inputs_embeds=prompt_embeds,
                    memory_state=memory_tensor,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0),
                    top_p=1.0,
                    temperature=temperature if temperature > 0 else 1.0,
                    pad_token_id=tokenizer.eos_token_id
                )
            out_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            pred = extract_pred_answer(out_text)
            gold = parse_gsm8k_groundtruth(gt)
            hit = (pred == gold)
            correct += int(hit); total += 1
            preds.append({"question": q, "pred": pred, "gold": gold, "hit": hit})

    acc = correct / max(1, total)
    return {"accuracy": acc, "num": total, "correct": correct, "details": preds}

# ========================= 训练（单组超参） + 评测 ============================
def train_mem_for_fewshot(
    args,
    model,
    tokenizer,
    system_prompt_text: str,
    train_dataset: GSMKDTrainDataset,
    N_mem_tokens: int,
    lr: float,
    kd_T_tokens: int,
    lm_loss_weight: float,
    use_teacher_gen: bool,
    outputs_root: Path,
) -> Tuple[MemoryCell, torch.Tensor, Dict[str, Any]]:
    """
    执行 Phase-1（AE + KD），仅更新 [MEM]；返回训练好的包裹模型 + 最优 memory 向量 + 训练日志
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = getattr(torch, args.dtype)

    # [AE] 向量
    ae_vector = torch.load(args.ae_vector_path, map_location=device).to(dtype)
    if ae_vector.dim() == 1: ae_vector = ae_vector.unsqueeze(0)   # (1, D)
    ae_vector_batch = ae_vector.unsqueeze(0).to(dtype)            # (1, 1, D)

    # Memory wrapper
    config = model.config
    memory_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size'))
    model_with_memory = MemoryCell(base_model=model, num_mem_tokens=N_mem_tokens, memory_dim=memory_dim).to(device)

    # 数据加载器（随机迭代样本做 KD）
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_single)

    opt = AdamW(model_with_memory.parameters(), lr=lr)
    ce_loss = torch.nn.CrossEntropyLoss()

    # AE 的 label 就是 few-shot system 文本
    label_ids = tokenizer(system_prompt_text, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

    ae_losses, kd_losses, total_losses = [], [], []
    best_loss = float('inf'); best_state = None; best_iter = -1; no_improve = 0

    pbar = tqdm(range(args.initial_max_iterations), desc=f"[Train] lr={lr},T={kd_T_tokens},lmw={lm_loss_weight},tgen={use_teacher_gen}")
    loader_iter = iter(loader)

    for it in pbar:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        with torch.amp.autocast(device_type='cuda' if device=='cuda' else 'cpu', dtype=dtype):
            # ----- AE：重建 few-shot 文本 -----
            memory_state = model_with_memory.set_memory(label_ids.shape)
            label_embeds = model.get_input_embeddings()(label_ids)
            full_ae_embeds = torch.cat([memory_state, ae_vector_batch, label_embeds], dim=1)
            ae_outputs = model(inputs_embeds=full_ae_embeds)
            logits_for_ae = ae_outputs.logits[:, N_mem_tokens:N_mem_tokens + label_ids.shape[1], :]
            loss_ae = ce_loss(logits_for_ae.reshape(-1, logits_for_ae.size(-1)), label_ids.reshape(-1))

            # ----- KD：assistant 起点后的前 T 个位置 -----
            kd_loss, used_T = kd_loss_on_assistant_prefix(
                tokenizer=tokenizer,
                teacher_model=model,
                student_with_mem=model_with_memory,
                system_prompt_text=system_prompt_text,
                prompt_text=batch["prompt_text"],
                response_text=batch["response_text"],
                N_mem_tokens=N_mem_tokens,
                kd_T_tokens=kd_T_tokens,
                kd_temperature=args.kd_temperature,
                kd_use_teacher_gen_if_empty=args.kd_use_teacher_gen,
                use_teacher_gen=use_teacher_gen,  # 新增开关
            )

            # 总损失
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

        pbar.set_postfix({"total": lt, "AE": la, "KD": lk, "best": best_loss, "pat": no_improve})
        if args.early_stopping_patience and no_improve >= args.early_stopping_patience:
            print(f"⏹ Early stop @iter={it}, best_iter={best_iter}, best_loss={best_loss:.6f}")
            break
    pbar.close()

    # 回滚到最佳 memory
    if best_state is not None:
        model_with_memory.memory.data.copy_(best_state)

    # 保存曲线
    tag = tag_from_grid(lr, kd_T_tokens, lm_loss_weight, use_teacher_gen, N_mem_tokens, args.initial_max_iterations)
    out_dir = outputs_root / tag; out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / f"phase1_losses_{tag}.png"
    save_curve([ae_losses, kd_losses, total_losses], ["AE", "KD", "Total"], f"Phase1 Losses ({tag})", curve_path)

    # 保存 memory
    mem_out_path = out_dir / "fewshot_memory.pt"
    torch.save(model_with_memory.memory.data.clone(), mem_out_path)

    log = {
        "best_total_loss": best_loss,
        "best_iter": best_iter,
        "curve_path": str(curve_path),
        "memory_path": str(mem_out_path),
        "tag": tag,
    }
    return model_with_memory, model_with_memory.memory.data.clone().unsqueeze(0), log

# ================================ 主流程（网格） ==============================
def parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_list_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_list_bools(s: str) -> List[bool]:
    out = []
    for x in s.split(","):
        x = x.strip().lower()
        if not x: continue
        if x in ["true", "1", "yes", "y"]: out.append(True)
        elif x in ["false", "0", "no", "n"]: out.append(False)
        else: raise ValueError(f"Bad bool: {x}")
    return out

def main():
    p = argparse.ArgumentParser(description="Compress few-shot prompt into [MEM] for GSM8K (AE+KD) + grid search + accuracy eval")
    # Model / Tokenizer / AE
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--tokenizer_path', type=str, default=None)
    p.add_argument('--ae_vector_path', type=str, required=True)

    # Data
    p.add_argument('--math_instruct_path', type=str, default='./data/MathInstruct.json')
    p.add_argument('--gsm8k_test_path', type=str, default='./data/gsm8k_test.jsonl')

    # Training common
    p.add_argument('--N_mem_tokens', type=int, required=True)
    p.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'float16', 'bfloat16'])
    p.add_argument('--initial_max_iterations', type=int, default=1000)
    p.add_argument('--early_stopping_patience', type=int, default=80)
    p.add_argument('--kd_temperature', type=float, default=2.0)
    p.add_argument('--kd_use_teacher_gen', action='store_true', help='当样本没有 answer 时，允许 teacher 生成 T 个 token')

    # Eval
    p.add_argument('--gen_max_new_tokens', type=int, default=256)
    p.add_argument('--gen_temperature', type=float, default=0.0)

    # Output
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--results_jsonl', type=str, required=True)

    # Grid lists（默认即题主给定）
    p.add_argument('--t_list', type=str, default='32')                 # kd_T_tokens 候选
    p.add_argument('--lr_list', type=str, default='1e-3')      # 学习率候选
    p.add_argument('--lmw_list', type=str, default='0.5')    # lm_loss_weight 候选
    p.add_argument('--use_teacher_gen_list', type=str, default='true') # 是否强制使用 teacher 生成序列

    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

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

    # 训练集
    train_dataset = GSMKDTrainDataset(Path(args.math_instruct_path))

    # 网格
    T_list = parse_list_ints(args.t_list)
    LR_list = parse_list_floats(args.lr_list)
    LMW_list = parse_list_floats(args.lmw_list)
    TGEN_list = parse_list_bools(args.use_teacher_gen_list)

    results_path = Path(args.results_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # 准备 system 文本（few-shot）
    system_text = FEWSHOT_SYSTEM_TEXT

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "grid_sizes": {
            "T": len(T_list),
            "lr": len(LR_list),
            "lmw": len(LMW_list),
            "use_teacher_gen": len(TGEN_list),
        },
        "combos": []
    }

    for kdT, lr, lmw, use_tgen in itertools.product(T_list, LR_list, LMW_list, TGEN_list):
        print("\n" + "#"*100)
        print(f" Grid Combo => T={kdT} | lr={lr} | lm_loss_weight={lmw} | use_teacher_gen={use_tgen}")
        print("#"*100 + "\n")

        out_dir = Path(args.output_dir)
        try:
            model_with_memory, final_memory_tensor, train_log = train_mem_for_fewshot(
                args=args,
                model=model,
                tokenizer=tokenizer,
                system_prompt_text=system_text,
                train_dataset=train_dataset,
                N_mem_tokens=args.N_mem_tokens,
                lr=lr,
                kd_T_tokens=kdT,
                lm_loss_weight=lmw,
                use_teacher_gen=use_tgen,
                outputs_root=out_dir
            )

            # 评测 GSM8K
            eval_res = evaluate_gsm8k_accuracy(
                tokenizer=tokenizer,
                base_model=model,
                model_with_memory=model_with_memory,
                memory_tensor=final_memory_tensor,
                test_jsonl_path=Path(args.gsm8k_test_path),
                max_new_tokens=args.gen_max_new_tokens,
                temperature=args.gen_temperature
            )

            record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "params": {
                    "kd_T_tokens": kdT,
                    "lr": lr,
                    "lm_loss_weight": lmw,
                    "use_teacher_gen": use_tgen,
                    "N_mem_tokens": args.N_mem_tokens,
                    "initial_max_iterations": args.initial_max_iterations,
                    "early_stopping_patience": args.early_stopping_patience,
                    "dtype": args.dtype,
                },
                "train": train_log,
                "eval": {
                    "accuracy": eval_res["accuracy"],
                    "num": eval_res["num"],
                    "correct": eval_res["correct"],
                },
            }
            with results_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            summary["combos"].append({
                "tag": train_log["tag"],
                "accuracy": eval_res["accuracy"]
            })
            print(f"✅ Combo {train_log['tag']} | GSM8K Acc = {eval_res['accuracy']:.4f}")

        except Exception as e:
            err_record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "error": repr(e),
                "params": {
                    "kd_T_tokens": kdT,
                    "lr": lr,
                    "lm_loss_weight": lmw,
                    "use_teacher_gen": use_tgen,
                }
            }
            with results_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(err_record, ensure_ascii=False) + "\n")
            print(f"❌ Failed combo (T={kdT}, lr={lr}, lmw={lmw}, tgen={use_tgen}): {e}")

    with results_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps({"grid_summary": summary}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

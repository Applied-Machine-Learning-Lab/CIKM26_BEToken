# Behavior-Equivalent Token

Official implementation of **“Behavior-Equivalent Token: Single-Token Replacement for Long Prompts in LLMs.”**
We compress a long system prompt **P** into a **single learnable embedding** (the **`[BE]`token**) so that, at inference time, `[BE]` can replace the entire prompt while preserving the model’s behavior.

---

## Quick start

**Environment**

```bash
# Create the conda environment
conda env create -f myenv.yaml

# Install FlashAttention v2
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

---

## Method overview

We train in three stages; model weights remain **frozen** except for the special-token rows we introduce.

* **Stage 0 — Auto-Encoder (AE) pre-training.**
  Learn a *universal trigger* embedding `[AE]` by reconstructing input text. Only the `[AE]` row in the embedding matrix is updated.
* **Stage 1 — BE reconstruction.**
  Given the universal trigger from Stage 0, learn a *prompt-specific* embedding `[BE]` for target prompt **P = (s₁,…,sₘ)** by training the model to reconstruct **P** when conditioned on the sequence `[BE][AE]`.
* **Stage 2 — Behavior distillation (KD).**
  Ensure that replacing **P** with `[BE]` yields the same conditional output distribution on downstream queries.
  The **teacher** is the same LLM conditioned on the full prompt **P**; the **student** is the LLM conditioned on `[BE]`.
  For an unlabeled query *q*, the teacher generates a response *A = (a₁,…,a_T)*. We then train the student to match the teacher’s token-level distributions along that trajectory by **minimizing the KL divergence** between teacher and student logits at each step.

> Stages 1 and 2 are implemented in the `train_be_*.py` scripts listed below.

---

## Special tokens & tokenizer changes

`[AE]` and `[BE]` are **not** part of the base tokenizer:

* **Stage 0:** Add `[AE]` via `additional_special_tokens`, append a row to the embedding matrix, **freeze all other parameters**, and save tokenizer + weights.
* **Stages 1–2:** Add `[BE]` in the same way. During training, **only** the `[BE]` row is updated; all other weights (including LM head) remain frozen. Save tokenizer and model weights for later use.

---

## Data preparation

Place datasets under `data/` (customize via script arguments):

| Dataset                       | Purpose                                                                                                        | Location / Format                                      |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **AE corpus**           | Generic text to train the universal `[AE]` token. Each `.jsonl` has one field `text` per line.           | `data/*.jsonl`                                       |
| **RoleBench / RoleLLM** | Role profiles (`desc.json`), instruction files, and test splits.                                             | `data/RoleBench/`                                    |
| **GSM8K**               | Math word problems for KD and evaluation. Train:`data/MathInstruct.json`; Test: `data/gsm8k_test.jsonl`.   | `data/`                                              |
| **HPD**                 | Role-play dataset used for HPD experiments. Expect a Hugging Face dataset saved via `datasets.save_to_disk`. | e.g.,`data/hpd/train`, `data/hpd/en_test_set.json` |

---

## Stage 0 — AE pre-training

The **auto-encoder stage** learns the universal vector `[AE]` by asking the model to reconstruct arbitrary text when `[AE]` is appended to the input. Only the `[AE]` embedding row is trained; the base model is frozen.

**Script:** `train_ae_token_llama.py` (and `train_ae_token_qwen.py` for Qwen models)

**Example:**

```bash
python train_ae_token_llama.py \
  --model_name_or_path /path/to/base-model \
  --jsonl_path ./data/ae_corpus/merged_corpus.jsonl \
  --ae_token "<|AE|>" \
  --output_dir ./results/ae_token \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-2 \
  --num_train_epochs 2
```

Outputs the trained vector `ae_vector.pt` in `--output_dir` and saves the tokenizer with the added special token.

---

## Stages 1–2 — BE reconstruction + knowledge distillation

Stage 1 learns the **behavior-equivalent** token `[BE]` for a specific long prompt **P**; Stage 2 distills behavior from the teacher (full **P**) into the student (`[BE]`).

Different scripts target different datasets:



### RoleBench / RoleLLM

**Script:** `train_be_role.py`
Compress each **RoleBench** system prompt into `[BE]` and perform KD on role-specific instruction/answer pairs.

**Example:**

```bash
python train_be_role.py \
  --model_path /path/to/base-model \
  --ae_vector_path ./results/ae_token/ae_vector.pt \
  --desc_json ./data/RoleBench/profiles-eng/desc.json \
  --instructions_dir ./data/RoleBench/instructions-eng \
  --test_jsonl ./data/RoleBench/rolebench-eng/instruction-generalization/role_specific/test.jsonl \
  --N_mem_tokens 1 \
  --initial_max_iterations 2500 \
  --lm_loss_weight 0.5 \
  --kd_T_tokens 32 \
  --kd_temperature 2.0 \
  --output_dir ./results/be_role \
  --results_jsonl ./results/be_role/summary.jsonl
```

Writes per-role outputs and a `summary.jsonl`. You can limit roles with `--num_roles` and choose a prompt formatting style (`llama`, `qwen3`).

---

### HPD (Harry-Potter Dialogue)

**Script:** `train_be_hpd.py`
Compress the *n-shot HPD system prompt* into a single `[BE]` token. Requires a base model, tokenizer, pre-trained `ae_vector.pt`, and the HPD training dataset.

Key args: KD temperature (`--kd_temperature`), number of assistant prefix tokens to align (`--kd_T_tokens`), and loss weight between reconstruction and KD (`--lm_loss_weight`).

**Example:**

```bash
python train_be_hpd.py \
  --model_path /path/to/base-model \
  --ae_vector_path ./results/ae_token/ae_vector.pt \
  --hpd_train_ds ./data/hpd/train \
  --tokenizer_path ./results/ae_token \
  --N_mem_tokens 1 \
  --initial_lr 1e-2 \
  --initial_max_iterations 3000 \
  --lm_loss_weight 0.7 \
  --kd_T_tokens 32 \
  --kd_temperature 2.0 \
  --output_dir ./results/be_hpd
```

Trains with early stopping and saves `system_prompt_be_hpd.pt` plus a loss-curve plot. Evaluate PPL with `eval_ppl_hpd.py` (see **Evaluation**).

---

### GSM8K

**Script:** `train_be_gsm.py`
Train a `[BE]` token to replace a long **few-shot chain-of-thought** (CoT) prompt for math word problems. Performs grid search over KD hyper-parameters and reports GSM8K accuracy.

**Minimal example:**

```bash
python train_be_gsm.py \
  --model_path /path/to/base-model \
  --tokenizer_path ./results/ae_token \
  --ae_vector_path ./results/ae_token/ae_vector.pt \
  --math_instruct_path ./data/MathInstruct.json \
  --gsm8k_test_path ./data/gsm8k_test.jsonl \
  --N_mem_tokens 1 \
  --t_list 32 --lr_list 1e-3 --lmw_list 0.9 --use_teacher_gen_list true \
  --output_dir ./results/be_gsm \
  --results_jsonl ./results/be_gsm/summary.jsonl
```

For each hyper-parameter combo, the script trains a BE token, evaluates GSM8K accuracy, and records the results (used to reproduce the paper’s GSM8K table).

---

## Evaluation

Scripts to measure performance of learned `[BE]` tokens and baselines:

* **GSM8K baselines**
  `baseline_eval_llama.py`, `baseline_eval_qwen.py` compare a base model **with** the full few-shot prompt vs. **without** any prompt. Adjust `--gen_max_new_tokens` and `--gen_temperature` for decoding.
  (Prompts follows the settings from the Hugging Face dataset *meta-llama/Llama-3.1-8B-Instruct-evals*.)
* **HPD perplexity**
  `eval_ppl_hpd.py` computes token-level PPL on the HPD test set by prepending `[BE]` to user queries. Provide `--mem_vector_path` (the saved BE vector), base model path, and test JSON; outputs a PPL summary.
* **RoleLLM win-rate**
  `rolellm_*_eval.py` uses an external judge (e.g., GPT-4o via HTTP API). Produces win/loss/tie counts and win rate. Requires judge API credentials; used only for the RoleLLM win-rate metric.Configure `HOST`, `PORT`, `MODEL`, and `API_KEY`. 

---

## Recommended repository layout

```
.
├── README.md                  # This file – overview and instructions
├── data/                      # Data placeholders (not tracked)
│   ├── ae_corpus/             # Generic text corpus for AE pre-training
│   ├── hpd/                   # HPD dataset (saved via datasets.save_to_disk)
│   ├── RoleBench/             # RoleBench descriptions, instructions, test splits
│   ├── MathInstruct.json      # Math data for GSM8K KD
│   └── gsm8k_test.jsonl       # GSM8K test set
├── train/                     # Stage 0/1/2 training scripts
│   ├── train_ae_token_llama.py
│   ├── train_ae_token_qwen.py
│   ├── train_be_gsm.py
│   ├── train_be_hpd.py
│   ├── train_be_role.py
│   ├── train_mem_kd_role.py
│   ├── train_mem_pt_role.py
│   ├── train_none_pt.py
│   └── t_hpd.py
├── eval/                      # Evaluation scripts
│   ├── baseline_gsm8k_eval_llama.py
│   ├── eval_ppl_hpd.py
│   ├── rolellm_fewshot.py
│   └── rolellm_gpt_eval.py
├── results/                   # Saved AE vectors
│   └── ae_token_tuning2/
├── scripts/                  
│   ├── gsm.sh
│   ├── hpd.sh
│   └── rolellm.sh
└── myenv.yaml                 # Conda environment (dependency versions)
```

> **Note:** Datasets are **not** distributed with this repository. Download/prepare them as described above and place them under `data/`.

---

## Citation

If you use this codebase or the BE-Token methodology in your research, please cite our paper (BibTeX to be added once the paper is published).

---

## License

This project is released under the **MIT License**.

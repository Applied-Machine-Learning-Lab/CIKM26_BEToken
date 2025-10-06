#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=""
OUT_DIR="./outputs"
RESULTS_JSONL="${OUT_DIR}/all_roles_results.jsonl"
ROLE_OUTPUTS_JSONL="${OUT_DIR}/role_specific_outputs.jsonl"

nohup python train_be_role.py \
  --model_path "${MODEL_PATH}" \
  --N_mem_tokens 1 \
  --initial_max_iterations 2500 \
  --early_stopping_patience 1000 \
  --lm_loss_weight 0.9 \
  --num_roles 100 \
  --output_dir "${OUT_DIR}-teacherKD" \
  --results_jsonl "${OUT_DIR}-teacherKD/all_roles_results.jsonl" \
  --role_specific_outputs_jsonl "${OUT_DIR}-teacherKD/role_specific_outputs.jsonl" \
  --prompt_style "llama" \
  --kd_T_tokens 32 \
  --kd_temperature 2.0 \
  --kd_use_teacher_gen \
  > train_phase1_kd_2500_teacher.log 2>&1


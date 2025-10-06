#!/usr/bin/env bash
set -euo pipefail


MODEL_PATH=""
TOKENIZER_PATH=""
AE_VECTOR="./results/ae_token_tuning2/ae_vector.pt"
HPD_TRAIN_DS="./"
HPD_TEST_JSON="./en_test_set.json"

OUT_DIR="./outputs-hpd-memkd"
MEM_PATH="${OUT_DIR}/system_prompt_memory_hpd.pt"
PPL_JSON="${OUT_DIR}/ppl_report.json"

python train_mem_phase1_kd_hpd.py \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${MODEL_PATH}" \
  --ae_vector_path "${AE_VECTOR}" \
  --hpd_train_ds "${HPD_TRAIN_DS}" \
  --N_mem_tokens 1 \
  --initial_max_iterations 2500 \
  --early_stopping_patience 500 \
  --lm_loss_weight 0.5 \
  --kd_T_tokens 32 \
  --kd_temperature 2.0 \
  --kd_use_teacher_gen \
  --prompt_style "llama" \
  --output_dir "${OUT_DIR}"

python eval_ppl_hpd.py \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${MODEL_PATH}" \
  --test_json "${HPD_TEST_JSON}" \
  --mem_vector_path "${MEM_PATH}" \
  --N_mem_tokens 1 \
  --dtype bfloat16 \
  --prompt_style "llama" \
  --output_json "${PPL_JSON}"



cd ../gpu_tools/
nohup bash run_gpu.sh &



MODEL_PATH=""
AE_VECTOR_PATH="./results/ae_token_tuning2/ae_vector.pt"
N_MEM_TOKENS=1
MAX_ITERS=2500
EARLY_STOP_PATIENCE=500
DTYPE="bfloat16"
OUTPUT_DIR="./out_mem_gsm8k"

RESULTS_JSONL="${OUTPUT_DIR}/grid_results.jsonl"

TRAIN_LOG="./train_phase1_gsm.log"

mkdir -p ${OUTPUT_DIR}

nohup python train_be_gsm.py --model_path "${MODEL_PATH}"  --ae_vector_path "${AE_VECTOR_PATH}" --N_mem_tokens ${N_MEM_TOKENS}  --initial_max_iterations ${MAX_ITERS} --early_stopping_patience ${EARLY_STOP_PATIENCE} --dtype ${DTYPE} --output_dir "${OUTPUT_DIR}" --results_jsonl "${RESULTS_JSONL}" > ${TRAIN_LOG} 2>&1

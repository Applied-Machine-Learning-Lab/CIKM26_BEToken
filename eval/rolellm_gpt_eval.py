"""
RoleLLM Win-Rate Evaluator (new vs. reference)

This script pairwise-compares a model's answers against a reference/baseline
answer set on RoleLLM-style data using a judge model via an HTTP
/OpenAI-compatible/ /chat/completions endpoint. It computes:
- wins, losses, ties
- win rate = (wins + 0.5 * ties) / total

Quick setup:
1) Point LLM_ANSWERS_PATH to your model outputs (jsonl).
2) Point STANDARD_ANSWERS_PATH to the reference/baseline file (jsonl).
3) Fill HOST, PORT, MODEL, API_KEY for the judge endpoint.
4) Run: python evaluate.py
"""

import http.client
import json
import os
import random
import time
import traceback
import socket

# --- 1) Configuration ---

NUM_SAMPLES = -1  # -1 means evaluate all entries

LLM_ANSWERS_PATH = './3boutputs-kdst052500-1-l/outputs-kdst052500-1-l.jsonl'
# STANDARD_ANSWERS_PATH = './data/RoleBench/rolebench-eng/instruction-generalization/role_specific/test.jsonl'
STANDARD_ANSWERS_PATH = './data/RoleBench/rolebench-eng/instruction-generalization/role_specific/rolegpt_baseline.jsonl'
OUTPUT_DIR = './3boutputs-kdst052500-1-l/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_FILE_PATH = os.path.join(OUTPUT_DIR, 'evaluation_summary.txt')
GPT4O_LOGS_PATH = os.path.join(OUTPUT_DIR, 'gpt4o_mini_evaluations.jsonl')

# --- 2) Direct HTTP API (OpenAI-compatible) ---

HOST = ""   # e.g., "localhost" or "api.your-endpoint.com"
PORT = 8200
MODEL = "gpt-4o"
API_KEY = ""

# Timeouts & retries
HTTP_TIMEOUT = 20         # initial request timeout (seconds)
MAX_TIMEOUT = 120         # cap per-request timeout (seconds)
MAX_RETRIES = 5           # max retries on API failure
RETRY_BACKOFF_BASE = 2.0  # exponential backoff base
RATE_LIMIT_SLEEP = 0.8    # small sleep between successful requests

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
"""

# --- 3) Utilities ---

def load_jsonl(filepath):
    if not os.path.exists(filepath):
        print(f"Error: file not found at {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def create_standard_answers_map(standard_answers_list):
    """
    Build a map from (role, question) -> reference answer (take first if list).
    """
    answers_map = {}
    for item in standard_answers_list:
        role = (item.get("role") or "").strip()
        question = (item.get("question") or "").strip()
        generated = item.get("generated")
        std_answer = None
        if isinstance(generated, list) and len(generated) > 0:
            std_answer = generated[0]
        elif isinstance(generated, str):
            std_answer = generated
        if role and question and std_answer:
            answers_map[(role, question)] = std_answer
    return answers_map

def _parse_retry_after(headers_dict):
    """
    Parse Retry-After seconds if present (ignore date-form variants).
    Returns float seconds or None.
    """
    ra = headers_dict.get('retry-after')
    if not ra:
        return None
    try:
        return float(ra)
    except Exception:
        return None

def _post_chat_completions(payload_dict, timeout):
    """
    Low-level HTTP call: POST /chat/completions and return (status, text, headers_dict).
    Creates a fresh connection per call to avoid sticky connection issues.
    """
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    try:
        payload = json.dumps(payload_dict, ensure_ascii=False).encode('utf-8')
        headers = {
            "Authorization": "Bearer " + API_KEY,
            "Content-Type": "application/json",
            "Connection": "close",
        }
        conn.request("POST", "/chat/completions", body=payload, headers=headers)
        res = conn.getresponse()
        data_bytes = res.read()
        text = data_bytes.decode("utf-8", errors="replace")
        headers_dict = {k.lower(): v for (k, v) in res.getheaders()}
        return res.status, text, headers_dict
    finally:
        try:
            conn.close()
        except Exception:
            pass

def _send_chat_completion(messages, max_tokens=128, temperature=0.0, retries=MAX_RETRIES):
    """
    Higher-level wrapper with growing timeouts, error handling, retries, and light rate control.
    Returns (choices[0].message.content, raw_response_text) or (None, None).
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    last_err_msg = None
    base_timeout = HTTP_TIMEOUT

    for attempt in range(retries + 1):
        # Increase timeout per attempt, capped by MAX_TIMEOUT
        attempt_timeout = min(int(base_timeout * (2 ** attempt)), MAX_TIMEOUT)

        try:
            status, text, hdrs = _post_chat_completions(payload, timeout=attempt_timeout)
        except (TimeoutError, socket.timeout) as te:
            last_err_msg = f"Timeout (attempt={attempt}, timeout={attempt_timeout}s): {te}"
        except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError) as ce:
            last_err_msg = f"Connection interrupted (attempt={attempt}): {type(ce).__name__}: {ce}"
        except Exception as e:
            last_err_msg = f"Request exception (attempt={attempt}): {type(e).__name__}: {e}"
        else:
            # Got an HTTP response
            if status == 200:
                try:
                    resp_json = json.loads(text)
                    choices = resp_json.get("choices") or []
                    if choices and "message" in choices[0] and "content" in choices[0]["message"]:
                        return choices[0]["message"]["content"], text
                    if "output_text" in resp_json:
                        return resp_json["output_text"], text
                    last_err_msg = f"Unexpected JSON shape: {text[:300]}"
                except Exception as je:
                    last_err_msg = f"JSON parse failed: {je}; snippet: {text[:300]}"
            elif status in (429, 503, 504):
                # Rate-limited / overloaded / gateway timeout: wait then retry
                ra = _parse_retry_after(hdrs)
                sleep_s = ra if ra is not None else (RETRY_BACKOFF_BASE ** attempt)
                time.sleep(sleep_s)
                continue
            else:
                last_err_msg = f"HTTP {status}: {text[:300]}"

        # If not successful, backoff if retries remain
        if attempt < retries:
            time.sleep(RETRY_BACKOFF_BASE ** attempt)
        else:
            break

    print(f"API call failed: {last_err_msg}")
    return None, None

def call_gpt4o_mini_evaluator(instruction, response_a, response_b):
    """
    Ask the judge model to pick 'A', 'B', or 'Tie' for the given instruction/responses.
    """
    prompt = PROMPT_TEMPLATE.format(
        instruction=instruction,
        response_a=response_a,
        response_b=response_b
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        content, raw_text = _send_chat_completion(messages, max_tokens=128, temperature=0.0)
    except Exception as e:
        # Belt-and-suspenders: swallow unexpected exceptions and keep going
        print(f"Judge call raised (ignored): {e}")
        content, raw_text = None, None
    return content, raw_text

# --- 4) Main ---

def main():
    print("Starting evaluation...")
    print(f"NUM_SAMPLES: {'ALL' if NUM_SAMPLES == -1 else NUM_SAMPLES}")

    print(f"Loading reference answers from: {STANDARD_ANSWERS_PATH}")
    standard_answers_list = load_jsonl(STANDARD_ANSWERS_PATH)
    if standard_answers_list is None:
        return
    standard_answers_map = create_standard_answers_map(standard_answers_list)
    print(f"Loaded {len(standard_answers_map)} reference Q&A pairs.")

    print(f"Loading model answers from: {LLM_ANSWERS_PATH}")
    llm_answers = load_jsonl(LLM_ANSWERS_PATH)
    if llm_answers is None:
        return

    if NUM_SAMPLES != -1 and NUM_SAMPLES > 0:
        if len(llm_answers) > NUM_SAMPLES:
            print(f"Found {len(llm_answers)} model answers; will evaluate first {NUM_SAMPLES}.")
            llm_answers = llm_answers[:NUM_SAMPLES]
        else:
            print(f"Requested {NUM_SAMPLES}, but only {len(llm_answers)} available. Evaluating all.")

    print(f"Prepared to evaluate {len(llm_answers)} model answers.")

    wins = 0
    ties = 0
    losses = 0
    errors = 0
    total_comparisons = 0
    evaluation_details = []

    with open(GPT4O_LOGS_PATH, 'w', encoding='utf-8') as gpt4o_log_file:
        for i, item in enumerate(llm_answers):
            try:
                role = (item.get("role") or "").strip()
                question = (item.get("question") or "").strip()

                model_answer = item.get("model_answer")
                if model_answer is None:
                    model_answer = item.get("answer")
                if model_answer is None:
                    print(f"Warning: item {i} missing model_answer/answer. Skipping.")
                    errors += 1
                    evaluation_details.append({"index": i, "result": "DATA_ERROR"})
                    continue

                standard_answer = standard_answers_map.get((role, question))
                if standard_answer is None:
                    print(f"Warning: no reference found for role='{role}', question='{question[:50]}...'. Skipping.")
                    errors += 1
                    evaluation_details.append({"index": i, "result": "STD_NOT_FOUND"})
                    continue

                total_comparisons += 1
                print(f"Evaluating {total_comparisons}/{len(llm_answers)} ...")

                # Randomize A/B assignment
                our_model_is_A = random.choice([True, False])
                if our_model_is_A:
                    response_a, response_b = model_answer, standard_answer
                else:
                    response_a, response_b = standard_answer, model_answer

                eval_result, raw_resp_text = call_gpt4o_mini_evaluator(question, response_a, response_b)

                # Light rate limiting
                time.sleep(RATE_LIMIT_SLEEP)

                if eval_result is None:
                    errors += 1
                    result_category = "API_ERROR"
                    parsed_judgment = "API_ERROR"
                else:
                    clean_result = eval_result.strip().upper().replace('"', '').replace("'", "")
                    if clean_result.startswith("A"):
                        parsed_judgment = "A"
                    elif clean_result.startswith("B"):
                        parsed_judgment = "B"
                    elif clean_result.startswith("TIE"):
                        parsed_judgment = "TIE"
                    else:
                        parsed_judgment = "PARSE_ERROR"
                        errors += 1

                    if (parsed_judgment == "A" and our_model_is_A) or \
                       (parsed_judgment == "B" and not our_model_is_A):
                        wins += 1
                        result_category = "win"
                    elif parsed_judgment == "TIE":
                        ties += 1
                        result_category = "tie"
                    elif parsed_judgment in ["A", "B"]:
                        losses += 1
                        result_category = "loss"
                    else:
                        result_category = "error"

                log_entry = {
                    "role": role,
                    "question": question,
                    "model_answer": model_answer,
                    "standard_answer": standard_answer,
                    "our_model_assigned_to": "A" if our_model_is_A else "B",
                    "gpt4o_mini_raw_output": raw_resp_text or "",
                    "parsed_judgment": parsed_judgment,
                    "result_category": result_category
                }
                gpt4o_log_file.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                gpt4o_log_file.flush()

                evaluation_details.append({"index": i, "result": result_category})

            except KeyboardInterrupt:
                print("Interrupted by user.")
                break
            except Exception as loop_e:
                # Skip single-entry errors without stopping the run
                errors += 1
                print(f"Item {i} raised {type(loop_e).__name__}: {loop_e} (skipped)")
                log_entry = {
                    "role": item.get("role"),
                    "question": item.get("question"),
                    "model_answer": item.get("model_answer") or item.get("answer"),
                    "standard_answer": None,
                    "our_model_assigned_to": None,
                    "gpt4o_mini_raw_output": "",
                    "parsed_judgment": "EXCEPTION",
                    "result_category": "error"
                }
                gpt4o_log_file.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                gpt4o_log_file.flush()
                continue

    if total_comparisons > 0:
        win_rate = (wins + 0.5 * ties) / total_comparisons
    else:
        win_rate = 0

    summary_content = (
        f"--- Evaluation Summary ---\n\n"
        f"Requested samples: {'ALL' if NUM_SAMPLES == -1 else NUM_SAMPLES} (actual: {total_comparisons})\n\n"
        f"Total comparisons: {total_comparisons}\n"
        f"Wins: {wins}\n"
        f"Losses: {losses}\n"
        f"Ties: {ties}\n"
        f"Errors/Unparsed: {errors}\n\n"
        f"Final win rate (wins + 0.5 * ties) / total: {win_rate:.4f}\n\n"
        f"--- Per-item results ---\n"
    )

    for detail in evaluation_details:
        summary_content += f"Item {detail['index']}: {detail['result']}\n"

    with open(SUMMARY_FILE_PATH, 'w', encoding='utf-8') as summary_file:
        summary_file.write(summary_content)

    print("\n--- Done ---")
    print(f"Summary written to: {SUMMARY_FILE_PATH}")
    print(f"Judge raw logs written to: {GPT4O_LOGS_PATH}")
    print(f"Final win rate: {win_rate:.4f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print("Unhandled exception:", e)
        traceback.print_exc()

# run once to create the disk dataset used by train_mem_phase1_kd_hpd.py
import json
from datasets import Dataset

with open("./en_train_set.json", "r", encoding="utf-8") as f:
    data = json.load(f)           # { "dialogue-xxx": { ... }, ... }
rows = list(data.values())
ds = Dataset.from_list(rows)
ds.save_to_disk("./train")
print("Saved to disk dataset for KD at: experience/.../train")

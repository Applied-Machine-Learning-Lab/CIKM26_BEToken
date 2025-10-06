#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_mem_phase1_kd_hpd.py

Phase-1: Train a single soft [BE] token (AE + KD) on the HPD dataset.

- Freeze the base model; only learn the prefix vector stored in MemoryCell.memory (the [BE] token).
  A fixed [AE] vector (trained elsewhere) is concatenated to help reconstruct the HPD 6-shot system prompt.
- Knowledge distillation (KD): align the next-token distribution for the first T assistant tokens
  with temperature scaling. Teacher can use either the dataset answer or teacher-generated text.
- Outputs: the learned [BE] vector and loss curves.

This script assumes a causal LM (e.g., LLaMA/Qwen-style chat formatting).
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================ Prompt style support ============================
SHORT_PROMPT_TEMPLATE = "\nQuestion:\n{instruction}\nAnswer:\n"
LLAMA3_SYSTEM_BLOCK = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
LLAMA3_USER_BLOCK = "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
LLAMA3_ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"

QWEN3_SYSTEM_BLOCK = "<|im_start|>system\n{content}<|im_end|>\n"
QWEN3_USER_BLOCK = "<|im_start|>user\n{instruction}<|im_end|>\n"
QWEN3_ASSISTANT_PREFIX = "<|im_start|>assistant\n"

def format_system(content: str, style: str = "llama") -> str:
    if style == "llama":
        return LLAMA3_SYSTEM_BLOCK.format(content=content)
    elif style == "qwen3":
        return QWEN3_SYSTEM_BLOCK.format(content=content)
    elif style == "short":
        return content
    else:
        raise ValueError(f"Unknown prompt_style: {style}")

def format_prompt(instruction: str, style: str = "llama") -> str:
    if style == "short":
        return SHORT_PROMPT_TEMPLATE.format(instruction=instruction)
    elif style == "llama":
        return f"{LLAMA3_USER_BLOCK.format(instruction=instruction)}{LLAMA3_ASSISTANT_PREFIX}"
    elif style == "qwen3":
        return f"{QWEN3_USER_BLOCK.format(instruction=instruction)}{QWEN3_ASSISTANT_PREFIX}"
    else:
        raise ValueError(f"Unknown prompt_style: {style}")

# ================================= MemoryCell ================================
class MemoryCell(torch.nn.Module):
    """
    Wrap a base model with a learnable prefix embedding [BE].
    - Base model parameters are frozen; only `self.memory` (num_mem_tokens x D) is trained (the [BE] token).
    - On forward/generate, we prepend memory_state to the rest of the inputs.
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
        device = embeddings.weight.device
        dtype = embeddings.weight.dtype
        memory_params = torch.randn((self.num_mem_tokens, self.memory_dim), device=device, dtype=dtype) \
                        * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_params, requires_grad=True))

    def set_memory(self, input_shape):
        # Repeat the [BE] prefix for the batch size -> (B, N_mem, D)
        return self.memory.repeat(input_shape[0], 1, 1)

    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        mem_mask = torch.ones(shape[0], self.num_mem_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
        return torch.cat([mem_mask, attention_mask], dim=1)

    def process_input(self, input_ids=None, memory_state=None, **kwargs):
        mem_kwargs = dict(**kwargs)
        if memory_state is None:
            raise ValueError("memory_state is required")

        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            if input_ids is None:
                inputs_embeds = memory_state
            else:
                tok_emb = self.model.get_input_embeddings()(input_ids)
                inputs_embeds = torch.cat([memory_state, tok_emb], dim=1)
        else:
            inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)

        mem_kwargs['input_ids'] = None
        mem_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            mem_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape)
        else:
            mem_kwargs['attention_mask'] = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)
        return mem_kwargs

    def forward(self, input_ids=None, memory_state=None, **kwargs):
        # If memory_state is not provided, create it based on the batch size (for the [BE] prefix).
        if memory_state is None:
            if input_ids is None and kwargs.get('inputs_embeds') is None:
                raise ValueError("Either input_ids/inputs_embeds or memory_state must be provided.")
            if input_ids is not None:
                memory_state = self.set_memory(input_ids.shape)
            else:
                inputs_embeds = kwargs['inputs_embeds']
                fake_ids = torch.empty((inputs_embeds.shape[0], 1), dtype=torch.long, device=inputs_embeds.device)
                memory_state = self.set_memory(fake_ids.shape)

        mem_kwargs = self.process_input(input_ids=input_ids, memory_state=memory_state, **kwargs)
        out = self.model(**mem_kwargs)
        return out, memory_state

    def generate(self, inputs_embeds, memory_state, attention_mask, **generate_kwargs):
        # Prepend [BE] prefix when generating
        full_inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)
        full_attention_mask = self.pad_attention_mask(attention_mask, full_inputs_embeds.shape)
        out = self.model.generate(
            inputs_embeds=full_inputs_embeds,
            attention_mask=full_attention_mask,
            **generate_kwargs
        )
        return out

# ========================== HPD 6-shot System Prompt =========================
def hpd_6shot_system_content() -> str:
    # The templates below are taken as-is from user-provided content.
    tmplate = """To better help you mimic the behavior of Harry Potter, we additionally provide the following background information ofthe dialogue:
1. Dialogue position, which represents the timeline of the dialogue in Happy Potter Novels. For example, "Dialogue Position: Book5-chapter28" means the dialogues occurs in Chapter28,Book5.
2. Dialogue speakers.
3. Harry Potter’s attributes, which refers to basic properties of Harry Potter when the dialogue happens. It can contains 13 categories: Gender, Age, Lineage, Talents, Looks, Achievement, Title, Belongings, Export, Hobby, Character, Spells and Nickname.
4. Speaker relations with Harry, such as whether he was a friend, classmate, or family member;
5. Harry's Familiarity to the speaker, which ranges from 0 to 10. Concretely, 0 denotes stranger, and 10 denotes close friends who often stay together for many years and are very familiar with each other’s habits, secrets and temperaments, where Ron meets this condition in Book 7.
6. Harry’s Affection to the speaker, which ranges from -10 to 10. 1 refers to speaker met Harry for the first time. For instance, when Hary first met Ron and Hermione in Book 1, Harry’s Affection to them are both set to 1. And -10 means the speaker killed Harry’s parents, where Voldemort meets this condition in the novels.


Here is an example:
Dialogue position: Book1-chapter2
Dialogue speakers: Harry, Petunia, Vernon

Harry’s attributes:
"name": "Harry",
"nickname": "The boy who lived",
"gender": "male",
"age": "age 11",
"looks": "Very thin, black hair, emerald green eyes, wearing glasses, knife injury with lightning shape at the forehead",
"hobbies": "None",
"character": "None",
"talents": "None",
"export": "None",
"belongings": "None",
"affiliation": "None",
"lineage": "None",
"title": "The boy who lived",
"spells": "None"

Speakers relations with Harry: Vernon is Harry’s uncle and Petunia is Harry’s aunt.
Harry’s Familiarity to Vernon: 8
Harry’s Affection to Vernon: -4
Harry’s Familiarity to Petunia: 8
Harry’s Affection to Petunia: -4

Dialogue: 
"Petunia: Bad news, Vernon, Mrs. Figg’s broken her leg. She can’t take him. Now what?"
"Vernon: I’m warning you, I’m warning you now, boy — any funny business, anything at all — and you’ll be in that cupboard from now until Christmas." 
Thought: Let’s think step by step. According to the conversation history, Vernon warned Harry not to spoil the special day. According to Harry Potter’s attributes, he is still very thin, does not know any spells, and has not gone to Hogwarts yet.
So he is currently incapable of resisting them. At the same time, based on his affection for them is -4, it means that he relatively doesn’t like them, and may even be a little scared. Therefore, Harry possiblely says: I know, I will obediently obedient, and I won’t cause you trouble.

Harry’s Response: I know, I will obediently obedient, and I won’t cause you trouble.


Keep in mind the following requirements:
1. Before generating the response, you should read the above information and dialogue content carefully.
2. You can not generate the response that is against Harry Potter’s attributes and Harry’s relations with the speaker.
3. Not every component in the background information may be useful, you should choose some of them to help you generate more concise and comprehensive responses that satisfy the behavior of Harry Potter in the dialogue.
4. Not every speaker have relations, familiarity ad affection to Harry. At that time, you can directly predict what would Harry say only based on the dialogue context..
"""
    tmplate1 = """Here is an example:
Dialogue position: Book1-chapter2
Dialogue speakers: Harry, Petunia, Vernon

Harry’s attributes:
"name": "Harry",
"nickname": "The boy who lived",
"gender": "male",
"age": "age 11",
"looks": "Very thin, black hair, emerald green eyes, wearing glasses, knife injury with lightning shape at the forehead",
"hobbies": "None",
"character": "None",
"talents": "None",
"export": "None",
"belongings": "None",
"affiliation": "None",
"lineage": "None",
"title": "The boy who lived",
"spells": ""

Speakers relations with Harry:             
"name": "Dudley",
"friend": 0.0,
"classmate": 0.0,
"teacher": 0.0,
"family": 1.0,
"immediate family": 0.0,
"lover": 0.0,
"opponent": 0.0,
"colleague": 0.0,
"teammate": 0.0,
"enemy": 0.0,
"Harry's affection to him": -4.0,
"Harry's familiarity with him": 8.0,
"His affection to Harry": -4.0,
"His familiarity with Harry": 6.0

"name": "Piers",
"friend": 0.0,
"classmate": 0.0,
"teacher": 0.0,
"family": 0.0,
"immediate family": 0.0,
"lover": 0.0,
"opponent": 0.0,
"colleague": 0.0,
"teammate": 0.0,
"enemy": 0.0,
"Harry's affection to him": -4.0,
"Harry's familiarity with him": 2.0,
"His affection to Harry": -4.0,
"His familiarity with Harry": 2.0

Dialogue: 
"Dudley: Make it move,",
"the snake: I get that all the time.",
"Harry: I know, Where do you come from, anyway? Was it nice there? Oh, I see — so you’ve never been to Brazil?",
"keeper of the reptile house: DUDLEY! MR. DURSLEY! COME AND LOOK AT THIS SNAKE! YOU WON’T BELIEVE WHAT IT’S DOING!",
"Dudley: Out of the way, you,",
"the snake: Brazil, here I come. . . . Thanksss, amigo.",
"Piers: Harry was talking to it, weren’t you, Harry?"
Harry’s Response: I don't know, I don't know anything.Don't target me anymore, Piers.
Keep in mind the following requirements:
1. Before generating the response, you should read the above information and dialogue content carefully.
2. You can not generate the response that is against Harry Potter’s attributes and Harry’s relations with the speaker.
3. Not every component in the background information may be useful, you should choose some of them to help you generate more concise and comprehensive responses that satisfy the behavior of Harry Potter in the dialogue.
4. Not every speaker have relations, familiarity ad affection to Harry. At that time, you can directly predict what would Harry say only based on the dialogue context..
"""
    tmplate2 = """Here is an example:
Dialogue position: Book1-chapter3
Dialogue speakers: Harry, Petunia

Harry’s attributes:
"name": "Harry",
"nickname": "The boy who lived",
"gender": "male",
"age": "age 11",
"looks": "Very thin, black hair, emerald green eyes, wearing glasses, knife injury with lightning shape at the forehead",
"hobbies": "None",
"character": "None",
"talents": "None",
"export": "None",
"belongings": "None",
"affiliation": "None",
"lineage": "None",
"title": "The boy who lived",
"spells": ""

Speakers relations with Harry:             
"name": "Petunia",
"friend": 0.0,
"classmate": 0.0,
"teacher": 0.0,
"family": 1.0,
"immediate family": 0.0,
"lover": 0.0,
"opponent": 0.0,
"colleague": 0.0,
"teammate": 0.0,
"enemy": 0.0,
"Harry's affection to him": -4.0,
"Harry's familiarity with him": 8.0,
"His affection to Harry": -4.0,
"His familiarity with Harry": 6.0

Dialogue: 
"Harry: What’s this?",
"Petunia: Your new school uniform,",
"Harry: Oh, I didn’t realize it had to be so wet.",
"Petunia: Don’t be stupid, I’m dyeing some of Dudley’s old things gray for you. It’ll look just like everyone else’s when I’ve finished."
Harry’s Response: Well, there is no other choice.
Keep in mind the following requirements:
1. Before generating the response, you should read the above information and dialogue content carefully.
2. You can not generate the response that is against Harry Potter’s attributes and Harry’s relations with the speaker.
3. Not every component in the background information may be useful, you should choose some of them to help you generate more concise and comprehensive responses that satisfy the behavior of Harry Potter in the dialogue.
4. Not every speaker have relations, familiarity ad affection to Harry. At that time, you can directly predict what would Harry say only based on the dialogue context..
"""
    tmplate3 = """Here is an example:
Dialogue position: Book1-chapter3
Dialogue speakers: Vernon, Dudley, Harry, Petunia

Harry’s attributes:
"name": "Harry",
"nickname": "The boy who lived",
"gender": "male",
"age": "age 11",
"looks": "Very thin, black hair, emerald green eyes, wearing glasses, knife injury with lightning shape at the forehead",
"hobbies": "None",
"character": "None",
"talents": "None",
"export": "None",
"belongings": "None",
"affiliation": "None",
"lineage": "None",
"title": "The boy who lived",
"spells": ""

Speakers relations with Harry:             
"Petunia": {
    "name": "Petunia",
    "friend": 0.0, "classmate": 0.0, "teacher": 0.0, "family": 1.0, "immediate family": 0.0, "lover": 0.0,
    "opponent": 0.0, "colleague": 0.0, "teammate": 0.0, "enemy": 0.0,
    "Harry's affection to him": -4.0, "Harry's familiarity with him": 8.0,
    "His affection to Harry": -4.0, "His familiarity with Harry": 6.0
},
"Vernon": {
    "name": "Vernon Dursley",
    "friend": 0.0, "classmate": 0.0, "teacher": 0.0, "family": 1.0, "immediate family": 0.0, "lover": 0.0,
    "opponent": 0.0, "colleague": 0.0, "teammate": 0.0, "enemy": 0.0,
    "Harry's affection to him": -4.0, "Harry's familiarity with him": 8.0,
    "His affection to Harry": -4.0, "His familiarity with Harry": 6.0
},
"Dudley": {
    "name": "Dudley",
    "friend": 0.0, "classmate": 0.0, "teacher": 0.0, "family": 1.0, "immediate family": 0.0, "lover": 0.0,
    "opponent": 0.0, "colleague": 0.0, "teammate": 0.0, "enemy": 0.0,
    "Harry's affection to him": -4.0, "Harry's familiarity with him": 8.0,
    "His affection to Harry": -4.0, "His familiarity with Harry": 6.0
}

Dialogue: 
"Vernon: Get the mail, Dudley,",
"Dudley: Make Harry get it.",
"Harry: Make Dudley get it.",
"Vernon: Poke him with your Smelting stick, Dudley.",
"Harry: That’s mine!",
"Vernon: Who’d be writing to you? P-P-Petunia!",
"Petunia: Vernon! Oh my goodness — Vernon!",
"Dudley: I want to read that letter,",
"Harry: I want to read it, as it’s mine.",
"Vernon: Get out, both of you,",
"Harry: I WANT MY LETTER!",
"Dudley: Let me see it!",
"Vernon: OUT!"
Harry’s Response: I don’t go out, this letter is written to me, why not show me? I want to see my letter.
Keep in mind the following requirements:
1. Before generating the response, you should read the above information and dialogue content carefully.
2. You can not generate the response that is against Harry Potter’s attributes and Harry’s relations with the speaker.
3. Not every component in the background information may be useful, you should choose some of them to help you generate more concise and comprehensive responses that satisfy the behavior of Harry Potter in the dialogue.
4. Not every speaker have relations, familiarity ad affection to Harry. At that time, you can directly predict what would Harry say only based on the dialogue context..
"""
    tmplate4 = """Here is an example:
Dialogue position: Book1-chapter5
Dialogue speakers: "Harry","Hagrid"

Harry’s attributes:
"name": "Harry",
"nickname": "The boy who lived",
"gender": "male",
"age": "age 11",
"looks": "Very thin, black hair, emerald green eyes, wearing glasses, knife injury with lightning shape at the forehead",
"hobbies": "None",
"character": "None",
"talents": "None",
"export": "None",
"belongings": "None",
"affiliation": "None",
"lineage": "None",
"title": "The boy who lived",
"spells": ""

Speakers relations with Harry:             
"Hagrid": {
    "name": "Hagrid",
    "friend": 1.0, "classmate": 0.0, "teacher": 0.0, "family": 0.0, "immediate family": 0.0, "lover": 0.0,
    "opponent": 0.0, "colleague": 0.0, "teammate": 0.0, "enemy": 0.0,
    "Harry's affection to him": 5.0, "Harry's familiarity with him": 4.0,
    "His affection to Harry": 4.0, "His familiarity with Harry": 6.0
}

Dialogue: 
"Harry: It was a dream, I dreamed a giant called Hagrid came to tell me I was going to a school for wizards. When I open my eyes I’ll be at home in my cupboard. And there’s Aunt Petunia knocking on the door, All right, I’m getting up. Don’t do that.",
"Hagrid: Pay him,",
"Harry: What? What?",
"Hagrid: He wants payin’ fer deliverin’ the paper. Look in the pockets. Give him five Knuts,",
"Harry: Knuts?",
"Hagrid: The little bronze ones. Best be off, Harry, lots ter do today, gotta get up ter London an’ buy all yer stuff fer school.",
"Harry: Um — Hagrid?",
"Hagrid: Mm?",
"Harry: I haven’t got any money — and you heard Uncle Vernon last night . . . he won’t pay for me to go and learn magic. But if their house was destroyed —",
"Hagrid: They didn’ keep their gold in the house, boy! Nah, first stop fer us is Gringotts. Wizards’ bank. Have a sausage, they’re not bad cold — an’ I wouldn’ say no teh a bit o’ yer birthday cake, neither.",
"Harry: Wizards have banks?",
"Hagrid: Just the one.",
"Harry: Goblins? Goblins? Goblins? Goblins? Goblins? Goblins? Goblins? Goblins? Goblins? Goblins?",
"Hagrid: He usually gets me ter do important stuff fer him. Fetchin’ you — gettin’ things from Gringotts — knows he can trust me, see. Got everythin’? Come on, then. "
Harry’s Response: Well, I understand that Dumbledore trusted you very much.
Keep in mind the following requirements:
1. Before generating the response, you should read the above information and dialogue content carefully.
2. You can not generate the response that is against Harry Potter’s attributes and Harry’s relations with the speaker.
3. Not every component in the background information may be useful, you should choose some of them to help you generate more concise and comprehensive responses that satisfy the behavior of Harry Potter in the dialogue.
4. Not every speaker have relations, familiarity ad affection to Harry. At that time, you can directly predict what would Harry say only based on the dialogue context..
"""
    tmplate5 = """Here is an example:
Dialogue position: Book1-chapter5
Dialogue speakers: "Harry","Hagrid"

Harry’s attributes:
"name": "Harry",
"nickname": "The boy who lived",
"gender": "male",
"age": "age 11",
"looks": "Very thin, black hair, emerald green eyes, wearing glasses, knife injury with lightning shape at the forehead",
"hobbies": "None",
"character": "None",
"talents": "None",
"export": "None",
"belongings": "None",
"affiliation": "None",
"lineage": "None",
"title": "The boy who lived",
"spells": ""

Dialogue: 
"Hagrid: Got everythin’? Come on, then.",
"Harry: How did you get here?",
"Hagrid: Flew,",
"Harry: Flew?",
"Hagrid: Yeah ",
"Harry: Why would you be mad to try and rob Gringotts?",
"Hagrid: Spells — enchantments, They say there’s dragons guardin’ the high-security vaults. And then yeh gotta find yer way — Gringotts is hundreds of miles under London, see. Deep under the Underground. Yeh’d die of hunger tryin’ ter get out, even if yeh did manage ter get yer hands on summat. Ministry o’ Magic messin’ things up as usual,",
"Harry: There’s a Ministry of Magic?",
"Hagrid: ’Course, Bungler if ever there was one. So he pelts Dumbledore with owls every morning, askin’ fer advice.",
"Harry: But what does a Ministry of Magic do?",
"Hagrid: Well, their main job is to keep it from the Muggles that there’s still witches an’ wizards up an’ down the country.",
"Harry: Why? Why? Why?",
"Hagrid: Why? Blimey, Harry, everyone’d be wantin’ magic solutions to their problems. Nah, we’re best left alone."
Harry’s Response: I know, we should not provoke these things.
Keep in mind the following requirements:
1. Before generating the response, you should read the above information and dialogue content carefully.
2. You can not generate the response that is against Harry Potter’s attributes and Harry’s relations with the speaker.
3. Not every component in the background information may be useful, you should choose some of them to help you generate more concise and comprehensive responses that satisfy the behavior of Harry Potter in the dialogue.
4. Not every speaker have relations, familiarity ad affection to Harry. At that time, you can directly predict what would Harry say only based on the dialogue context..
"""

    content_head = (
        "Your task is to act as a Harry Potter-like dialogue agent in a Magic World. "
        "There is a dialogue between Harry Potter and other characters, and you are required to respond as Harry Potter, "
        "reflecting his personality, mindset, and the context of his past experiences."
    )
    return content_head + "\n" + (tmplate + tmplate1 + tmplate2 + tmplate3 + tmplate4)

def build_hpd_system_prompt(style: str = "llama") -> str:
    return format_system(hpd_6shot_system_content(), style=style)

# =============================== HPD Dataset =================================
def build_hpd_instruction(sample: Dict[str, Any]) -> Tuple[str, str]:
    """
    Build a single HPD training example:
    - instruction: user block (scene/position/speakers/attributes/relations/dialogue + final 'Harry's Response:')
    - answer: the ground-truth positive_response
    """
    scene = sample.get("scene", "")
    position = sample.get("position", "")
    speakers = sample.get("speakers", [])
    if isinstance(speakers, (list, tuple)):
        speakers_str = ", ".join(map(str, speakers))
    else:
        speakers_str = str(speakers)

    attributes = sample.get("attributes", {})
    harry_attr = attributes.get("Harry", attributes.get("harry", attributes))
    rel = sample.get("relations with Harry", sample.get("relations_with_Harry", sample.get("relations_with_harry", {})))
    dialogue_lines = sample.get("dialogue", [])
    if isinstance(dialogue_lines, list):
        dialogue_text = "\n".join(dialogue_lines)
    else:
        dialogue_text = str(dialogue_lines)

    instruction = (
        f"Scene: {scene}\n"
        f"Dialogue Position: {position}\n"
        f"Speakers: {speakers_str}\n\n"
        f"Harry’s attributes: {json.dumps(harry_attr, ensure_ascii=False)}\n\n"
        f"Speakers relations with Harry: {json.dumps(rel, ensure_ascii=False)}\n"
        f"Dialogue:\n{dialogue_text}\n"
        "Harry's Response:\n"
    )
    answer = sample.get("positive_response", sample.get("answer", ""))
    return instruction, str(answer)

class HPDTrainKDDataset(Dataset):
    """
    HPD KD dataset used for teacher-forcing samples.
    Each item constructs (prompt_text = user + assistant_BOS, response_text = ground-truth).
    """
    def __init__(self, hf_ds, tokenizer, prompt_style: str = "llama"):
        self.ds = hf_ds
        self.tok = tokenizer
        self.prompt_style = prompt_style

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        instruction, answer = build_hpd_instruction(ex)
        prompt_text = format_prompt(instruction, self.prompt_style)  # user + assistant_BOS
        response_text = answer or ""
        prompt_ids = self.tok.encode(prompt_text, add_special_tokens=False)
        response_ids = self.tok.encode(response_text, add_special_tokens=False)
        return {
            "prompt_text": prompt_text,
            "response_text": response_text,
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "response_ids": torch.tensor(response_ids, dtype=torch.long),
        }

def collate_single(batch): return batch[0]

# ================================= Utilities =================================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def save_curve(vals: List[float], title: str, out_path: Path, xlabel: str = "Iteration"):
    plt.figure()
    plt.plot(vals, label=title)
    plt.xlabel(xlabel); plt.ylabel("Loss"); plt.title(title)
    plt.legend(); out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()

# ========================== KD helper (same logic) ===========================
def kd_loss_on_assistant_prefix(
    tokenizer,
    teacher_model,                # base model (with system)
    student_with_mem: MemoryCell, # [BE] wrapper (no system)
    system_prompt: str,
    prompt_text: str,             # user + assistant_BOS
    response_text: str,           # ground-truth answer (can be empty)
    N_mem_tokens: int,
    kd_T_tokens: int = 32,
    kd_temperature: float = 2.0,
    use_teacher_gen: bool = False,
) -> Tuple[torch.Tensor, int]:
    """
    Returns: (KD loss, used token count T')
    Teacher: input = system + prompt + (answer[:T'-1]) -> take logits at [prefix_len-1 : prefix_len-1+T']
    Student: input = prompt + (answer[:T'-1]) with [BE] prepended -> take corresponding logits
    """
    device = next(student_with_mem.parameters()).device

    # Encode system + user prefix
    sys_ids = tokenizer(system_prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(device)

    # Target answer sequence
    if use_teacher_gen:
        with torch.no_grad():
            teacher_prefix = torch.cat([sys_ids[0], prompt_ids[0]], dim=0).unsqueeze(0)
            teacher_prefix_am = torch.ones_like(teacher_prefix, dtype=torch.long)  # attention mask
            gen_ids = teacher_model.generate(
                input_ids=teacher_prefix,
                attention_mask=teacher_prefix_am,
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

    # Teacher (with system)
    teacher_input_ids = torch.cat([sys_ids[0], prompt_ids[0], ans_ids_full[:max(0, Tprime-1)]], dim=0).unsqueeze(0)
    teacher_am = torch.ones_like(teacher_input_ids, dtype=torch.long)  # attention mask
    teacher_outputs = teacher_model(input_ids=teacher_input_ids, attention_mask=teacher_am)

    prefix_len_teacher = sys_ids.shape[1] + prompt_ids.shape[1]
    teacher_logits_slice = teacher_outputs.logits[:, prefix_len_teacher-1 : prefix_len_teacher-1+Tprime, :]  # (1, T', V)

    # Student ([BE] + prompt)
    student_input_ids = torch.cat([prompt_ids[0], ans_ids_full[:max(0, Tprime-1)]], dim=0).unsqueeze(0)
    student_am = torch.ones_like(student_input_ids, dtype=torch.long)  # attention mask
    # Option A: let forward create memory_state automatically
    student_outputs, _ = student_with_mem(input_ids=student_input_ids, attention_mask=student_am)
    # Option B (explicit):
    # be_state = student_with_mem.set_memory(student_input_ids.shape)
    # student_outputs, _ = student_with_mem(input_ids=student_input_ids, attention_mask=student_am, memory_state=be_state)

    prefix_len_student = N_mem_tokens + prompt_ids.shape[1]
    student_logits_slice = student_outputs.logits[:, prefix_len_student-1 : prefix_len_student-1+Tprime, :]

    # Temperature KD
    T = kd_temperature
    log_p = F.log_softmax(student_logits_slice / T, dim=-1)
    with torch.no_grad():
        q = F.softmax(teacher_logits_slice / T, dim=-1)
    kd = -(q * log_p).sum(dim=-1).mean() * (T * T)
    return kd, Tprime

# =============================== Train (Phase-1) =============================
def parse_arguments():
    p = argparse.ArgumentParser(description="Phase-1 for HPD: AE + KD (assistant prefix) to learn [BE]")
    # Model / Tokenizer / AE
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--tokenizer_path', type=str, default=None)
    p.add_argument('--ae_vector_path', type=str, default="./results/ae_token/ae_vector.pt",
                   help='Pretrained [AE] embedding vector (shape 1 x D or D)')

    # HPD dataset paths
    p.add_argument('--hpd_train_ds', type=str, required=True,
                   help='datasets.load_from_disk() directory, e.g. '
                        '"experience/role-play/hpd/dataset/Process_fine_tune_ds_no_mw_shot_small/train"')

    # Training
    p.add_argument('--N_mem_tokens', type=int, default=1)
    p.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'float16', 'bfloat16'])
    p.add_argument('--initial_lr', type=float, default=1e-2)
    p.add_argument('--initial_max_iterations', type=int, default=2500)
    p.add_argument('--lm_loss_weight', type=float, default=0.5)
    p.add_argument('--early_stopping_patience', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)

    # KD specifics
    p.add_argument('--kd_T_tokens', type=int, default=32, help='Number of assistant prefix tokens T to align')
    p.add_argument('--kd_temperature', type=float, default=2.0, help='KD temperature τ')
    p.add_argument('--kd_use_teacher_gen', action='store_true',
                   help='If set, ignore dataset answers and use teacher-generated first T tokens for KD; '
                        'otherwise use dataset answer (skip KD if empty).')

    # Prompt style
    p.add_argument('--prompt_style', type=str, default='llama', choices=['short', 'llama', 'qwen3'])

    # Output
    p.add_argument('--output_dir', type=str, required=True)
    return p.parse_args()

def main():
    args = parse_arguments()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]

    tok_path = args.tokenizer_path or args.model_path
    print(f"⏳ Loading tokenizer from {tok_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"⏳ Loading base model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2"   # use FlashAttention-2 if available
    )
    model.to(device)                               # move model to target device
    model.eval()

    # Build HPD 6-shot system prompt (AE reconstruction target)
    system_prompt = build_hpd_system_prompt(style=args.prompt_style)
    print("\n🧾 HPD SYSTEM_PROMPT (first 200 chars):\n", system_prompt[:200], "...\n", flush=True)

    # Load pretrained [AE] vector
    print(f"⏳ Loading AE vector: {args.ae_vector_path}")
    ae_vector = torch.load(args.ae_vector_path, map_location=device).to(dtype)
    if ae_vector.dim() == 1:
        ae_vector = ae_vector.unsqueeze(0)  # (1, D)
    ae_vector_batch = ae_vector.unsqueeze(0).to(dtype)         # (1, 1, D)

    # Memory wrapper (train only the [BE] prefix)
    config = model.config
    memory_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size'))
    model_with_memory = MemoryCell(base_model=model, num_mem_tokens=args.N_mem_tokens, memory_dim=memory_dim).to(device)

    # HPD training data (for KD)
    print(f"⏳ Loading HPD KD dataset from: {args.hpd_train_ds}")
    hf_ds = load_from_disk(args.hpd_train_ds)
    kd_dataset = HPDTrainKDDataset(hf_ds, tokenizer, prompt_style=args.prompt_style)
    kd_loader = DataLoader(kd_dataset, batch_size=1, shuffle=True, collate_fn=collate_single)

    # Optimizer & tracking
    opt = AdamW(model_with_memory.parameters(), lr=args.initial_lr)
    ce_loss = torch.nn.CrossEntropyLoss()
    ae_losses, kd_losses, total_losses = [], [], []

    # AE labels (system prompt tokens)
    label_ids = tokenizer(system_prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

    # Early stopping
    patience = args.early_stopping_patience
    best_loss = float('inf'); best_state = None; best_iter = -1; no_improve = 0

    # Training loop (each step: AE + one KD sample)
    pbar = tqdm(range(args.initial_max_iterations), desc="[HPD] Phase-1 AE+KD")
    loader_iter = iter(kd_loader)
    for it in pbar:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(kd_loader)
            batch = next(loader_iter)

        with torch.amp.autocast(device_type='cuda' if device=='cuda' else 'cpu', dtype=dtype):
            # -------- AE: reconstruct system ( [BE] + [AE] -> system ) --------
            memory_state = model_with_memory.set_memory(label_ids.shape)  # (1, N_mem, D)
            label_embeds = model.get_input_embeddings()(label_ids)        # (1, L_sys, D)
            full_ae_embeds = torch.cat([memory_state, ae_vector_batch, label_embeds], dim=1)
            # Pass attention_mask to avoid warnings
            ae_am = torch.ones(full_ae_embeds.shape[:2], dtype=torch.long, device=full_ae_embeds.device)
            ae_outputs = model(inputs_embeds=full_ae_embeds, attention_mask=ae_am)
            logits_for_ae = ae_outputs.logits[:, args.N_mem_tokens : args.N_mem_tokens + label_ids.shape[1], :]
            loss_ae = ce_loss(logits_for_ae.reshape(-1, logits_for_ae.size(-1)), label_ids.reshape(-1))

            # -------- KD: temperature KL on assistant first T tokens --------
            kd_l, used_T = kd_loss_on_assistant_prefix(
                tokenizer=tokenizer,
                teacher_model=model,
                student_with_mem=model_with_memory,
                system_prompt=system_prompt,
                prompt_text=batch["prompt_text"],
                response_text=batch["response_text"],
                N_mem_tokens=args.N_mem_tokens,
                kd_T_tokens=args.kd_T_tokens,
                kd_temperature=args.kd_temperature,
                use_teacher_gen=args.kd_use_teacher_gen,
            )

            if used_T > 0:
                loss_total = (1 - args.lm_loss_weight) * loss_ae + args.lm_loss_weight * kd_l
            else:
                loss_total = loss_ae

        loss_total.backward()
        opt.step(); opt.zero_grad()

        lt = float(loss_total.detach().cpu())
        la = float(loss_ae.detach().cpu())
        lk = float(kd_l.detach().cpu()) if isinstance(kd_l, torch.Tensor) else 0.0

        ae_losses.append(la); kd_losses.append(lk); total_losses.append(lt)
        if lt < best_loss - 1e-8:
            best_loss = lt; best_state = model_with_memory.memory.data.detach().clone()
            best_iter = it; no_improve = 0
        else:
            no_improve += 1

        pbar.set_postfix({"total": lt, "AE": la, "KD": lk, "best": best_loss, "pat": no_improve})
        if patience and patience > 0 and no_improve >= patience:
            print(f"⏹️ Early stopping at iter={it}, best_iter={best_iter}, best_loss={best_loss:.6f}")
            break
    pbar.close()

    # Roll back to the best [BE] state
    if best_state is not None:
        model_with_memory.memory.data.copy_(best_state)
    print(f"✅ Phase-1 complete. Best total loss: {best_loss:.6f} at iter={best_iter}")

    # Save artifacts
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1_plot = out_dir / "phase1_losses.png"
    plt.figure()
    plt.plot(ae_losses, label="AE Loss")
    plt.plot(kd_losses, label="KD Loss")
    plt.plot(total_losses, label="Total Loss")
    plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.title("Phase-1 Losses (HPD)")
    plt.legend(); plt.savefig(phase1_plot, dpi=150, bbox_inches='tight'); plt.close()
    print(f"📈 Saved Phase-1 loss curve to: {phase1_plot}")

    # Save the learned [BE] vector
    mem_out_path = out_dir / 'system_prompt_be_hpd.pt'
    torch.save(model_with_memory.memory.data.clone(), mem_out_path)
    print(f"💾 Saved final [BE] vector to: {mem_out_path}")

if __name__ == "__main__":
    main()

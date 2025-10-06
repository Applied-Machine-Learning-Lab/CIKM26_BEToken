# train_ae_token_v3_fixed.py

import argparse
import json
import os
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from dataclasses import dataclass, field

# --- Argument Classes (No changes) ---
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="../models/Qwen/Qwen3-4B-Instruct-2507",
metadata={"help": "The base model to use."})
    ae_token: str = field(default="<|AE|>", metadata={"help": "The special token for Auto-Encoding."})
    torch_dtype: str = field(default="bfloat16", metadata={"help": "Model dtype (bfloat16, float16, float32)."})
    use_flash_attention_2: bool = field(default=True, metadata={"help": "Whether to use Flash Attention 2."})

@dataclass
class DataArguments:
    jsonl_path: str = field(default="./data/merged_pwc_cosmopedia.jsonl", metadata={"help": "Path to the processed JSONL data."})
    max_text_length: int = field(default=1024, metadata={"help": "Maximum token length for a text chunk in the dataset."})
    min_text_length: int = field(default=64, metadata={"help": "Minimum token length for a text chunk."})

@dataclass
class ScriptTrainingArguments(TrainingArguments):
    output_dir: str = field(default="./results/ae_token_tuning2", metadata={"help": "Output directory for the trained AE vector."})
    num_train_epochs: float = field(default=2, metadata={"help": "Number of training epochs."})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size for training."})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "Gradient accumulation steps."})
    learning_rate: float = field(default=1e-3, metadata={"help": "Learning rate for tuning the AE token embedding."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=10000, metadata={"help": "Save checkpoint every X updates steps."})
    optim: str = field(default="adamw_torch", metadata={"help": "Optimizer to use."})
    # --- ADD THIS LINE ---
    save_safetensors: bool = field(default=False, metadata={"help": "Disable safetensors for saving checkpoints due to tied weights."})

# --- Model Wrapper (No changes) ---
class AEModelWrapper(nn.Module):
    def __init__(self, model: AutoModelForCausalLM, ae_token_id: int):
        super().__init__()
        self.model = model
        self.ae_token_id = ae_token_id
        for param in self.model.parameters():
            param.requires_grad = False
        embeddings = self.model.get_input_embeddings()
        embedding_dim = getattr(self.model.config, 'hidden_size')
        device = embeddings.weight.device
        dtype = embeddings.weight.dtype
        self.ae_vector = nn.Parameter(
            torch.randn(embedding_dim, device=device, dtype=dtype) * embeddings.weight.data.std(),
            requires_grad=True
        )

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        input_embeddings = self.model.get_input_embeddings()(input_ids)
        ae_token_mask = (input_ids == self.ae_token_id)
        input_embeddings[ae_token_mask] = self.ae_vector.to(input_embeddings.dtype)
        return self.model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

# --- FIX: Re-introduce the Custom Data Collator ---
@dataclass
class CustomDataCollator:
    """
    Pads `input_ids` and `attention_mask` using the tokenizer's settings,
    and manually pads `labels` with -100 to ignore these tokens in loss calculation.
    """
    tokenizer: AutoTokenizer

    def __call__(self, features: list[dict[str, any]]) -> dict[str, torch.Tensor]:
        # Separate labels from the rest of the features
        label_features = [feature.pop("labels") for feature in features]
        
        # Pad the remaining features (input_ids, attention_mask) using the tokenizer
        batch = self.tokenizer.pad(
            features,
            padding="longest",
            return_tensors="pt",
        )

        # Manually pad the labels to the same length as the padded inputs
        max_label_length = batch['input_ids'].shape[1]
        padded_labels = [label + [-100] * (max_label_length - len(label)) for label in label_features]
        
        # Add the padded labels back to the batch
        batch['labels'] = torch.tensor(padded_labels, dtype=torch.long)
        
        return batch


# --- Main Logic ---
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, ScriptTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- 1. Setup Model and Tokenizer ---  
    print("Setting up model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # ==================== FIX FOR QWEN3 ====================
    # Qwen3 with Flash Attention requires left padding.
    tokenizer.padding_side = 'left'
    # Many models like Qwen don't have a default pad token. It's common
    # practice to use the end-of-sequence token as the pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # =======================================================

    special_tokens_dict = {'additional_special_tokens': [model_args.ae_token]}
    tokenizer.add_special_tokens(special_tokens_dict)
    ae_token_id = tokenizer.convert_tokens_to_ids(model_args.ae_token)

    # Note: The original code's check for pad_token comes after adding special tokens.
    # It's better to set it before any potential padding operations.
    # The rest of your script can remain the same.
    ae_token_id = tokenizer.convert_tokens_to_ids(model_args.ae_token)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    base_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=getattr(torch, model_args.torch_dtype),
        use_flash_attention_2=model_args.use_flash_attention_2
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = AEModelWrapper(base_model, ae_token_id)
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"- {name} ({param.numel()} params)")

    # --- 2. Tokenize Dataset for Training (No changes) ---
    def tokenize_function(examples):
        prompts = [text + model_args.ae_token for text in examples["text"]]
        completions = [text + tokenizer.eos_token for text in examples["text"]]
        full_texts = [p + c for p, c in zip(prompts, completions)]
        max_len = (data_args.max_text_length * 2) + 10
        full_tokenized = tokenizer(
            full_texts, add_special_tokens=True, padding=False, truncation=True, max_length=max_len
        )
        prompt_tokenized = tokenizer(prompts, add_special_tokens=False)
        labels = []
        for i, full_input_ids in enumerate(full_tokenized['input_ids']):
            prompt_len_in_full_seq = len(prompt_tokenized['input_ids'][i]) + 1
            label = list(full_input_ids)
            label[:prompt_len_in_full_seq] = [-100] * prompt_len_in_full_seq
            labels.append(label)
        return {"input_ids": full_tokenized.input_ids, "attention_mask": full_tokenized.attention_mask, "labels": labels}

    print("Loading and tokenizing dataset...")
    raw_dataset = load_dataset('json', data_files=data_args.jsonl_path, split='train')
    def filter_by_length(example):
        token_count = len(tokenizer(example['text']).input_ids)
        return data_args.min_text_length <= token_count <= data_args.max_text_length
    filtered_dataset = raw_dataset.filter(filter_by_length)
    print(f"Filtered dataset to {len(filtered_dataset)} samples (length between {data_args.min_text_length} and {data_args.max_text_length} tokens).")
    tokenized_dataset = filtered_dataset.map(
        tokenize_function, batched=True, remove_columns=filtered_dataset.column_names, num_proc=os.cpu_count() // 2 or 1
    )
    print(f"Dataset prepared. Number of training samples: {len(tokenized_dataset)}")

    # --- FIX: Instantiate the CustomDataCollator ---
    data_collator = CustomDataCollator(tokenizer=tokenizer)

    # --- 3. Train (No changes) ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    print("Starting training...")
    trainer.train()

    # --- 4. Save the Final Vector (No changes) ---
    print("Training complete.")
    final_vector_path = os.path.join(training_args.output_dir, "ae_vector.pt")
    torch.save(model.ae_vector, final_vector_path)
    print(f"✅ Trained AE vector saved to: {final_vector_path}")
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"✅ Tokenizer saved to: {training_args.output_dir}")

if __name__ == "__main__":
    main()
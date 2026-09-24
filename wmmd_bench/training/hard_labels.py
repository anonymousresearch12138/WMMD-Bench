"""Historical chat serialization, response mask and right-padding collator."""
import json, time
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import TrainerCallback

class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.rows = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
        self.tokenizer, self.max_length = tokenizer, max_length
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        user = row["instruction"] + (("\n\nInput:\n" + row["input"]) if row["input"] else "")
        prompt = self.tokenizer.apply_chat_template([{"role":"user","content":user}], tokenize=False, add_generation_prompt=True)
        full = self.tokenizer.apply_chat_template([{"role":"user","content":user},{"role":"assistant","content":row["teacher_raw_answer"]}], tokenize=False, add_generation_prompt=False)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded = self.tokenizer(full, add_special_tokens=False, truncation=True, max_length=self.max_length)
        labels = encoded["input_ids"].copy()
        labels[:min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "labels": labels}

class Collator:
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def __call__(self, features):
        width = max(len(x["input_ids"]) for x in features)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = width - len(item["input_ids"])
            result["input_ids"].append(item["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            result["attention_mask"].append(item["attention_mask"] + [0] * pad)
            result["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(value) for key, value in result.items()}

class TimingCallback(TrainerCallback):
    def __init__(self): self.started=None;self.steps=[]
    def on_step_begin(self,args,state,control,**kwargs): self.started=time.monotonic()
    def on_step_end(self,args,state,control,**kwargs):
        if self.started is not None:self.steps.append(time.monotonic()-self.started)

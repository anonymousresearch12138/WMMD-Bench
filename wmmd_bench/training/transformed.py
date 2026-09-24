"""Frozen paraphrased-answer formatter used for active XBb."""
import json
from pathlib import Path
from torch.utils.data import Dataset

class SFTDataset(Dataset):

    def __init__(self, path, tokenizer, max_length):
        self.rows = [json.loads(line) for line in Path(path).open(encoding='utf-8') if line.strip()]
        self.tokenizer, self.max_length = (tokenizer, max_length)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        user = row['instruction'] + ('\n\nInput:\n' + row['input'] if row['input'] else '')
        prompt = self.tokenizer.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True)
        full = self.tokenizer.apply_chat_template([{'role': 'user', 'content': user}, {'role': 'assistant', 'content': row['paraphrased_answer']}], tokenize=False, add_generation_prompt=False)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        encoded = self.tokenizer(full, add_special_tokens=False, truncation=True, max_length=self.max_length)
        labels = encoded['input_ids'].copy()
        labels[:min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        return {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask'], 'labels': labels}

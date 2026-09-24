import hashlib, hmac
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class KeyedCipher(nn.Module):

    def __init__(self, dim, n_layers, key, dtype=torch.bfloat16):
        super().__init__()
        layers = []
        for index in range(n_layers):
            seed = int(hmac.new(key, f'layer{index}'.encode(), hashlib.sha256).hexdigest(), 16) % 2 ** 31
            torch.manual_seed(seed)
            layer = nn.Linear(dim, dim, bias=False)
            nn.init.orthogonal_(layer.weight)
            layers.append(layer.to(dtype))
        self.layers = nn.ModuleList(layers)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, value):
        for layer in self.layers:
            value = value + layer(value)
        return value

class TextDataset(Dataset):

    def __init__(self, texts, tokenizer, max_length):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(self.texts[index], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        ids = encoded.input_ids.squeeze(0)
        return {'input_ids': ids, 'attention_mask': encoded.attention_mask.squeeze(0), 'labels': ids.clone()}

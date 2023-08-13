import numpy as np

import torch
from torch.utils.data import Dataset

label_ids = {}

class ClassificationDataset(Dataset):
    def __init__(self, df, tokenizer, label_ids, max_length):
        self.tokenizer = tokenizer
        self.label_ids = label_ids
        self.max_length = max_length
        self.labels = [self.label_ids[label] for label in df['label']]
        self.texts = [self.tokenizer(text,
                                padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                return_tensors="pt") for text in df['text']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_texts = self.texts[idx]
        batch_y = np.array(self.labels[idx])
        return batch_texts, batch_y
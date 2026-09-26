import torch
from torch.utils.data import Dataset
import pandas as pd

from constants import DATASET_SIZE, mbti_types

mbtis_df = pd.read_csv("mbti_texts.csv")

class MBTIDataset(Dataset):
    def __init__(self, size=DATASET_SIZE):
        self.data = []
        self.sentence_embeddings = []
        self.label_one_hots = []
        for _ in range(size):
            row = mbtis_df.sample(n=1).iloc[0]
            text = row['Text']
            label = row['Type']
            self.data.append((text, label))
        self.label2idx = {t:i for i,t in enumerate(mbti_types)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label_onehot = self.label_one_hots[idx]
        # Fake embedding: random vector (replace with BERT embeddings later)
        embedding = self.sentence_embeddings[idx]  # Use the pre-generated embeddings        
        return embedding, label_onehot

traits_df = pd.read_csv("mbti_traits.csv")

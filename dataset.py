"""Dataset helpers for sampled MBTI text and its prepared model inputs."""

import torch
from torch.utils.data import Dataset
import pandas as pd

from constants import DATASET_SIZE, mbti_types

# The training dataset is sampled from the generated text collection.
mbtis_df = pd.read_csv("mbti_texts.csv")

class MBTIDataset(Dataset):
    """Hold sampled text labels and their later-generated model inputs."""

    def __init__(self, size=DATASET_SIZE):
        """Sample ``size`` text-label pairs and index their MBTI labels."""
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
        """Return the number of sampled records in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return the prepared embedding and one-hot label at ``idx``."""
        label_onehot = self.label_one_hots[idx]
        # Embeddings and labels are attached after the dataset is sampled.
        embedding = self.sentence_embeddings[idx]
        return embedding, label_onehot

# Trait descriptions are looked up after the classifier predicts an MBTI type.
traits_df = pd.read_csv("mbti_traits.csv")

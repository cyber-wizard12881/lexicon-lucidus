from transformers import AutoTokenizer, AutoModel
import torch
from dataset import MBTIDataset
from constants import DATASET_SIZE, mbti_types

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def generate_embeddings(dataset: MBTIDataset):
    sentences = [(text, label) for text, label in dataset.data]  # Extract text and labels from the dataset

    # Sift out the One Hot Labels
    label_onehots = []
    labels = [label for _, label in sentences]
    for label in labels:
        label_idx = dataset.label2idx[label]
        label_onehot = torch.zeros(len(mbti_types))
        label_onehot[label_idx] = 1
        label_onehots.append(label_onehot)
    
    # Tokenize all sentences
    inputs = tokenizer([s[0] for s in sentences], padding=True, truncation=True, return_tensors="pt")

    # Generate embeddings
    with torch.no_grad():
        model_output = model(**inputs)

    token_embeddings = model_output[0]
    attention_mask = inputs['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    sentence_embeddings = sum_embeddings / sum_mask
    print("Embeddings shape:", sentence_embeddings.shape)  
    # Expected: torch.Size([640, 384])
    return sentence_embeddings, label_onehots

def generate_embeddings_for_text(text: str):
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    # Generate embeddings
    with torch.no_grad():
        model_output = model(**inputs)

    token_embeddings = model_output[0]
    attention_mask = inputs['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    sentence_embedding = sum_embeddings / sum_mask
    print("Single embedding shape:", sentence_embedding.shape)  
    # Expected: torch.Size([1, 384])
    return sentence_embedding

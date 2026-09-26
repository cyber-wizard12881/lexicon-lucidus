import torch

from train import encoder, decoder, classifier
from constants import EMBEDDING_DIM, mbti_types
from embeddings import generate_embeddings_for_text
from dataset import traits_df
import pandas as pd

# Load the input sentence from a CSV file or define it directly
input_df = pd.read_csv("inputs.csv")

for index, row in input_df.iterrows():
    expected_label = row['Type']
    sentence = row['Text']
    sentence_embedding = generate_embeddings_for_text(sentence)
    # Forward pass
    with torch.no_grad():
        z = encoder(sentence_embedding)       # latent representation
        logits = classifier(z)                # MBTI prediction
        predicted_idx = torch.argmax(logits, dim=1).item()
        predicted_type = mbti_types[predicted_idx]

    print("Input Sentence:", sentence)
    print("Expected MBTI Type:", expected_label)
    print("Predicted MBTI Type:", predicted_type)

    traits = traits_df[traits_df['Type'] == predicted_type].iloc[0]
    print("Predicted Character Traits:")
    for trait in traits.index:
        if trait != 'Type':
            print(f"  {trait}: {traits[trait]}")

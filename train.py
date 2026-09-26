from torch import nn, optim
import torch

from caae import Classifier, Decoder, Discriminator, Encoder
from constants import BATCH_SIZE, LABEL_DIM, LATENT_DIM, NUM_CLASSES
from constants import LATENT_DIM, NUM_EPOCHS, EMBEDDING_DIM
from dataset import MBTIDataset, DATASET_SIZE
from torch.utils.data import DataLoader
from embeddings import generate_embeddings

input_dim = EMBEDDING_DIM  # Assuming BERT embeddings of size 384

mbti_dataset = MBTIDataset(size=DATASET_SIZE)
sentence_embeddings,label_onehots = generate_embeddings(mbti_dataset)
mbti_dataset.sentence_embeddings = sentence_embeddings
mbti_dataset.label_one_hots = label_onehots

encoder = Encoder(input_dim, LATENT_DIM)
decoder = Decoder(LATENT_DIM, LABEL_DIM, input_dim)
discriminator = Discriminator(LATENT_DIM)
classifier = Classifier(LATENT_DIM, num_classes=NUM_CLASSES)

loader = DataLoader(mbti_dataset, batch_size=BATCH_SIZE, shuffle=True)

optim_enc = optim.Adam(encoder.parameters(), lr=1e-3)
optim_dec = optim.Adam(decoder.parameters(), lr=1e-3)
optim_disc = optim.Adam(discriminator.parameters(), lr=1e-3)
optim_cls = optim.Adam(classifier.parameters(), lr=1e-3)

loss_fn = nn.CrossEntropyLoss()
bce = nn.BCELoss()
mse = nn.MSELoss()

for epoch in range(NUM_EPOCHS):
    for x, labels in loader:
        # Encode
        z = encoder(x)

        # --- Discriminator ---
        real = torch.ones(x.size(0),1)
        fake = torch.zeros(x.size(0),1)
        disc_real = discriminator(torch.randn_like(z))
        disc_fake = discriminator(z.detach())
        loss_disc = bce(disc_real, real) + bce(disc_fake, fake)
        optim_disc.zero_grad(); loss_disc.backward(); optim_disc.step()

        # --- Encoder (adversarial) ---
        disc_fake = discriminator(z)
        loss_enc_adv = bce(disc_fake, real)

        # --- Decoder (reconstruction) ---
        x_hat = decoder(z, labels)
        loss_rec = mse(x_hat, x)

        # --- Classifier ---
        logits = classifier(z)
        target = torch.argmax(labels, dim=1)
        loss_cls = loss_fn(logits, target)

        # --- Total loss ---
        loss_total = loss_rec + loss_enc_adv + loss_cls
        optim_enc.zero_grad(); optim_dec.zero_grad(); optim_cls.zero_grad()
        loss_total.backward()
        optim_enc.step(); optim_dec.step(); optim_cls.step()

    print(f"Epoch {epoch+1}, Loss: {loss_total.item():.4f}")

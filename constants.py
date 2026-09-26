# Captures Constant values used throughout the project.

# MBTI labels are kept in the same order used by the one-hot encodings.
mbti_types = [
    "INTJ","INTP","ENTJ","ENTP",
    "INFJ","INFP","ENFJ","ENFP",
    "ISTJ","ISFJ","ESTJ","ESFJ",
    "ISTP","ISFP","ESTP","ESFP"
]

# Total number of samples drawn for the generated training dataset.
DATASET_SIZE = 640
# Size produced by the sentence-transformer model.
EMBEDDING_DIM = 384

# Size of the compressed representation used by the neural networks.
LATENT_DIM = 64
# Number of values in each one-hot MBTI label.
LABEL_DIM = 16

# Number of MBTI types predicted by the classifier.
NUM_CLASSES = 16

# Number of samples processed in one training step.
BATCH_SIZE = 32

# Number of complete passes through the generated dataset.
NUM_EPOCHS = 5
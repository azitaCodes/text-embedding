import os
import warnings
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Suppress tqdm Jupyter widget warning if running outside Jupyter
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")

# Disable HF_HUB symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Optional: suppress other UserWarnings from huggingface_hub if you want
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# Load pre-trained tokenizer and model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text):
    # Tokenize input text with padding and truncation
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get last hidden states and attention mask
    last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
    attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (batch_size, seq_len, 1)

    # Apply attention mask to ignore padded tokens in the mean pooling
    masked_embeddings = last_hidden_state * attention_mask

    # Sum embeddings and divide by number of valid tokens
    summed = masked_embeddings.sum(dim=1)
    counts = attention_mask.sum(dim=1)
    mean_pooled = summed / counts

    # Normalize the embedding vector (optional but common)
    normalized_embedding = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

    # Convert to numpy array and return
    return normalized_embedding[0].cpu().numpy()

# Example input text
input_text = "The quick brown fox jumps over the lazy dog."

embedding_vector = embed_text(input_text)

print(f"Input text: {input_text}")
print(f"Embedding vector shape: {embedding_vector.shape}")
print(f"Embedding vector (first 10 values): {embedding_vector[:10]}")
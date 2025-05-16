# 🧠 Text Embedding using Hugging Face Transformers

This project demonstrates how to generate sentence embeddings using the [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model with the Hugging Face Transformers library.  
The output is a fixed-length 384-dimensional vector that captures the semantic meaning of the input text. This is useful for tasks such as text similarity, clustering, and semantic search.

## 📦 Requirements

- Python 3.7+
- transformers
- torch
- numpy
- tqdm

Install dependencies using:

```bash
pip install -r requirements.txt

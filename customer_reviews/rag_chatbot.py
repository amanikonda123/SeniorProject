# rag_chatbot.py

import os
import pandas as pd
from typing import List

# RAG dependencies
from sentence_transformers import SentenceTransformer
import faiss

# Transformers for Mistral
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Summarizer utility (for chunking) and your scraper
from customer_reviews.reviews_summary import chunk_text
from customer_reviews.walmart_scraper import WalmartScraper

# ──────────────────────────────────────────────────────────────────────────────
# 1) Load and initialize Mistral 7B
# ──────────────────────────────────────────────────────────────────────────────
# Adjust device map as needed (e.g., 'auto' will use available GPUs)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model (ensure you have installed `transformers`, `torch`)
tokenizer = AutoTokenizer.from_pretrained(
    "mistral-ai/mistral-7b-v0.1",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "mistral-ai/mistral-7b-v0.1",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ──────────────────────────────────────────────────────────────────────────────
# 2) Globals for FAISS index
# ──────────────────────────────────────────────────────────────────────────────
_emb_model: SentenceTransformer = None
_index: faiss.IndexFlatL2 = None
_chunks: List[str] = []

# ──────────────────────────────────────────────────────────────────────────────
# 3) Build & query the FAISS index
# ──────────────────────────────────────────────────────────────────────────────
def build_rag_index(chunks: List[str]) -> None:
    """Encode chunks and build a FAISS L2 index."""
    global _emb_model, _index, _chunks
    _emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = _emb_model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    _index = faiss.IndexFlatL2(dim)
    _index.add(embeddings)
    _chunks = chunks


def retrieve_top_k(question: str, k: int = 5) -> List[str]:
    """Return the top-k most similar chunks to the question."""
    if _emb_model is None or _index is None:
        raise RuntimeError("RAG index has not been built. Call build_rag_index first.")
    q_emb = _emb_model.encode([question], convert_to_numpy=True)
    distances, indices = _index.search(q_emb, k)
    return [_chunks[i] for i in indices[0]]

# ──────────────────────────────────────────────────────────────────────────────
# 4) Generation helper using Mistral 7B
# ──────────────────────────────────────────────────────────────────────────────
def generate_answer(prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ──────────────────────────────────────────────────────────────────────────────
# 5) Chat function: retrieve and answer using Mistral
# ──────────────────────────────────────────────────────────────────────────────
def chat_with_reviews(
    prod_sku: str,
    question: str,
    top_k: int = 5
) -> str:
    """
    1) Load or scrape reviews, chunk them
    2) Build (or reuse) a FAISS index
    3) Retrieve top_k relevant chunks
    4) Answer the question using Mistral 7B as LLM
    """
    # a) Load or scrape reviews
    csv_file = f"{prod_sku}_reviews.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        reviews = WalmartScraper().get_reviews(prod_sku)
        df = pd.DataFrame(reviews)
        df.to_csv(csv_file, index=False)

    # b) Concatenate and chunk for RAG
    full_text = "\n".join(df["text"].astype(str).tolist())
    chunks = chunk_text(full_text, max_chars=2000)

    # c) Build or refresh the index
    if _index is None or len(_chunks) != len(chunks):
        build_rag_index(chunks)

    # d) Retrieve top_k relevant chunks
    context_chunks = retrieve_top_k(question, k=top_k)
    context = "\n\n---\n\n".join(context_chunks)

    # e) Create prompt and generate answer using a single f-string
    prompt = f"""
You are a customer-review expert. Use the following review excerpts to answer the user’s question.

{context}

Question: {question}
Answer concisely, citing any context briefly.
"""
    return generate_answer(prompt)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Optional CLI for testing
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Chatbot over Walmart reviews using Mistral 7B"
    )
    parser.add_argument("prod_sku", help="Walmart product SKU to load reviews for")
    parser.add_argument("question", help="User question about the reviews")
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of review chunks to retrieve for context"
    )
    args = parser.parse_args()

    answer = chat_with_reviews(args.prod_sku, args.question, top_k=args.top_k)
    print("\n=== Chatbot Answer ===\n", answer)

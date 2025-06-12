# rag_exp.py

import os
import re
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from dotenv import load_dotenv
from mistralai.client import MistralClient
from customer_reviews.reviews_summary import get_review_summary
from urllib.parse import urlparse
import nltk

# nltk.download('punkt')  # if running first time

load_dotenv()
client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
model_name = "open-mistral-7b"

def extract_sku_from_url(url: str) -> str:
    parsed = urlparse(url)
    segs = [s for s in parsed.path.split('/') if s]
    if segs and segs[-1].isdigit():
        return segs[-1]
    m = re.search(r"/(\d{5,})(?:$|\?)", url)
    return m.group(1) if m else ""

def get_text_embedding(text: str) -> np.ndarray:
    resp = client.embeddings(model="mistral-embed", input=[text])
    return np.array(resp.data[0].embedding, dtype="float32")

def chunk_reviews(texts: list) -> list:
    chunks = []
    for t in texts:
        chunks.extend(nltk.sent_tokenize(t))
    return chunks

def get_or_build_index(sku: str, chunks: list) -> (faiss.IndexFlatL2, list):
    cache_dir = ".cache_embeddings"
    os.makedirs(cache_dir, exist_ok=True)
    emb_path = os.path.join(cache_dir, f"{sku}_embeddings.npy")
    docs_path = os.path.join(cache_dir, f"{sku}_docs.pkl")

    if os.path.exists(emb_path) and os.path.exists(docs_path):
        embs = np.load(emb_path)
        with open(docs_path, 'rb') as f:
            docs = pickle.load(f)
    else:
        embs_list = []
        for c in tqdm(chunks, desc="Embedding review chunks"):
            embs_list.append(get_text_embedding(c))
        embs = np.stack(embs_list)
        docs = chunks
        np.save(emb_path, embs)
        with open(docs_path, 'wb') as f:
            pickle.dump(docs, f)

    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)
    return idx, docs

def run_mistral(prompt: str) -> str:
    resp = client.chat(model=model_name, messages=[{"role":"user","content":prompt}])
    return resp.choices[0].message.content

def run_standard_rag_on_reviews(
    product_url: str,
    question: str,
    base_k: int = 3,
    max_k: int = 10,
    thresh_ratio: float = 0.7
) -> (str, str):
    sku = extract_sku_from_url(product_url)
    if not sku:
        raise ValueError("Invalid product URL: could not extract SKU.")

    _, df_reviews, _ = get_review_summary("CSV", sku)
    texts = df_reviews["text"].astype(str).tolist()
    records = []
    for _, row in df_reviews.iterrows():
        parts = [
            f"Product: {row.get('productName','')}",
            f"Title: {row.get('title','')}",
            f"Rating: {row.get('rating','')}",
            f"Submitted: {row.get('submissionTime','')}",
            f"Badges: {row.get('badges','')}",
            f"Positive feedback: {row.get('positivefeedback','')}",
            f"Negative feedback: {row.get('negativefeedback','')}",
            f"Review: {row.get('text','')}"
        ]
        records.append(". ".join(parts))
    texts = records
    chunks = chunk_reviews(texts)
    idx, docs = get_or_build_index(sku, chunks)

    # embed & retrieve up to max_k
    qemb = get_text_embedding(question).reshape(1, -1)
    scores, ids = idx.search(qemb, max_k)
    candidates = [(scores[0][i], docs[ids[0][i]]) for i in range(max_k)]

    cutoff = candidates[0][0] * thresh_ratio
    selected = [doc for score, doc in candidates if score >= cutoff]
    if not selected:
        selected = [doc for _, doc in candidates[:base_k]]
    context = " ".join(selected)

    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    answer = run_mistral(prompt)
    return answer, context

def run_multihop_rag_on_reviews(
    product_url: str,
    question: str,
    hops: int = 2,
    base_k: int = 3,
    max_k: int = 10
) -> (str, list):
    sku = extract_sku_from_url(product_url)
    if not sku:
        raise ValueError("Invalid product URL.")
    _, df, _ = get_review_summary("CSV", sku)
    texts = df["text"].astype(str).tolist()
    chunks = chunk_reviews(texts)
    idx, docs = get_or_build_index(sku, chunks)

    contexts = []
    current_query = question
    for i in range(hops):
        k_i = max_k if i == 0 else base_k
        qemb = get_text_embedding(current_query).reshape(1, -1)
        scores, ids = idx.search(qemb, k_i)
        ctx = " ".join(docs[ids[0][j]] for j in range(min(k_i, len(ids[0]))))
        contexts.append(ctx)

        prompt = f"Context: {ctx}\nQuestion: {current_query}\nAnswer:"
        current_query = run_mistral(prompt)

    final_ctx = "\n---\n".join(contexts)
    final_prompt = f"Contexts:\n{final_ctx}\nOriginal question: {question}\nAnswer:"
    final_answer = run_mistral(final_prompt)
    return final_answer, contexts

# alias old behavior
run_rag_on_reviews = run_standard_rag_on_reviews

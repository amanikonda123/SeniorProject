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
# nltk.download('punkt_tab')


# Load environment variables
load_dotenv()

# Initialize Mistral client
client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
model_name = "open-mistral-7b"

# --- Utility Functions ---

def extract_sku_from_url(url: str) -> str:
    """
    Extracts the numeric SKU from a Walmart product URL.
    """
    parsed = urlparse(url)
    segments = [seg for seg in parsed.path.split('/') if seg]
    if segments and segments[-1].isdigit():
        return segments[-1]
    match = re.search(r"/(\d{5,})(?:$|\?)", url)
    return match.group(1) if match else ""


def get_text_embedding(text: str) -> np.ndarray:
    """
    Generate and return a float32 numpy embedding for the given text
    using Mistral's embedding endpoint.
    """
    resp = client.embeddings(model="mistral-embed", input=[text])
    return np.array(resp.data[0].embedding, dtype="float32")


def chunk_reviews(texts: list) -> list:
    """
    Split each review into sentences for granular retrieval.
    """
    chunks = []
    for text in texts:
        chunks.extend(nltk.sent_tokenize(text))
    return chunks


def get_or_build_index(sku: str, chunks: list) -> (faiss.IndexFlatL2, list):
    """
    Load cached embeddings/docs if available, otherwise compute and cache them.
    Shows embedding progress on first run.
    Returns a FAISS index and the list of text chunks.
    """
    cache_dir = ".cache_embeddings"
    os.makedirs(cache_dir, exist_ok=True)
    emb_path = os.path.join(cache_dir, f"{sku}_embeddings.npy")
    docs_path = os.path.join(cache_dir, f"{sku}_docs.pkl")

    if os.path.exists(emb_path) and os.path.exists(docs_path):
        print(f"Loading cached embeddings for SKU: {sku}")
        embeddings = np.load(emb_path)
        with open(docs_path, 'rb') as f:
            docs = pickle.load(f)
    else:
        print(f"Generating embeddings for SKU: {sku} (this may take a minute)...")
        embeddings = []
        for chunk in tqdm(chunks, desc="Embedding review chunks", unit="chunk"):
            emb = get_text_embedding(chunk)
            embeddings.append(emb)
        embeddings = np.stack(embeddings)
        docs = chunks

        # Save to cache
        np.save(emb_path, embeddings)
        with open(docs_path, 'wb') as f:
            pickle.dump(docs, f)
        print(f"Saved {len(embeddings)} embeddings to cache.")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, docs


def retrieve_documents(query: str, docs: list, index: faiss.IndexFlatL2, k: int = 3) -> str:
    """
    Retrieve top-k most similar document chunks for the query.
    Returns the concatenated text of those chunks.
    """
    q_emb = get_text_embedding(query).reshape(1, -1)
    _, idxs = index.search(q_emb, k)
    return " ".join(docs[i] for i in idxs[0])


def run_mistral(prompt: str) -> str:
    """
    Send the prompt to Mistral Chat and return the generated response.
    """
    messages = [{"role": "user", "content": prompt}]
    resp = client.chat(model=model_name, messages=messages)
    return resp.choices[0].message.content


def run_rag_on_reviews(product_url: str, query: str, top_k: int = 3) -> (str, str):
    """
    Full RAG workflow for reviews:
      1. Extract SKU
      2. Scrape/load reviews
      3. Split reviews into chunks
      4. Load/build FAISS index (cached)
      5. Retrieve context
      6. Call Mistral for answer
      Returns (answer, retrieved_context).
    """
    sku = extract_sku_from_url(product_url)
    if not sku:
        raise ValueError("Invalid product URL: could not extract SKU.")

    _, df_reviews, _ = get_review_summary("CSV", sku)
    texts = df_reviews["text"].astype(str).tolist()
    chunks = chunk_reviews(texts)

    index, docs = get_or_build_index(sku, chunks)

    context = retrieve_documents(query, docs, index, k=top_k)
    prompt = (
        f"Answer the following question based on the provided customer reviews context.\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    answer = run_mistral(prompt)
    return answer, context

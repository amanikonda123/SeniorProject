import os
import numpy as np
import faiss
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from mistralai.client import MistralClient
from nltk.tokenize import sent_tokenize
import nltk

# Try to import RAGAS metrics
try:
    from ragas.metrics import (
        retrieval_precision,
        retrieval_recall,
        generation_relevance,
        generation_faithfulness,
        answer_correctness
    )
    ragas_available = True
except ImportError:
    print("RAGAS metrics not available. Skipping evaluation metrics.")
    ragas_available = False

# Ensure sentence tokenizer
nltk.download("punkt", quiet=True)
load_dotenv()

# Initialize Mistral client
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise RuntimeError("Please set MISTRAL_API_KEY in your environment.")
client = MistralClient(api_key=api_key)
model_name = "open-mistral-7b"

# --- Load HotpotQA Dataset ---
dataset = load_dataset("hotpot_qa", "distractor", split="train[:100]")  # small subset

# --- Prepare documents ---
documents = []
for example in dataset:
    ctx = example["context"]
    titles = " ".join(ctx.get("title", []))
    sentences = " ".join([" ".join(s) for s in ctx.get("sentences", [])])
    documents.append(f"{titles} {sentences}")

# --- Build or load embeddings cache ---
def build_or_load_index(docs, cache_file=".hotpotembeddings.npy"):
    # Invalidate if shape mismatch
    if os.path.exists(cache_file):
        emb_matrix = np.load(cache_file)
        if emb_matrix.shape[0] != len(docs):
            print(f"Cache size {emb_matrix.shape[0]} != docs {len(docs)}, rebuilding...")
            os.remove(cache_file)
    if not os.path.exists(cache_file):
        print("Embedding documents (this may take a while)...")
        embeddings = []
        for doc in tqdm(docs, desc="Embedding docs", unit="doc"):
            resp = client.embeddings(model="mistral-embed", input=[doc])
            embeddings.append(np.array(resp.data[0].embedding, dtype="float32"))
        emb_matrix = np.stack(embeddings)
        np.save(cache_file, emb_matrix)
        print(f"Saved {emb_matrix.shape[0]} embeddings to {cache_file}")
    else:
        emb_matrix = np.load(cache_file)
        print(f"Loaded embeddings from cache: {cache_file}")
    # Build FAISS index
    index = faiss.IndexFlatL2(emb_matrix.shape[1])
    index.add(emb_matrix)
    return index

index = build_or_load_index(documents)

# --- Retrieval & generation functions ---
def retrieve_documents(query, docs, idx, k=5):
    resp = client.embeddings(model="mistral-embed", input=[query])
    q_emb = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    _, inds = idx.search(q_emb, k)
    return [docs[i] for i in inds[0]]


def run_mistral(prompt):
    messages = [{"role": "user", "content": prompt}]
    resp = client.chat(model=model_name, messages=messages)
    return resp.choices[0].message.content

# --- Evaluation loop ---
if ragas_available:
    metrics = {"precision": [], "recall": [], "relevance": [], "faithfulness": [], "correctness": []}

for example in dataset:
    question = example["question"]
    gold = example.get("answer", "")
    support = example.get("supporting_facts", [])  # list of [doc_idx, sent]

    retrieved = retrieve_documents(question, documents, index, k=5)
    context = " ".join(retrieved)
    prompt = f"Answer based on this context:\n{context}\n\nQuestion: {question}\nAnswer:"
    gen = run_mistral(prompt)

    if ragas_available:
        # gold_sents is list of ground-truth supporting sentences
        gold_sents = [sent for _, sent in support]
        rp = retrieval_precision(retrieved, gold_sents)
        rr = retrieval_recall(retrieved, gold_sents)
        gr = generation_relevance(gen, question, context)
        gf = generation_faithfulness(gen, context)
        ac = answer_correctness(gen, gold)
        metrics["precision"].append(rp)
        metrics["recall"].append(rr)
        metrics["relevance"].append(gr)
        metrics["faithfulness"].append(gf)
        metrics["correctness"].append(ac)

# --- Sample output ---
print("=== Sample RAG Output ===")
p0 = dataset[0]
retr0 = retrieve_documents(p0["question"], documents, index, k=5)
print("Question:", p0["question"])
print("Retrieved:", retr0)
print("Answer:", run_mistral(f"Answer based on this context:\n{' '.join(retr0)}\n\nQuestion: {p0['question']}\nAnswer:"))

if ragas_available:
    import numpy as _np
    print("\n=== Averaged RAGAS Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {_np.mean(v):.4f}")

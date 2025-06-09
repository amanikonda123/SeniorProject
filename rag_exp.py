# Install required packages (if not already installed)
!pip install datasets mistralai faiss-cpu python-dotenv

# Load environment variables from .env
from dotenv import load_dotenv
import os
load_dotenv()  # This will load variables from a .env file in the current directory

# Import necessary libraries
import numpy as np
import faiss
from datasets import load_dataset
from mistralai.client import MistralClient, ChatMessage

# Initialize Mistral client (ensure your .env file contains MISTRAL_API_KEY)
client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
model_name = "open-mistral-7b"  # Generative model name for chat responses

# Define a function to generate text embeddings using Mistral's embedding endpoint
def get_text_embedding(text):
    response = client.embeddings(model="mistral-embed", input=[text])
    embedding = np.array(response.data[0].embedding).astype("float32")
    return embedding

# --- Load and Prepare the HotpotQA Dataset ---
# We load a subset of the training data for demonstration.
dataset = load_dataset("hotpot_qa", "distractor", split="train[:1000]")

# Create documents by concatenating the list of context paragraphs.
documents = []
for example in dataset:
    context_text = " ".join(example["context"])
    documents.append(context_text)

print("Loaded", len(documents), "documents.")

# --- Generate Embeddings and Build FAISS Index ---
# Generate an embedding for each document.
embeddings = np.stack([get_text_embedding(doc) for doc in documents])

# Build a FAISS vector index using L2 (Euclidean) distance.
d = embeddings.shape[1]  # Dimension of the embeddings
index = faiss.IndexFlatL2(d)
index.add(embeddings)
print("FAISS index built with", index.ntotal, "vectors.")

# --- Define the Retrieval Function ---
def retrieve_documents(query, documents, index, k=3):
    """
    Embed the query, search for the top-k most similar documents,
    and return the concatenated text of these documents.
    """
    q_embedding = get_text_embedding(query)
    q_embedding = np.array([q_embedding])
    distances, indices = index.search(q_embedding, k)
    retrieved_text = " ".join([documents[i] for i in indices[0]])
    return retrieved_text

# --- Define the Mistral Chat Call ---
def run_mistral(prompt):
    """
    Send the prompt to the Mistral Chat API and return the response.
    """
    messages = [ChatMessage(role="user", content=prompt)]
    response = client.chat(model=model_name, messages=messages)
    return response.choices[0].message.content

# --- Define the Standard RAG Pipeline ---
def standard_rag(query, documents, index):
    """
    Retrieve context based on the query and then generate an answer
    using the Mistral Chat API.
    """
    retrieved_context = retrieve_documents(query, documents, index, k=3)
    prompt = (
        f"Answer the following question based on the provided context.\n\n"
        f"Context: {retrieved_context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    return run_mistral(prompt)

# --- Test the Pipeline ---
query = "What is the favorite color of the women who invented 'Troopers'?"
answer = standard_rag(query, documents, index)
print("Query:", query)
print("Answer:", answer)

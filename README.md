# Walmart E-Commerce Customer Review Engine

A Streamlit application to explore and summarize Walmart product reviews using Retrieval-Augmented Generation (RAG).  
Supports four RAG strategies and an end-to-end evaluation suite.

---

## Features

1. **Review Summarization**

   - Scrape customer reviews by SKU or URL.
   - Handle long texts with token counting and chunking.
   - Generate summaries via Google Gemini or Mistral models.

2. **RAG Chatbot**

   - **Standard (Single-Hop):** Fixed-k snippet retrieval.
   - **Multi-Hop:** Iterative retrieval & answering over multiple hops.
   - **Adaptive:** Threshold-driven dynamic retrieval.
   - **Hybrid:** Combines adaptive breadth + focused depth, then synthesizes.
   - Stateful Streamlit UI with chat history, context expander, and mode selector.

3. **Evaluation Suite**
   - Built-in RAGAS integration for automated metrics.
   - **Retrieval:** nDCG@5
   - **Generation:** BERTScore (F1)
   - **Factuality:** QA-consistency check
   - **Efficiency:** Context-token usage statistics
   - Example evaluation on custom “creatine” reviews and HotpotQA benchmark.

## Setup

### 1. Clone & Enter the Repository

```bash
git clone https://github.com/amanikonda123/SeniorProject.git
cd SeniorProject
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Configure Environment Variables

Create a .env file in the project root with:

```bash
GOOGLE_API_KEY=<your Google API key>
MISTRAL_API_KEY=<your Mistral API key>
HUGGINGFACE_TOKEN=<optional for private HF models>
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run the App

```bash
streamlit run app.py
```

### 1. Enter a Walmart product URL

### 2. Load or cache reviews

### 2. Select a RAG mode

### 3. Ask questions; view answers and retrieved context

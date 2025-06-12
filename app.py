import os
import re
import streamlit as st
from dotenv import load_dotenv
from customer_reviews.reviews_summary import get_review_summary
from rag_exp import chunk_reviews, get_or_build_index, retrieve_documents, run_mistral

# Load environment variables
load_dotenv()

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Helper to extract SKU from a Walmart URL
def extract_sku_from_url(url: str) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    segments = [seg for seg in parsed.path.split('/') if seg]
    if segments and segments[-1].isdigit():
        return segments[-1]
    match = re.search(r"/(\d{5,})(?:$|\?)", url)
    return match.group(1) if match else ""

# Streamlit App Layout
st.title("Walmart E-Commerce Customer Review Engine")
st.header("Provide the URL, then ask questions about these reviews")

# Step 1: URL form to load reviews and build index
if "df_reviews" not in st.session_state:
    with st.form("url_form"):
        product_url = st.text_input(
            "Walmart product URL",
            placeholder="https://www.walmart.com/ip/.../17235783"
        )
        scrape_btn = st.form_submit_button("Load Reviews")
    if scrape_btn:
        sku = extract_sku_from_url(product_url)
        if not sku:
            st.error("Couldn't extract SKU. Check the URL and try again.")
            st.stop()
        # Scrape reviews
        with st.spinner("Scraping reviews…"):
            tokens, df, summary = get_review_summary("CSV", sku)
        # Store in session
        st.session_state.product_url = product_url
        st.session_state.df_reviews   = df
        st.session_state.summary      = summary
        # Build or load cached FAISS index
        with st.spinner("Embedding reviews and building index…"):
            texts  = df["text"].astype(str).tolist()
            chunks = chunk_reviews(texts)
            index, docs = get_or_build_index(sku, chunks)
        st.session_state.rag_index = index
        st.session_state.rag_docs  = docs

# Step 2: Display summary and chat UI once loaded
if "df_reviews" in st.session_state:
    summary     = st.session_state.summary
    product_url = st.session_state.product_url

    st.markdown("#### Initial Summary of Reviews")
    st.write(summary["text"])

    st.markdown("---")
    st.markdown("### Conversation History")
    for i, (q, a) in enumerate(st.session_state.history, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")

    st.markdown("---")
    # Query form
    with st.form("query_form"):
        question = st.text_input("Ask a question about these reviews:")
        ask_btn  = st.form_submit_button("Submit")
    if ask_btn and question:
        # Build retrieval query including history
        history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.history])
        retrieval_query = (history_text + "\n" + question) if history_text else question
        # Retrieve context
        with st.spinner("Retrieving relevant reviews…"):
            context = retrieve_documents(retrieval_query, st.session_state.rag_docs, st.session_state.rag_index)
        # Generate answer
        prompt = (
            f"You are a helpful assistant.\n"
            f"{history_text}\n"
            f"Context: {context}\n"
            f"Q: {question}\nA:"
        )
        with st.spinner("Generating answer…"):
            answer = run_mistral(prompt)
        # Store in history
        st.session_state.history.append((question, answer))
        # Display new turn
        st.markdown("**Q:** " + question)
        st.markdown("**A:** " + answer)

    # Clear history button
    if st.button("Clear Conversation History"):
        st.session_state.history = []

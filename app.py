# app.py

import os
import re
import streamlit as st
from dotenv import load_dotenv
from customer_reviews.reviews_summary import get_review_summary
from rag_exp import (
    extract_sku_from_url,
    run_standard_rag_on_reviews,
    run_multihop_rag_on_reviews,
    run_adaptive_rag_on_reviews,
    run_hybrid_rag_on_reviews,
)

load_dotenv()

st.title("Walmart E-Commerce Customer Review Engine")
st.header("Provide the URL, then ask questions about these reviews")

if "history" not in st.session_state:
    st.session_state.history = []
if "multi_contexts" not in st.session_state:
    st.session_state.multi_contexts = []

# Step 1: load reviews
with st.form("url_form"):
    product_url = st.text_input(
        "Walmart product URL",
        placeholder="https://www.walmart.com/ip/.../17235783"
    )
    load_btn = st.form_submit_button("Load Reviews")
if load_btn and product_url:
    sku = extract_sku_from_url(product_url)
    if not sku:
        st.error("Could not extract SKU.")
        st.stop()
    with st.spinner("Scraping reviews…"):
        tokens, df_reviews, summary = get_review_summary("CSV", sku)
    st.session_state.product_url = product_url
    st.session_state.df_reviews = df_reviews
    st.session_state.summary = summary
    st.session_state.history = []
    st.session_state.multi_contexts = []

# Step 2: display & chat
if "df_reviews" in st.session_state:
    st.markdown("#### Initial Summary of Reviews")
    st.write(st.session_state.summary["text"])

    st.markdown("#### Top 5 Reviews")
    st.dataframe(st.session_state.df_reviews.head(5), hide_index=True)

    st.markdown("---")
    mode = st.radio(
        "Select RAG Mode:",
        ["Standard", "Multi-Hop", "Adaptive", "Hybrid"]
    )

    st.markdown("### Conversation History")
    for i, (q, a) in enumerate(st.session_state.history, 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")

    st.markdown("---")
    with st.form("query_form"):
        question = st.text_input("Ask a question about these reviews:")
        ask_btn = st.form_submit_button("Submit")
    if ask_btn and question:
        if mode == "Standard":
            answer, context = run_standard_rag_on_reviews(
                st.session_state.product_url, question
            )
        elif mode == "Multi-Hop":
            answer, contexts = run_multihop_rag_on_reviews(
                st.session_state.product_url, question
            )
            st.session_state.multi_contexts.extend(contexts)
            context = "\n---\n".join(st.session_state.multi_contexts)
        elif mode == "Adaptive":
            answer, context = run_adaptive_rag_on_reviews(
                st.session_state.product_url, question
            )
        else:  # Hybrid
            answer, contexts = run_hybrid_rag_on_reviews(
                st.session_state.product_url, question
            )
            st.session_state.multi_contexts.extend(contexts)
            context = "\n---\n".join(st.session_state.multi_contexts)

        st.session_state.history.append((question, answer))

        with st.expander("🔍 Retrieved Context"):
            st.write(context)

        idx = len(st.session_state.history)
        st.markdown(f"**A{idx}:** {answer}")

    if st.button("Clear Conversation History"):
        st.session_state.history = []
        st.session_state.multi_contexts = []

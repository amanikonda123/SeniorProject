import os
import re
import time
import streamlit as st
import pandas as pd
from apify_client import ApifyClient
from customer_reviews.reviews_summary import get_review_summary
from customer_reviews.rag_chatbot import chat_with_reviews

# --- Helper to extract SKU ---
def extract_sku_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        segments = [seg for seg in parsed.path.split('/') if seg]
        if segments and segments[-1].isdigit():
            return segments[-1]
        match = re.search(r"/(\d{5,})(?:$|\?)", url)
        return match.group(1) if match else ""
    except:
        return ""

# --- Streamlit UI ---
st.title("Walmart E-Commerce Customer Review Engine")
st.header("Please provide the requested information")

with st.form("user_interaction"):
    product_url = st.text_input(
        "Please input your Walmart product URL:",
        placeholder="https://www.walmart.com/ip/.../17235783"
    )
    submit_button = st.form_submit_button("Submit")

if submit_button:
    sku = extract_sku_from_url(product_url)
    if not sku:
        st.error("Could not extract SKU from the provided URL. Please check the URL and try again.")
        st.stop()

    with st.spinner("Scraping reviews..."):
        tokens, df_reviews, summary = get_review_summary(
            "CSV", sku
        )
    st.success(f"Scraped {len(df_reviews)} reviews successfully!")

    # Download button for full reviews CSV
    csv = df_reviews.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download All Reviews as CSV",
        data=csv,
        file_name=f"{sku}_reviews.csv",
        mime="text/csv"
    )

    # Show a preview
    st.markdown("### Top 10 Customer Reviews:")
    st.dataframe(df_reviews.head(10), hide_index=True)

    # Display summaries
    st.markdown("### Customer reviews summary:")
    st.write(summary["text"])

    # st.markdown("---")
    # st.markdown("## Chat with the reviews")
    # question = st.text_input("Ask a question about these reviews:")
    # if question:
    #     with st.spinner("Thinking…"):
    #         answer = chat_with_reviews(sku, question, top_k=5)
    #     st.markdown("### Answer:")
    #     st.write(answer)
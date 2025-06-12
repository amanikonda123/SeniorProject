# import os
# import tiktoken
# import numpy as np
# import pandas as pd
# from typing import Any
# from dotenv import load_dotenv
# from string import ascii_lowercase
# from langchain_openai import ChatOpenAI
# from langchain.llms import VertexAI
# from langchain.prompts import PromptTemplate
# from langchain.docstore.document import Document
# from customer_reviews.walmart_scraper import WalmartScraper
# from langchain.chains import LLMChain, StuffDocumentsChain
# from langchain.chains.summarize import load_summarize_chain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
# import streamlit as st
# from io import StringIO
# import sys

# tmp = sys.stdout
# my_result = StringIO()
# sys.stdout = my_result
# load_dotenv()

# # importing api keys and initiate llm
# VertexAI.model_rebuild()
# llm = VertexAI(
#     model_name="gemma-2-medium",           # 12 B Gemma 2
#     model_kwargs={"temperature": 0}
# )

# # Counting AutoScraper output tokens
# def count_tokens(string: str, encoding_name: str) -> int:
#     """Returns the number of tokens in a text string."""

#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens

# # generate review summary for smaller revieww
# def small_reviews_summary(cust_reviews: str) -> str:
#     summary_statement = """You are an expeienced copy writer providing a world-class summary of product reviews {cust_reviews} from numerous customers \
#                         on a given product from different leading e-commerce platforms. You write summary of all reviews for a target audience \
#                         of wide array of product reviewers ranging from a common man to an expeirenced product review professional."""
#     summary_prompt = PromptTemplate(input_variables = ["cust_reviews"], template=summary_statement)
#     llm_chain = LLMChain(llm=llm, prompt=summary_prompt)
#     review_summary = llm_chain.invoke(cust_reviews)
#     return review_summary

# # split large reviews
# def document_split(cust_reviews: str, chunk_size: int, chunk_overlap: int) -> Any:
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""])
    
#     # converting string into a document object
#     docs = [Document(page_content = t) for t in cust_reviews.split('\n')]
#     split_docs = text_splitter.split_documents(docs)
#     return split_docs

# # Applying map reduce to summarize large document
# def map_reduce_summary(split_docs: Any) -> str: 
#     map_template = """Based on the following docs {docs}, please provide summary of reviews presented in these documents. 
#     Review Summary is:"""

#     map_prompt = PromptTemplate.from_template(map_template)
#     map_chain = LLMChain(llm=llm, prompt=map_prompt)

#     # Reduce
#     reduce_template = """The following is set of summaries: 
#     {doc_summaries}
#     Take these document and return your consolidated summary in a professional manner addressing the key points of the customer reviews. 
#     Review Summary is:"""
#     reduce_prompt = PromptTemplate.from_template(reduce_template)

#     # Run chain
#     reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

#     # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
#     combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="doc_summaries")

#     # Combines and iteratively reduces the mapped documents
#     reduce_documents_chain = ReduceDocumentsChain(
#         # This is final chain that is called.
#         combine_documents_chain=combine_documents_chain,
#         # If documents exceed context for `StuffDocumentsChain`
#         collapse_documents_chain=combine_documents_chain,
#         # The maximum number of tokens to group documents into.
#         token_max=3500,
#     )

#     # Combining documents by mapping a chain over them, then combining results
#     map_reduce_chain = MapReduceDocumentsChain(
#         # Map chain
#         llm_chain=map_chain,
#         # Reduce chain
#         reduce_documents_chain=reduce_documents_chain,
#         # The variable name in the llm_chain to put the documents in
#         document_variable_name="docs",
#         # Return the results of the map steps in the output
#         return_intermediate_steps=False,
#     )
    
#     # generating review summary for map reduce method
#     cust_review_summary_mr = map_reduce_chain.invoke(split_docs)

#     return cust_review_summary_mr

# # Applying refine method to summarize large document
# def refine_method_summary(split_docs) -> str:
#     prompt_template = """
#                   Please provide a summary of the following text.
#                   TEXT: {text}
#                   SUMMARY:
#                   """

#     question_prompt = PromptTemplate(
#         template=prompt_template, input_variables=["text"]
#     )

#     refine_prompt_template = """
#                 Write a concise summary of the following text delimited by triple backquotes.
#                 Return your response in that covers the key points of the text.
#                 ```{text}```
#                 BULLET POINT SUMMARY:
#                 """

#     refine_prompt = PromptTemplate(
#         template=refine_prompt_template, input_variables=["text"])

#     # Load refine chain
#     chain = load_summarize_chain(
#         llm=llm,
#         chain_type="refine",
#         question_prompt=question_prompt,
#         refine_prompt=refine_prompt,
#         return_intermediate_steps=False,
#         input_key="input_text",
#     output_key="output_text",
#     )
    
#     # generating review summary using refine method
#     cust_review_summary_refine = chain.invoke({"input_text": split_docs}, return_only_outputs=True)
#     return cust_review_summary_refine


# def get_review_summary(inp_opt: str, prod_sku: str) -> Any:
#     # only URL mode supported
#     csv_file = f"{prod_sku}_reviews.csv"
#     if os.path.exists(csv_file):
#         df = pd.read_csv(csv_file)
#         st.write(f"Loaded {len(df)} reviews from local CSV: {csv_file}")
#     else:
#         # otherwise, scrape and save
#         scraper = WalmartScraper()
#         reviews = scraper.get_reviews(prod_sku)
#         df = pd.DataFrame(reviews)
#         df.to_csv(csv_file, index=False)
#         st.write(f"Scraped and saved {len(df)} reviews to {csv_file}")

#     # csv_filename = f"{prod_sku}_reviews.csv"
#     # df.to_csv(csv_filename, index=False)
#     # st.write(f"Saved all reviews to `{csv_filename}` on disk")


#     # 2) Concatenate all review texts and count tokens
#     reviews_str = "\n".join(df["text"].astype(str).tolist())
#     total_tokens = count_tokens(reviews_str, "cl100k_base")

#     # 3) Branch on length: small vs. map‐reduce + refine
#     if total_tokens <= 3500:
#         summary_small = {"text": small_reviews_summary(reviews_str)}
#         summary_map    = {"output_text": "N.A."}
#         summary_refine = {"output_text": "N.A."}
#     else:
#         docs = document_split(reviews_str, chunk_size=1000, chunk_overlap=50)
#         summary_map    = {"output_text": map_reduce_summary(docs)}
#         summary_refine = {"output_text": refine_method_summary(docs)}
#         summary_small  = {"text": "N.A."}

#     # 4) Return token count, DataFrame, and all summaries
#     return total_tokens, df, summary_small, summary_map, summary_refine

# summarizer.py

import os
from dotenv import load_dotenv
import tiktoken
import pandas as pd
from typing import Tuple, List, Dict, Any

from google import genai
from google.genai import types
from customer_reviews.walmart_scraper import WalmartScraper

# 1) Load & configure your API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY in your environment or .env file")

client = genai.Client(api_key=API_KEY)

# 2) Utility: count tokens
def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

# 3) Utility: break text into ~40 000-char chunks
def chunk_text(text: str, max_chars: int = 40_000) -> List[str]:
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

# 4) Summarize via Gemini, passing sampling params in a config object
def gemini_summary(text: str, max_output_tokens: int = 500) -> str:
    partials: List[str] = []
    for chunk in chunk_text(text):
        resp = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=chunk,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=max_output_tokens,
            ),
        )
        partials.append(resp.text.strip())

    # If more than one chunk, combine & refine
    if len(partials) > 1:
        combined = "\n\n".join(partials)
        resp = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=(
                "Combine and polish these partial summaries into a single cohesive summary. Provide a pro/con list. Begin the response directly with the summary. Here are the partial reviews:\n\n"
                + combined
            ),
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=max_output_tokens,
            ),
        )
        return resp.text.strip()

    return partials[0]

# 5) Main entrypoint — same signature your Streamlit app expects
def get_review_summary(
    inp_opt: str,
    prod_sku: str
) -> Tuple[int, pd.DataFrame, Dict[str, Any]]:
    """
    inp_opt: "CSV" to load from cache if present, otherwise scrape
             anything else (e.g. "SKU") to always scrape.
    prod_sku: Walmart product SKU or identifier.
    Returns: (token_count, DataFrame, summary)
    """
    csv_file = f"{prod_sku}_reviews.csv"

    # a) Load or scrape
    if inp_opt.upper() == "CSV" and os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        reviews = WalmartScraper().get_reviews(prod_sku)
        df = pd.DataFrame(reviews)
        df.to_csv(csv_file, index=False)

    # b) Concatenate review text & count tokens
    text = "\n".join(df["text"].astype(str).tolist())
    total_tokens = count_tokens(text)

    # c) Summarize via Gemini
    summary_text = gemini_summary(text)

    # d) Package outputs to match your old signature
    summary  = {"text": summary_text}

    return total_tokens, df, summary

# 6) Optional CLI for local testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch & summarize Walmart reviews via Gemini"
    )
    parser.add_argument(
        "inp_opt",
        choices=["CSV", "SKU"],
        help="Use cached CSV if present ('CSV') or always scrape ('SKU')",
    )
    parser.add_argument("prod_sku", help="Walmart product SKU or identifier")
    args = parser.parse_args()

    tokens, df, summary = get_review_summary(args.inp_opt, args.prod_sku)
    print(f"Token count: {tokens}")
    print("\n=== Summary ===\n", summary["text"])

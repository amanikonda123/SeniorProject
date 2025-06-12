# evaluate_rag.py

import os
import tiktoken
import pandas as pd
from dotenv import load_dotenv

# RAGAS imports
from ragas import SingleTurnSample, EvaluationDataset, evaluate  # :contentReference[oaicite:0]{index=0}
from ragas.metrics import context_precision, context_recall, answer_relevancy, faithfulness  # :contentReference[oaicite:1]{index=1}

# Your RAG functions
from rag_exp import (
    run_standard_rag_on_reviews,
    run_multihop_rag_on_reviews,
    run_adaptive_rag_on_reviews,
    run_hybrid_rag_on_reviews,
)

# 1) Load env (API keys, etc)
load_dotenv()

# 2) Define your evaluation cases
eval_cases = [
    {
        "sku": "17235783",
        "question": "What do customers say about mixability?",
        "reference": (
            "Most users report the powder dissolves completely in water or shakes out smooth "
            "without clumps, even with minimal stirring."
        ),
    },
    {
        "sku": "17235783",
        "question": "Are any side effects mentioned?",
        "reference": (
            "A small fraction mention mild stomach discomfort or bloating, "
            "particularly when taken on an empty stomach."
        ),
    },
    {
        "sku": "17235783",
        "question": "How is the taste described?",
        "reference": (
            "Reviewers generally call the flavor neutral to slightly chalky, "
            "with many noting it’s easy to mask in juice."
        ),
    },
    {
        "sku": "17235783",
        "question": "Do customers feel it’s good value for money?",
        "reference": (
            "Most feel the price is reasonable for the quality, though a few "
            "wish there were larger tub sizes available."
        ),
    },
    {
        "sku": "17235783",
        "question": "What improvements in recovery do users report?",
        "reference": (
            "Many state they experienced reduced muscle soreness and faster recovery "
            "after workouts compared to previous products."
        ),
    },
    # …add as many cases as you like…
]

# 3) Build a list of SingleTurnSample instances
samples = []
enc = tiktoken.get_encoding("cl100k_base")
for case in eval_cases:
    sku       = case["sku"]
    question  = case["question"]
    reference = case["reference"]

    for model_name, fn in [
        ("standard", run_standard_rag_on_reviews),
        ("multihop", run_multihop_rag_on_reviews),
        ("adaptive", run_adaptive_rag_on_reviews),
        ("hybrid", run_hybrid_rag_on_reviews),
    ]:
        # call the RAG variant
        if model_name in ("multihop", "hybrid"):
            pred, contexts = fn(sku, question)
            retrieved_contexts = contexts
        else:
            pred, ctx = fn(sku, question)
            retrieved_contexts = [ctx]

        # record token‐count (optional, for efficiency analysis)
        token_count = sum(len(enc.encode(c)) for c in retrieved_contexts)

        # create the sample
        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=retrieved_contexts,
                response=pred,
                reference=reference,
                metadata={"model": model_name, "ctx_tokens": token_count},
            )
        )

# 4) Assemble into an EvaluationDataset
dataset = EvaluationDataset(samples=samples)  # :contentReference[oaicite:2]{index=2}

# 5) Run RAGAS evaluation
results = evaluate(
    dataset,
    metrics=[
        context_precision,   # how precise is your retrieval? :contentReference[oaicite:3]{index=3}
        context_recall,      # how much relevant context did you recover? :contentReference[oaicite:4]{index=4}
        answer_relevancy,    # semantic match between answer & query :contentReference[oaicite:5]{index=5}
        faithfulness,        # are you grounded / non-hallucinating? :contentReference[oaicite:6]{index=6}
    ],
)

# 6) Inspect & save
# Convert the per-sample scores to a DataFrame
scores_df = results.scores.to_pandas()
print("\n=== Detailed Scores ===")
print(scores_df)

# Optionally, aggregate by model
agg = (
    scores_df
    .groupby("metadata.model")
    .agg({
        "context_precision": "mean",
        "context_recall":    "mean",
        "answer_relevancy":  "mean",
        "faithfulness":      "mean",
        "metadata.ctx_tokens": ["mean","std"]
    })
    .round(3)
)
print("\n=== Summary by Model ===")
print(agg)

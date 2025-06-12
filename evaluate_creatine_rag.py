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
# eval_cases.py

sku = "17235783"

qa_pairs = [
    ("What do customers say about mixability?",
     "Most users report the powder dissolves completely in water or shakes out smooth without clumps, even with minimal stirring."),
    ("Are any side effects mentioned?",
     "A small fraction mention mild stomach discomfort or bloating, particularly when taken on an empty stomach."),
    ("How is the taste described?",
     "Reviewers generally call the flavor neutral to slightly chalky, with many noting it’s easy to mask in juice."),
    ("Do customers feel it’s good value for money?",
     "Most feel the price is reasonable for the quality, though a few wish there were larger tub sizes available."),
    ("What improvements in recovery do users report?",
     "Many state they experienced reduced muscle soreness and faster recovery after workouts compared to previous products."),
    ("How well does the powder dissolve in cold water?",
     "Most users report full dissolution in cold water within 30 seconds of shaking."),
    ("Is there any noticeable aftertaste?",
     "A minority mention a slight chalky aftertaste, but most say it’s barely perceptible."),
    ("What do customers say about texture or mouthfeel?",
     "Users describe a smooth, creamy texture with no graininess when mixed properly."),
    ("How do reviewers rate the packaging quality?",
     "Packaging is generally praised for being sturdy with an easy-to-open seal."),
    ("Are shipping times mentioned positively?",
     "Most buyers report receiving their order within 3–5 business days."),
    ("Did anyone report issues with clumping?",
     "Very few mention clumping; those who did suggest using a blender bottle."),
    ("How clear are the dosage instructions?",
     "Reviewers find the dosing instructions printed on the label easy to follow."),
    ("What do customers say about product smell?",
     "Most say the powder has a neutral odor, with no strong or off-putting scents."),
    ("How accurate is the label information?",
     "Users overwhelmingly trust the label’s ingredient list and dosage claims."),
    ("Is the product suitable for vegans?",
     "Multiple buyers confirm it’s vegan-friendly, containing no animal-derived ingredients."),
    ("Do reviews mention any allergic reactions?",
     "Allergic reactions are extremely rare; only one user reported mild itching."),
    ("How do users describe muscle gains?",
     "Many state they noticed noticeable strength gains within 4–6 weeks of use."),
    ("What do customers say about energy levels?",
     "Some report a modest boost in workout energy when taken pre-exercise."),
    ("Is there feedback on mixing with milk?",
     "Mixing with milk works fine, though a few note it takes slightly longer to dissolve."),
    ("Do users recommend it for beginners?",
     "Several first-time creatine users say it’s a great starter product."),
    ("What do reviewers say about repeat purchases?",
     "A high percentage indicate they have already reordered or plan to."),
    ("How do users feel it compares to competitors?",
     "Many feel it’s on par with premium brands at a lower price point."),
    ("Any comments on mixing without a blender?",
     "Users say a standard shaker bottle is sufficient, no blender needed."),
    ("Do customers mention product color consistency?",
     "Most observe a uniform white color across all batches."),
    ("Is there feedback on shelf life or expiration?",
     "Buyers report no change in quality up to the printed expiration date."),
    ("How do reviewers rate overall satisfaction?",
     "Overall satisfaction is high, with an average rating above 4.5 stars."),
    ("What do customers say about price fluctuations?",
     "Some note periodic sales, but most feel the regular price is fair."),
    ("Any complaints about the scoop size?",
     "A few suggest the included scoop is slightly small but functional."),
    ("How do users describe solubility in juice?",
     "Juice masking is effective; powder dissolves well in most fruit juices."),
    ("What do reviewers say about strength improvements?",
     "Many report lifting heavier weights after consistent use."),
    ("Is there feedback on endurance during workouts?",
     "Some mention slightly longer endurance during high-rep sets."),
    ("How do customers rate the pro/con list feature?",
     "Users find the automated pro/con list helpful for quick decision-making."),
    ("What do reviewers say about mail-in shipping costs?",
     "A few note shipping fees, but most feel ordering in bulk offsets the cost."),
    ("Any mention of GMO-free or organic claims?",
     "Users appreciate the non-GMO label, though it’s not certified organic."),
    ("What do customers say about brand reputation?",
     "The brand is frequently praised for quality and transparency."),
    ("Are there any mentions of vegan versus non-vegan blends?",
     "Some clarify they chose this product specifically because it’s vegan."),
    ("What feedback is there on customer service?",
     "A few users contacted support with questions and rate responses as prompt."),
    ("Do reviews mention mixing temperature?",
     "Most say room-temperature water works best for quick dissolution."),
    ("How is the ingredient transparency perceived?",
     "Buyers praise the clear ingredient list and lack of proprietary blends."),
    ("Are there comments on texture when made into a paste?",
     "A few tried making a paste and found it smooth and lump-free."),
    ("What do users say about label font size and readability?",
     "Most find the label text legible, even for small print."),
    ("Do customers report any bloating after use?",
     "Bloating is rarely mentioned, and usually only when over-dosed."),
    ("What do reviewers say about subscription discounts?",
     "Subscribers enjoy a 10% discount and report reliable deliveries."),
    ("Is there feedback on mixing in smoothies?",
     "Smoothies mask taste well and blend the powder evenly."),
    ("What do customers say about label accuracy over time?",
     "No discrepancies have been noted between batches and label claims."),
    ("Are there any complaints about scoop residue?",
     "Some mention powder sticking to the scoop, but it’s easily tapped off."),
    ("What do reviewers say about flavor variety?",
     "Only unflavored is available; customers who prefer flavored products wish there were more options."),
    ("How do users describe anabolic effects?",
     "Several note visible improvements in lean mass over a month."),
    ("Is there feedback on product stability in high humidity?",
     "Most say it stays free-flowing even when stored in humid conditions."),
    ("What do reviewers say about mixing frequency recommendations?",
     "Users follow the recommended once-daily mix without issues."),
    ("Do customers mention recommended cycling protocols?",
     "A few advise cycling off after 8–12 weeks for best results."),
    ("How do buyers rate the overall packaging design?",
     "Packaging design is noted as sleek and gym-appropriate by many users."),
    ("What improvements in workout performance are reported?",
     "Users cite stronger lifts and better pump during sessions."),
]

eval_cases = [
    {"sku": sku, "question": q, "reference": r}
    for q, r in qa_pairs
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

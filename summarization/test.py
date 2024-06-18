from pathlib import Path

from tira.rest_api_client import Client

from rouge_score import rouge_scorer

import pandas as pd

tira = Client()

predictions = pd.read_json("predictions.jsonl", lines=True).set_index("id")

summaries = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")
summaries.index = summaries.index.astype("int64")

# join predictions with summaries
summaries = summaries.join(predictions, rsuffix="_prediction")

rouge1, rouge2, rougeL = 0, 0, 0
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
for r in summaries.iterrows():
    row = r[1]
    scores = scorer.score(row["summary"], row["summary_prediction"])
    rouge1 += scores["rouge1"].fmeasure
    rouge2 += scores["rouge2"].fmeasure
    rougeL += scores["rougeL"].fmeasure

rouge1 /= len(summaries)
rouge2 /= len(summaries)
rougeL /= len(summaries)

print(f"ROUGE-1: {rouge1}")
print(f"ROUGE-2: {rouge2}")
print(f"ROUGE-L: {rougeL}")

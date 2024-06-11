from pathlib import Path

from word_similarity import word_similarity

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")

    # Compute the similarity
    THRESHOLD = 0.464
    df["similarity"] = word_similarity(df)
    df["label"] = (df["similarity"] > THRESHOLD).astype(int)
    df = df.drop(columns=["similarity", "sentence1", "sentence2"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

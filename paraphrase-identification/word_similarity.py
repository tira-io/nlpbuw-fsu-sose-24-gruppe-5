import pandas as pd

def word_similarity_sentence(sentence1: str, sentence2: str):
    set1 = set(sentence1.lower().split())
    set2 = set(sentence2.lower().split())
    return len(set1 & set2) / len(set1 | set2)

def word_similarity(df: pd.DataFrame):
    return df.apply(lambda x: word_similarity_sentence(x.iloc[0], x.iloc[1]), axis=1)
        

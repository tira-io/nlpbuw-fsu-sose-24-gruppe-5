import pandas as pd

def word_similarity_sentence(sentence1: str, sentence2: str):
    set1 = set(sentence1.lower().split())
    set2 = set(sentence2.lower().split())
    return len(set1 & set2) / len(set1 | set2)

def get_words_with_count(sentence):
    words = sentence.lower().split()
    word_count = {}
    word_set = set()
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
        word_set.add((word, word_count[word]))
    return word_set

def word_similarity_sentence_v2(sentence1: str, sentence2: str):
    word_set1 = get_words_with_count(sentence1)
    word_set2 = get_words_with_count(sentence2)
    return len(word_set1 & word_set2) / len(word_set1 | word_set2)

def word_similarity(df: pd.DataFrame, v2=True):
    if v2:
        return df.apply(lambda x: word_similarity_sentence_v2(x.iloc[0], x.iloc[1]), axis=1)
    else:
        return df.apply(lambda x: word_similarity_sentence(x.iloc[0], x.iloc[1]), axis=1)
        

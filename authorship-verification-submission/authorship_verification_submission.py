from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

import re

if __name__ == "__main__":

    tira = Client()

    # loading train data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    text_train = pd.merge(text_train, targets_train, on="id", how="inner", validate="one_to_one")

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )

    def ngramsetloss(textlist, n):
        if n > 1:
            ngrams = [' '.join(textlist[i:i+n]) for i in range(len(textlist)-n+1)]
        else:
            ngrams = textlist
        if len(ngrams) == 0:
            return 1
        return len(set(ngrams))/len(ngrams)

    def get_words_of_text(text):
        # extract words from text
        return re.findall(r'\w+', text)

    def get_values_of_text(text):
        # extract words from text
        words = get_words_of_text(text)
        # calculate n-gram set loss for n=1 to 5
        results = [ngramsetloss(words, n) for n in range(1, 6)]
        return results
    
    def prediction_function(text, alpha=0.9974):
        words = get_words_of_text(text)
        loss = ngramsetloss(words, 5)
        if loss < alpha:
            return 0
        return 1

    testing_stuff_1 = False

    if testing_stuff_1:
        # add length of text to the training data
        text_train["length"] = text_train["text"].apply(lambda x: len(get_words_of_text(x)))

        # add n-gram set loss for n=1 to 5 to the training data
        results = [get_values_of_text(row[1]["text"]) for row in text_train.iterrows()]
        for i in range(5):
            text_train["ngramsetloss" + str(i+1)] = [result[i] for result in results]

        # draw a plot of the n-gram set loss for each text (n=1 to 5) (human=blue, ai=red)
        max_length = max(text_train["length"])
        alphas = []
        for row in text_train.iterrows():
            # brightness of color based on the length of the text
            if row[1]["length"] > 0 and max_length > 0:
                alpha = row[1]["length"]/max_length
            else:
                alpha = 0
            plt.plot(range(1, 6), row[1][4:], 'r' if row[1]["generated"] == 1 else 'b', alpha=alpha)
        for i in range(1, 6):
            values_ai = text_train[text_train["generated"] == 1]["ngramsetloss" + str(i)].values
            values_human = text_train[text_train["generated"] == 0]["ngramsetloss" + str(i)].values
            # plot as boxplot
            # plt.boxplot([values_human, values_ai], positions=[i-0.1, i+0.1], widths=0.2)
            plt.boxplot(values_human, positions=[i-0.1], widths=0.2, patch_artist=True, boxprops=dict(facecolor='blue'))
            plt.boxplot(values_ai, positions=[i+0.1], widths=0.2, patch_artist=True, boxprops=dict(facecolor='red'))
        plt.xlabel("n")
        plt.ylabel("n-gram set loss")
        # custom legend -> blue square=human, red square=ai
        plt.legend(handles=[plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='b', markersize=10, label='human'),
                            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='r', markersize=10, label='ai')], title="Author")
        plt.xticks(range(1, 6), labels=[str(i) for i in range(1, 6)])
        plt.show()
    
    testing_stuff_2 = False

    if testing_stuff_2:
        test_statistics = []
        for alpha_10000 in range(9800, 9990):
            alpha = alpha_10000/10000
            # calculate the prediction for the validation data
            prediction = text_train.set_index("id")["text"].apply(lambda x: prediction_function(x, alpha=alpha))
            accuracy = (prediction == text_train.set_index("id")["generated"]).sum()/len(prediction)
            # calculate the precision, recall and f1 score
            true_positives = ((prediction == 1) & (text_train.set_index("id")["generated"] == 1)).sum()
            false_positives = ((prediction == 1) & (text_train.set_index("id")["generated"] == 0)).sum()
            false_negatives = ((prediction == 0) & (text_train.set_index("id")["generated"] == 1)).sum()
            precision = true_positives/(true_positives+false_positives) if true_positives+false_positives > 0 else 0
            recall = true_positives/(true_positives+false_negatives) if true_positives+false_negatives > 0 else 0
            f1 = 2*(precision*recall)/(precision+recall) if precision+recall > 0 else 0
            print("alpha:", alpha, "accuracy:", accuracy, "precision:", precision, "recall:", recall, "f1:", f1)
            test_statistics.append((alpha, accuracy, precision, recall, f1))
        # plot
        test_statistics = pd.DataFrame(test_statistics, columns=["alpha", "accuracy", "precision", "recall", "f1"])
        plt.plot(test_statistics["alpha"], test_statistics["accuracy"], label="accuracy")
        plt.plot(test_statistics["alpha"], test_statistics["precision"], label="precision")
        plt.plot(test_statistics["alpha"], test_statistics["recall"], label="recall")
        plt.plot(test_statistics["alpha"], test_statistics["f1"], label="f1")
        plt.xlabel("alpha")
        plt.ylabel("score")
        plt.legend()
        plt.show()

    prediction = text_validation.set_index("id")["text"].apply(prediction_function)

    # converting the prediction to the required format
    prediction.name = "generated"
    prediction = prediction.reset_index()

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

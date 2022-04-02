from sklearn.naive_bayes import MultinomialNB
from utils import load_data, unique, calculate_avg_length, split_words_by_label,\
get_vocab_size, prob_Laplace_smoothing, accuracy, macro_F1
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from cleandata import total_cleaning
import numpy as np


if __name__ == '__main__':
    # Load data
    train_texts, train_labels = load_data('data/sst_train.csv')

    #divide the train dataset into 5 splits, train : validation = 4 : 1
    #valid_texts, valid_labels = train_texts[-int(len(train_texts)/5):], train_labels[-int(len(train_texts)/5):]
    #train_texts, train_labels = train_texts[:-int(len(train_texts)/5)], train_labels[:-int(len(train_texts)/5)]
    test_texts, test_labels = load_data('data/sst_test.csv')


    

    # Print basic statistics
    print("Training set size:", len(train_texts))
    #print("Validation set size:", len(valid_texts))
    print("Test set size:", len(test_texts))
    label_set = list(unique(train_labels))
    label_set.sort()
    print("Unique labels:", label_set)
    #print("Avg. length:", calculate_avg_length(train_texts + valid_texts + test_texts))

    print("Executing data cleaning!")
    #test_texts = total_cleaning(test_texts)
    #train_texts = total_cleaning(train_texts)
    

    # feature crafting using sklearn.feature_extraction
    vectorizer = CountVectorizer()
    train_X = vectorizer.fit_transform(train_texts)
    #valid_X = vectorizer.transform(valid_texts)
    test_X = vectorizer.transform(test_texts)

    #model training and fitting
    model = MultinomialNB()
    model.fit(train_X, train_labels)

    #predicting
    res = model.predict(test_X)

    #Evaluating
    print((res != test_labels).sum())
    label_num = len(label_set)
    TP = np.zeros((label_num,))
    TN = np.zeros((label_num, ))
    FP = np.zeros((label_num,))
    FN = np.zeros((label_num,))
    for index, label in enumerate(res):
        gold_label = test_labels[index]
        if label == gold_label:
            TP[label] += 1
            for l in label_set:
                TN[l] += 1
            TN[label] -= 1
        else:
            FP[label] += 1
            FN[gold_label] += 1
            for l in label_set:
                TN[l] += 1
            TN[label] -= 1
            TN[gold_label] -= 1

    print("TP: ", TP)
    print("TN: ", TN)
    print("FP: ", FP)
    print("FN: ", FN)

    Macro_f1 = 0
    for label in label_set:
        Macro_f1 += macro_F1(TP[label], FP[label], FN[label])
    Macro_f1 /= len(label_set)
    print("Macro-F1 score: ", Macro_f1)
    print(f1_score(test_labels, res, average = 'macro'))
    print("acuracy score: ", accuracy_score(test_labels, res))
    up = TP.sum()
    down = (TP.sum() + FP.sum() + TN.sum() + FN.sum()) / 5
    print(up / down)
        



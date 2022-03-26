from cgitb import text
import csv
from utils import load_data, unique, calculate_avg_length, split_words_by_label,\
get_vocab_size, prob_Laplace_smoothing, accuracy, macro_F1
from cleandata import total_cleaning
import numpy as np
from np_feature import get_feature_matrix, get_word_feature, training

def eval(prob_matrix, texts, labels, label_set, label_portion, word2id, unseen):

    print('start evaluation...')

    label_num = len(label_portion)
    vocab_size = len(prob_matrix[0])
    TP = np.zeros((label_num,))
    TN = np.zeros((label_num, ))
    FP = np.zeros((label_num,))
    FN = np.zeros((label_num,))

    #n * dim, n
    feature_matrix, unseen_word_num = get_feature_matrix(texts, word2id)
    for index, array in enumerate(feature_matrix):
        #print(index)
        # use log to switch multiplying to sum operation.
        # Question: How to handle log 0?
        midres = np.log(prob_matrix) * array
        midres = np.sum(midres, axis = 1) + np.log(unseen ** unseen_word_num[index])

        # strongly worsen the performance of sst-5
        midres += np.log(label_portion)
        res = np.argmax(midres)
        
        gold_label = labels[index]
        if res == gold_label:
            TP[res] += 1
            for label in label_set:
                TN[label] += 1
            TN[res] -= 1
        else:
            FP[res] += 1
            FN[gold_label] += 1
            for label in label_set:
                TN[label] += 1
            TN[res] -= 1
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

    acc = TP.sum() / (len(labels))
    print("accuracy score: ", acc)

    return Macro_f1, acc


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

    #Data cleaning ------ Naive Bayes model requires high purity of data.
    print("Executing data cleaning!")
    test_texts = total_cleaning(test_texts)
    train_texts = total_cleaning(train_texts)
    #valid_texts = total_cleaning(valid_texts)
    #print(test_texts)

    #After cleaning
    #print("Avg. length:", calculate_avg_length(train_texts + valid_texts + test_texts))
    
    # Extract features from the texts
    # Train the model and evaluate it on the valid set
    #Using hand-crafted Naive Bayes model to solve the problem.

    #c * dim, c = num of classes
    alpha = 1
    prob_matrix, word2id, unseen, label_portion = training(train_texts, train_labels, len(label_set), alpha)

    #Question: label_portion is assigned word portion per class or sample portion ??
    #Temporarily set to word portion

    

    print(label_portion)

    #print test labels' distribution
    print(test_labels.count(0)/len(test_labels))
    print(test_labels.count(1)/len(test_labels))
    print(test_labels.count(2)/len(test_labels))
    print(test_labels.count(3)/len(test_labels))
    print(test_labels.count(4)/len(test_labels))
    
    #eval(word_to_prob, label_set, valid_texts, valid_labels, vocab_size, label_portion, unseen)
    #eval(prob_matrix, valid_texts, valid_labels, label_set, label_portion, word2id, unseen)


    # Test the best performing model on the test set
    eval(prob_matrix, test_texts, test_labels, label_set, label_portion, word2id, unseen)
from cgitb import text
import csv
from utils import load_data, unique, calculate_avg_length, split_words_by_label,\
get_vocab_size, prob_Laplace_smoothing, accuracy, macro_F1

def eval(word_to_prob, label_set, valid_texts, valid_labels, vocab_size, label_portion, unseen):
    TP = {label:0 for label in label_set}
    TN = {label:0 for label in label_set}
    FP = {label:0 for label in label_set}
    FN = {label:0 for label in label_set}
    for text, gold_label in zip(valid_texts, valid_labels):
        max_likelihood = 0
        max_label = -1
        for label in label_set:
            #prob = label_portion[label]
            prob = label_portion[label] * 1000000
            for word in set(text.split()):
                if (word, label) in word_to_prob:
                    prob *= word_to_prob[(word, label)]
                else:
                    prob *= unseen[label]
            if prob > max_likelihood:
                max_label = label
                max_likelihood = prob
        print(text, " predicted over! likelihood = {}, label = {}".format(max_likelihood, max_label))
        if max_label == gold_label:
            TP[gold_label] += 1
            for label in label_set:
                TN[label] += 1
            TN[gold_label] -= 1

        else:
            for label in label_set:
                TN[label] += 1
            TN[gold_label] -= 1
            TN[max_label] -= 1
            FN[gold_label] += 1
            FP[max_label] += 1
    print("TP: ", TP)
    print("TN: ", TN)
    print("FP: ", FP)
    print("FN: ", FN)

    Macro_f1 = 0
    for label in label_set:
        Macro_f1 += macro_F1(TP[label], FP[label], FN[label])
    Macro_f1 /= len(label_set)
    print("Macro-F1 score: ", Macro_f1)

    return Macro_f1


if __name__ == '__main__':
    # Load data
    train_texts, train_labels = load_data('data/yelp_train.csv')

    #divide the train dataset into 5 splits, train : validation = 4 : 1
    valid_texts, valid_labels = train_texts[-int(len(train_texts)/5):], train_labels[-int(len(train_texts)/5):]
    train_texts, train_labels = train_texts[:-int(len(train_texts)/5)], train_labels[:-int(len(train_texts)/5)]
    test_texts, test_labels = load_data('data/yelp_test.csv')
    

    # Print basic statistics
    print("Training set size:", len(train_texts))
    print("Validation set size:", len(valid_texts))
    print("Test set size:", len(test_texts))
    label_set = list(unique(train_labels))
    label_set.sort()
    print("Unique labels:", label_set)
    print("Avg. length:", calculate_avg_length(train_texts + valid_texts + test_texts))

    # Extract features from the texts
    #Using hand-crafted Naive Bayes model to solve the problem.

    #mapping (word, label) to conditional probability
    word_to_prob = {}
    total_labeled_words, label_portion = split_words_by_label(train_texts, train_labels, label_set)
    #print(total_labeled_words)

    #Question: How many words should the vocabulary contain?
    #If only comes from train set, the prob of out-of-vocabulary is still quite high...
    vocab_size, _ = get_vocab_size(train_texts)
    #doubtful result
    print(vocab_size)
    
    # quite unbalanced data
    #for label in label_portion:
        #label_portion[label] /= len(train_texts)
    print(label_portion)

    # Train the model and evaluate it on the valid set
    for label in total_labeled_words:
        for word in total_labeled_words[label]:
            if (word, label) not in word_to_prob:
                word_to_prob[(word, label)] = prob_Laplace_smoothing(word, total_labeled_words, label, vocab_size, label_portion)
    
    #evaluating it on the valid set
    unseen = {label: 1 / (label_portion[label] + vocab_size) for label in label_set}
    print(unseen)
    for label in label_portion:
        label_portion[label] /= len(train_texts)

    eval(word_to_prob, label_set, valid_texts, valid_labels, vocab_size, label_portion, unseen)


    # Test the best performing model on the test set

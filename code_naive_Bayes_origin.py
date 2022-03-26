from cgitb import text
import csv

# texts, labels: list of raw sentences, list of labels.
def load_data(data_path):
    texts = []
    labels = []

    with open(data_path) as csvfile:
        reader = csv.reader(csvfile)            #generator
        next(reader) # skip header

        for row in reader:
            text = row[0].strip()
            label = int(row[1].strip())
            
            texts.append(text)
            labels.append(label)
    
    return texts, labels

def unique(xs):
    ret = set()
    for x in xs:
        ret.add(x)
    return ret

def calculate_avg_length(texts):
    cnt = 0
    for t in texts:
        cnt += len(t.split())
    return cnt / len(texts)

#texts, labels: list of texts, list of corresponding labels.
def split_words_by_label(texts, labels):
    total_labeled_words = {}
    label_portion = {}

    for text, label in zip(texts, labels):
        if label not in total_labeled_words:
            total_labeled_words[label] = {}
        # counting words' times of occurrence in labeled texts.
        # Sample counting!!!
        for word in set(text.split()):
            #print(word)
            if word not in total_labeled_words[label]:
                total_labeled_words[label][word] = 1
            else:
                total_labeled_words[label][word] += 1
        #counting the portion of label.
        if label not in label_portion:
            label_portion[label] = 1
        else:
            label_portion[label] += 1

    return total_labeled_words, label_portion

def get_vocab_size(train_texts):
    vocab = set()
    vocab_size = 0
    for seq in train_texts:
        for word in set(seq.split()):
            if word not in vocab:
                vocab.add(word)
                vocab_size += 1
    return vocab_size, vocab

def prob_Laplace_smoothing(word, total_labeled_words, label, vocab_size, label_portion):
    up = total_labeled_words[label][word] + 1
    down = label_portion[label] + vocab_size

    return up / down

def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + FP + FN + TN)

def macro_F1(TP, FP, FN):
    if TP == 0 and FP == 0:
        return 0
    precision = (TP)/(TP + FP)
    recall = (TP) / (TP + FN)
    return 2 * precision * recall / (precision + recall)


def eval(word_to_prob, label_set, valid_texts, valid_labels, vocab_size, label_portion, unseen):
    TP = {label:0 for label in label_set}
    TN = {label:0 for label in label_set}
    FP = {label:0 for label in label_set}
    FN = {label:0 for label in label_set}
    for text, gold_label in zip(valid_texts, valid_labels):
        max_likelihood = 0
        max_label = -1
        for label in label_set:
            prob = label_portion[label]
            for word in set(text.split()):
                if (word, label) in word_to_prob:
                    prob *= word_to_prob[(word, label)]
                else:
                    prob *= unseen[label]
            if prob > max_likelihood:
                max_label = label
                max_likelihood = prob
        #print(text, " predicted over! likelihood = {}, label = {}".format(max_likelihood, max_label))
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
    train_texts, train_labels = load_data('data/sst_train.csv')

    #divide the train dataset into 5 splits, train : validation = 4 : 1
    valid_texts, valid_labels = train_texts[-int(len(train_texts)/5):], train_labels[-int(len(train_texts)/5):]
    train_texts, train_labels = train_texts[:-int(len(train_texts)/5)], train_labels[:-int(len(train_texts)/5)]
    test_texts, test_labels = load_data('data/sst_test.csv')
    

    # Print basic statistics
    print("Training set size:", len(train_texts))
    print("Validation set size:", len(valid_texts))
    print("Test set size:", len(test_texts))
    label_set = unique(train_labels)
    print("Unique labels:", unique(label_set))
    print("Avg. length:", calculate_avg_length(train_texts + valid_texts + test_texts))

    # Extract features from the texts
    #Using hand-crafted Naive Bayes model to solve the problem.

    #mapping (word, label) to conditional probability
    word_to_prob = {}
    total_labeled_words, label_portion = split_words_by_label(train_texts, train_labels)
    #print(total_labeled_words)

    #Question: How many words should the vocabulary contain?
    #If only comes from train set, the prob of out-of-vocabulary is still quite high...
    vocab_size, _ = get_vocab_size(train_texts)
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
    for label in label_portion:
        label_portion[label] /= len(train_texts)

    eval(word_to_prob, label_set, valid_texts, valid_labels, vocab_size, label_portion, unseen)


    # Test the best performing model on the test set

from cgitb import text
import csv


# texts, labels: list of raw sentences, list of labels.
# considering aligning the labels 1-5 to 0-4
def load_data(data_path):
    texts = []
    labels = []
    print("loading data...")

    with open(data_path) as csvfile:
        reader = csv.reader(csvfile)            #generator
        next(reader) # skip header

        for row in reader:
            text = row[0].strip()
            label = int(row[1].strip())
            
            texts.append(text)
            if 'yelp' in data_path:
                labels.append(label-1)
            else:
                labels.append(label)
    
    return texts, labels

def unique(xs):
    ret = set()
    for x in xs:
        ret.add(x)
    return ret

#texts, labels: list of texts, list of corresponding labels.
def split_words_by_label(texts, labels, label_set):
    total_labeled_words = {label:{} for label in label_set}
    label_portion = {label: 0 for label in label_set}

    for text, label in zip(texts, labels):
        # counting words' times of occurrence in labeled texts.
        # Sample counting!!!
        for word in set(text.split()):
            #print(word)
            if word not in total_labeled_words[label]:
                total_labeled_words[label][word] = 1
            else:
                total_labeled_words[label][word] += 1
        #counting the portion of label.
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

def calculate_avg_length(texts):
    cnt = 0
    for t in texts:
        cnt += len(t.split())
    return cnt / len(texts)


def prob_Laplace_smoothing(word, total_labeled_words, label, vocab_size, label_portion, alpha = 0.2):
    up = total_labeled_words[label][word] + alpha
    down = label_portion[label] + vocab_size * alpha

    return up / down

def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + FP + FN + TN)

def macro_F1(TP, FP, FN):
    if TP == 0:
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
        #Not reasonable, some cases with prob = 0(too small to be represented by float) may be predicted to label 1.
        max_label = 1
        for label in label_set:
            #prob = label_portion[label]
            prob = label_portion[label] * 1000000000
            for word in set(text.split()):
                if (word, label) in word_to_prob:
                    prob *= word_to_prob[(word, label)]
                else:
                    prob *= unseen[label]
            if prob > max_likelihood:
                max_label = label
                max_likelihood = prob
        #print(text.split(), " predicted over! likelihood = {}, label = {}".format(max_likelihood, max_label))
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


def MSE(res, gold):
    n = len(res)
    return ((res-gold) ** 2).sum() / n




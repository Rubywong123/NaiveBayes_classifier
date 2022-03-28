from utils import macro_F1
import numpy as np

class NB:
    def __init__(self, alpha = 1):
        #setting hyperparameters
        self.alpha = alpha
    
    def get_word_feature(self, texts):
        print('building vocabulary based on training set...')
        self.word2id = {}
        self.dim = 0
        for text in texts:
            for word in text.split():
                if word not in self.word2id:
                    self.word2id[word] = self.dim
                    self.dim += 1
        print('vocabulary size: ', self.dim)

    def get_feature_matrix(self, texts):
        #print('generating feature matrix from scratch...')
        n = len(texts)
        dim = len(self.word2id)
        feature_matrix = np.zeros((n, dim))
        cnt = 0
        unseen_word_num = np.zeros((n,))

        for index, text in enumerate(texts):
            for word in text.split():
                if word in self.word2id:
                    feature_matrix[index][self.word2id[word]] += 1
                else:
                    cnt += 1
                    unseen_word_num[index] += 1
        #print("Total unseen words: ", cnt)
        return feature_matrix, unseen_word_num

    def fit(self, train_texts, train_labels, label_set):
        print('start training...')

        
        self.get_word_feature(train_texts)

        self.label_set = label_set
        self.max_label = len(self.label_set)
        self.prob_matrix = np.zeros((self.max_label, self.dim))
        
        for text, label in zip(train_texts, train_labels):
            for word in text.split():
                self.prob_matrix[label][self.word2id[word]] += 1

        #getting label portion.
        #a problem that is worth discussing.
        self.label_portion = self.prob_matrix.sum(axis = 1) / self.prob_matrix.sum()

        #calculating conditional probabilities
        up = self.prob_matrix + self.alpha
        down = self.prob_matrix.sum(axis = 1) + self.alpha * self.dim
        down = np.expand_dims(down, axis = 1)

        # For unseen tokens.
        self.unseen = np.ones((self.max_label,)) * self.alpha
        self.unseen = self.unseen / (self.prob_matrix.sum(axis = 1) + self.alpha * self.dim)
        self.prob_matrix = up / down

        
    def predict(self, texts):
        print('start predicting...')
        total_num = len(texts)
        res = np.zeros(total_num, dtype=int)
        
        #batched process, otherwise the need for memory can't be satisfied.;
        batch_size = 5
        for i in range(total_num // batch_size + 1):
            if batch_size*(i+1) > total_num:
                batch_texts = texts[batch_size*i:total_num]
            else:
                batch_texts = texts[batch_size*i:batch_size*(i+1)]


            feature_matrix, unseen_word_num = self.get_feature_matrix(batch_texts)
            feature_matrix = np.expand_dims(feature_matrix, axis = 0)
            prob_matrix = np.expand_dims(self.prob_matrix, axis = 1)
            # use log to switch multiply.reduce to sum operation.
            midres = np.log(prob_matrix) * feature_matrix
            unseen = np.expand_dims(self.unseen, axis = 1)
            unseen_word_num = np.expand_dims(unseen_word_num, axis = 0)
            midres = np.sum(midres, axis = 2) + np.log(unseen ** unseen_word_num)

            # strongly worsen the performance of sst-5
            label_portion = np.expand_dims(self.label_portion, axis = 1)
            midres += np.log(label_portion)

            if batch_size*(i+1) > total_num:
                res[batch_size*i: total_num] = np.argmax(midres, axis=0)
            else:
                res[batch_size*i: batch_size*(i+1)] = np.argmax(midres, axis = 0)
        return res

    def eval(self, results, labels):

        print('start evaluating...')


        #computing confusion matrix
        TP = np.zeros((self.max_label,))
        TN = np.zeros((self.max_label, ))
        FP = np.zeros((self.max_label,))
        FN = np.zeros((self.max_label,))

        for res, gold_label in zip(results, labels):
            if res == gold_label:
                TP[res] += 1
                for label in self.label_set:
                    TN[label] += 1
                TN[res] -= 1
            else:
                FP[res] += 1
                FN[gold_label] += 1
                for label in self.label_set:
                    TN[label] += 1
                TN[res] -= 1
                TN[gold_label] -= 1
            
        print("TP: ", TP)
        print("TN: ", TN)
        print("FP: ", FP)
        print("FN: ", FN)

        #computing macro-f1 score and accuracy score.

        Macro_f1 = 0
        for label in self.label_set:
            Macro_f1 += macro_F1(TP[label], FP[label], FN[label])
        Macro_f1 /= self.max_label
        print("Macro-F1 score: ", Macro_f1)

        acc = TP.sum() / (len(labels))
        print("accuracy score: ", acc)

        return Macro_f1, acc
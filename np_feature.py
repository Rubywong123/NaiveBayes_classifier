import numpy as np


'''
Totally Multinomial Distribution implemented by numpy.

'''
#Returns:
#word2id: mapping word to its dim in feature space
#index: dimension of feature space.
def get_word_feature(texts):
    print('getting word feature...')
    word2id = {}
    dim = 0
    for text in texts:
        for word in text.split():
            if word not in word2id:
                word2id[word] = dim
                dim += 1
    return word2id, dim

# use for generating feature vectors for samples.
# Return:
# feature_matrix: n * dim, n = num of sentences

def get_feature_matrix(texts, word2id):
    print('generating feature matrix for texts...')
    n = len(texts)
    dim = len(word2id)
    feature_matrix = np.zeros((n, dim))
    cnt = 0
    unseen_word_num = np.zeros((n,))

    for index, text in enumerate(texts):
        for word in text.split():
            if word in word2id:
                #two plans
                feature_matrix[index][word2id[word]] += 1
                #feature_matrix[index][word2id[word]] = 1
            else:
                cnt += 1
                #print("Unseen word No. %d" %(cnt), word)
                unseen_word_num[index] += 1


    return feature_matrix, unseen_word_num

#training NB model using train set.
#Return:
#prob_matrix: c * dim, c = num of classes
def training(train_texts, train_labels, max_label, alpha = 0.2):
    print('start training...')
    word2id, dim = get_word_feature(train_texts)
    #print(word2id.keys())
    prob_matrix = np.zeros((max_label, dim), dtype = np.float128)
    
    for text, label in zip(train_texts, train_labels):
        for word in text.split():
            # counting appearance times first. Two plans.
            prob_matrix[label][word2id[word]] += 1
            #prob_matrix[label][word2id[word]] = 1

    label_portion = prob_matrix.sum(axis = 1) / prob_matrix.sum()

    #calculating conditional probabilities
    up = prob_matrix + alpha
    down = prob_matrix.sum(axis = 1) + alpha * dim
    down = np.expand_dims(down, axis = 1)

    unseen = np.ones((max_label,)) * alpha
    unseen = unseen / (prob_matrix.sum(axis = 1) + alpha * dim)
    prob_matrix = up / down

    return prob_matrix, word2id, unseen, label_portion

            


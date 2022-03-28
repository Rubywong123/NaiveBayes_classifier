from utils import load_data, unique, MSE
from cleandata import total_cleaning
from model import NB
from config import get_opt

if __name__ == '__main__':

    #get options from parser
    args = get_opt()
    # Load data
    
    train_texts, train_labels = load_data('data/{}_train.csv'.format(args.dataset))
    test_texts, test_labels = load_data('data/{}_test.csv'.format(args.dataset))

    # Print basic statistics
    print("Training set size:", len(train_texts))
    print("Test set size:", len(test_texts))
    label_set = list(unique(train_labels))
    label_set.sort()
    print("Unique labels:", label_set)

    #Data cleaning ------ Naive Bayes model requires high purity of data.
    print("Executing data cleaning!")
    test_texts = total_cleaning(test_texts)
    train_texts = total_cleaning(train_texts)

    #After cleaning

    #Using hand-crafted Naive Bayes model to solve the problem.
    #Initiate Naive Bayes model
    model = NB(alpha = args.alpha)
    model.fit(train_texts, train_labels, label_set)


    '''print test labels' distribution
    print(test_labels.count(0)/len(test_labels))
    print(test_labels.count(1)/len(test_labels))
    print(test_labels.count(2)/len(test_labels))
    print(test_labels.count(3)/len(test_labels))
    print(test_labels.count(4)/len(test_labels))'''

    # Test the best performing model on the test set
    res = model.predict(test_texts)
    model.eval(res, test_labels)

    #MSE
    print(MSE(res, test_labels))
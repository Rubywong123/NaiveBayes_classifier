from cgitb import text
import csv

def load_data(data_path):
    texts = []
    labels = []

    with open(data_path) as csvfile:
        reader = csv.reader(csvfile)
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


if __name__ == '__main__':
    # Load data
    train_texts, train_labels = load_data('data/yelp_train.csv')
    valid_texts, valid_labels = train_texts[-int(len(train_texts)/5):], train_labels[-int(len(train_texts)/5):]
    train_texts, train_labels = train_texts[:-int(len(train_texts)/5)], train_labels[:-int(len(train_texts)/5)]
    test_texts, test_labels = load_data('data/yelp_test.csv')
    

    # Print basic statistics
    print("Training set size:", len(train_texts))
    print("Validation set size:", len(valid_texts))
    print("Test set size:", len(test_texts))
    print("Unique labels:", unique(train_labels))
    print("Avg. length:", calculate_avg_length(train_texts + valid_texts + test_texts))

    # Extract features from the texts


    # Train the model and evaluate it on the valid set


    # Test the best performing model on the test set

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


punct = "/-?!.,#$%()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&\n'

def total_cleaning(texts):
    nltk.download('stopwords')
    stop_list = stopwords.words('english')
    stop_list.append('\'s')
    lemmatizer = WordNetLemmatizer()
    #print(stop_list)
    cleaned_texts = []
    for text in texts:
        s = text
        for p in punct:
            s = s.replace(p, ' ')
        #clean numbers
        s = re.sub(r'[0-9]+', '', s)

        #eliminating stop words, and lemmatizing legal words.
        cleanwordlist = [lemmatizer.lemmatize(word) for word in s.lower().split() if word not in stop_list]
        s = ' '.join([word for word in cleanwordlist if len(word) > 1])

        cleaned_texts.append(s)

    #Remains: lemmatize

    return cleaned_texts




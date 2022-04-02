import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


punct = "/-?!.,#$%()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&\n'
custom_stoplist = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
     'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
      'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the','because', 'as', 'until', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
     'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from' ,'then', 'once', 'here', 'there','each', 'few', 'other']

def total_cleaning(texts):
    
    nltk.download('stopwords')
    nltk.download('wordnet')
    #stop_list = stopwords.words('english')
    print('start cleaning')
    stop_list = custom_stoplist
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

if __name__ == '__main__':
    stop_list = stopwords.words('english')
    print(stop_list)

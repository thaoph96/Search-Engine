import json
import nltk
import re
import pickle
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class Index_constructor:
    def __init__(self, bookkeeping, webpages, M1_index):
        self.bookkeeping = bookkeeping
        self.webpages = webpages
        self.inverted_index = {}
        self.N = 0
        self.M1_index = M1_index

    def load_ids(self):
        with open(self.bookkeeping) as f:
            data = json.load(f)
        self.N = len(list(data.keys()))
        return list(data.keys())

    def tokenize(self, string):
        s = ""
        for char in string:
            if char.isascii():
                if char.isalnum():
                    s += char
                else:
                    s += " "
        # removes punctuation and returns a list of tokens
        s = re.sub(r'[^\w\s]','',s.lower())
        tokens = word_tokenize(s)
        return tokens

    def lemmatization(self, tokens):
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_words

    def preprocess_text(self, text):
        tokens = self.tokenize(text)
        modified_tokens = self.lemmatization(tokens)
        modified_tokens = [t for t in tokens if not t in stopwords.words('english')]
        return modified_tokens

    def compute_word_frequencies(self, token_list) -> map:
        token_frequency = dict()
        for token in token_list:
            token_frequency[token] = token_frequency.get(token, 0) + 1
        return token_frequency

    def get_idf(self, term): #take a dictionary as parameter {term:docs}
        # idf weight is the number of documents that contain term t = log(N/df)
        posting = self.M1_index[term] # get the result of posting list of term from previous index : a list of docIDs
        df = len(posting)
        score = math.log10(self.N/df)
        return score
    
    def update_index(self, term, doc_id, tf_idf):
        """ A record in inverted index:
        { 
            "term": {doc_id: tf_idf}
        }
        """
        # If the term already exists
        if term in self.inverted_index.keys():
        # If doc_id not in doc_ids, add the doc_ids and tf_idf
            if doc_id not in self.inverted_index[term].keys():
                self.inverted_index[term][doc_id] = tf_idf                
        else:
            # If the document doesn't exist, create a new document
            self.inverted_index[term] = {doc_id: tf_idf}

    def save_index(self):
        with open("updated_index.pkl", "wb") as f:
            pickle.dump(self.inverted_index, f)

    def load_index(self):
        index = pickle.load(open("updated_index.pkl", "rb"))
        print(index)

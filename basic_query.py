import re
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import math
import numpy as np
from numpy.linalg import norm
from index_constructor import Index_constructor

class Basic_query:
    def __init__(self, query_string, inverted_index) -> None:
        self.query = query_string
        self.index = inverted_index

    def pageRank(self):
        bookkeeping = "WEBPAGES_RAW/bookkeeping.json"
        with open(bookkeeping) as f:
            data = json.load(f)
        N = len(list(data.keys()))
        query_tokens = self.preprocess_text(self.query)
        w_tq = self.term_query_weight(query_tokens, N)
        w_td = self.term_doc_weight(query_tokens)
        cosine_sim = self.cosine_similarity(query_tokens, w_tq, w_td)
        return cosine_sim
        # scores = dict()

        # query_words = index_constructor.lemmitization(self.query.lower().split())

        # for term in set(query_words):         
        #     score_dict = self.index[term]
            
        #     if len(score_dict.items()) > 0:
        #         scores[term] = score_dict

        #         sorted_scores = sorted(scores[term].items(), key=lambda item: item[1], reverse=True)[:20]

        #         scores[term] = sorted_scores.keys()
        #     else:
        #         scores[term] = []

        # return scores
    
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
        modified_tokens = [t for t in modified_tokens if not t in stopwords.words('english')]
        return modified_tokens
    
    def term_doc_weight(self, query_tokens):
        w_td = {}
        for term in query_tokens:
            for key in self.index[term].keys():
                if key not in w_td:
                    w_td[key] = {}

        for term in query_tokens:
            for doc in w_td:
                if doc in self.index[term].keys():
                    w_td[doc][term] = self.index[term][doc]
                else: w_td[doc][term] = 0
        return w_td

    def term_query_weight(self, query_tokens, N): 
        w_tq = {}
        query_tf = {}
        for term in query_tokens:
            query_tf[term] = 1 + math.log10(query_tokens.count(term)) 
            query_idf = math.log10(N/len(self.index[term]))
            # vector of query q
            w_tq[term] = query_tf[term]* query_idf    
        return w_tq

    def cosine_similarity(self, query_tokens, w_tq, w_td):
        cosine_sim = {}
        for docId in w_td.keys():
            # cosine similarity: cos(query, docId_i) = q*d_i/norm(d_i)*norm(q)
            # for docId in self.index[term].keys():
            weight_tq = list(w_tq.values())
            weight_td = list(w_td[docId].values())
            cosine_score = np.dot(weight_tq,weight_td)/(norm(weight_tq)*norm(weight_td))
                # if docId in cosine_sim.keys():
                #     cosine_sim[docId] += cosine_score
                # else:
            cosine_sim[docId] = cosine_score    
        return dict(sorted(cosine_sim.items(), key=lambda item:item[1], reverse = True)) 

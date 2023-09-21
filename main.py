import sys
import logging
import atexit
import json
import pickle
import math
import numpy as np
from numpy.linalg import norm
from nltk.corpus import stopwords



logger = logging.getLogger(__name__)

from index_constructor import Index_constructor
from DocumentParser import Parser
from basic_query import Basic_query


if __name__ == "__main__":
    # logger config. Don't worry about it
    logging.basicConfig(format='%(asctime)s (%(name)s) %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)

    webpages = "WEBPAGES_RAW" # just a hard_coded path to webpages
    # webpages = "C:/Users/zhiji/git/test_page"
    bookkeeping = "WEBPAGES_RAW/bookkeeping.json" # just a hard_coded path to webpages bookkeeping.json

#########################################
# the codes used for index creation
    # M1_index = pickle.load(open("M1_index.pkl", "rb"))
    # index_const = Index_constructor(bookkeeping, webpages, M1_index)
    # ids = index_const.load_ids()

    # parser = Parser(index_const)

    # atexit.register(index_const.save_index)

    # parser.start_parsing(ids)

    # index_const.load_index()

# the codes used for index creation
#########################################

# a sample query for unranked results(just first 20 url)

    # format of the index is "term"[a string] : {doc_id, tf_idf}[a dictionary]
    index = pickle.load(open("updated_index.pkl", "rb")) # changes this to your local path of updated_index if you want to test

    # opens the bookkeeping json so we can find urls using doc_id
    with open(bookkeeping) as f:
        data = json.load(f) # loads json as dictionary
    while(True):
        query = input("Input search query:")
        basic_query = Basic_query(query, index)

        cosine_sim = basic_query.pageRank()
        # prints length of query results and first 20 url
        keys = list(cosine_sim.keys())
        for i in range(20):
            print(f'Link {i+1}: https://{data[keys[i]]}')


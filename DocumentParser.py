import logging
import os
from bs4 import BeautifulSoup
from index_constructor import Index_constructor

logger = logging.getLogger(__name__)

class Parser:
    def __init__(self, index_constructor):
        self.index_const = index_constructor

    def start_parsing(self, ids):
        webpages = self.index_const.webpages
        count = 0
        for id in ids: 
            count += 1
            # a logger that keeps track of how many files to parse are left during index creation
            logger.info("Creating index...Currently at file %s. %s files left", id, len(ids) - count)

            # opens files and tokenize
            file_name = os.path.join(webpages, id)
            with open(file_name, "r", encoding="utf-8") as fp:
                text = BeautifulSoup(fp, 'html.parser').get_text()
            tokens = self.index_const.preprocess_text(text)

            # creates a hashmap of {token: token_frequency}. Sort first by alph then by freq
            frequency_map = self.index_const.compute_word_frequencies(tokens)
            sorted_frequency = sorted(frequency_map.items(), key=lambda x: (-x[1], x[0]))
            # sorted_frequency is a list of tuples (term : frequency)
            # index update
            for i in sorted_frequency:
                try:
                    idf = self.index_const.get_idf(i[0]) # calculate idf
                    tf = i[1] # term frequency in the doc
                    tf_idf = tf * idf
                    self.index_const.update_index(i[0], id, tf_idf) # param: term, doc_id, tf_idf
                # disregard probably nonsense tokens
                except KeyError:
                    pass

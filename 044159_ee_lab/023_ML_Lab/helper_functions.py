# helper_functions.py
# Created: Tal Daniel (August 2019)
# Updates: Ron Amit (March 2020)

# imports
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import nltk
from collections import Counter
import string

class EmailToWords(BaseEstimator, TransformerMixin):


    def __init__(self, stripHeaders=True, lowercaseConversion=True, punctuationRemoval=True,
                 urlReplacement=True, numberReplacement=True, stemming=True):
        # self.stripHeaders = stripHeaders  # - Strip email headers
        self.lowercaseConversion = lowercaseConversion  # - Convert to lowercase
        self.punctuationRemoval = punctuationRemoval  # - Remove punctuation
        # self.urlReplacement = urlReplacement # - Replace urls with "URL"
        # self.url_extractor = urlextract.URLExtract()
        # self.numberReplacement = numberReplacement   # - Replace numbers with "NUMBER"
        self.stemming = stemming   # - Perform Stemming
        self.stemmer = nltk.PorterStemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """"
        Transforms list of texts/emails to a list of word count dictionaries.
        Optional Pre-Processing (on by default):
        1. Change characters to lower-case.
        2. Remove punctuation
        3. Stemming - producing morphological variants of a root/base word (e.g,  "likes" -> "like", "liked"->"like" ... etc.)
        """
        X_to_words = []
        for email in X:
            text = email
            if text is None or not isinstance(text, str):
                text = 'empty'
            if self.lowercaseConversion:
                text = text.lower()

            # if self.urlReplacement:
            # urls = self.url_extractor.find_urls(text)
            # for url in urls:
            #    text = text.replace(url, 'URL')

            # Remove digits
            for substr in string.digits:
                text = text.replace(substr, '')
            if self.punctuationRemoval:
                for substr in string.punctuation:
                    text = text.replace(substr, ' ')
            word_counts = Counter(text.split())
            if self.stemming:
                stemmed_word_count = Counter()
                for word, count in word_counts.items():
                    stemmed_word = self.stemmer.stem(word)
                    stemmed_word_count[stemmed_word] += count
                word_counts = stemmed_word_count
            X_to_words.append(word_counts)
        return np.array(X_to_words)


class WordCountToVector(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=200):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        """"
        Creates a vocabulary: word dictionary of the most common words (key==word, value==index of word in the vocabulary)
        """
        total_word_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_word_count[word] += count
        self.most_common = total_word_count.most_common(self.vocabulary_size)
        self.vocabulary_ = {word: index for index, (word, count) in enumerate(self.most_common)}
        return self

    def transform(self, X, y=None):
        """"
        Transform a list of word counts per email into a "spare matrix" with dimensions #emails X vocabulary_size.
        The entries of the matrix count the number of times a given word appear in a given email.
        """
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                if word in self.vocabulary_:
                    rows.append(row)
                    cols.append(self.vocabulary_[word])
                    data.append(count)
        # create a sparse matrix
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size))


email_pipeline = Pipeline([
    ("Email to Words", EmailToWords()),
    ("Wordcount to Vector", WordCountToVector()),
])

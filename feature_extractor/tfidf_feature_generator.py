from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TfidfFeatureGenerator(object):
    def __init__(self):
        self.vectorizer_headline = TfidfVectorizer()
        self.vectorizer_body = TfidfVectorizer()

    def fit_transform(self, headlines, bodies):
        """
        Fits the TF-IDF model to the headlines and bodies and transforms the text into TF-IDF vectors.
        :param headlines: list of headline texts
        :param bodies: list of body texts
        :return: TF-IDF vectors for headlines, bodies, and cosine similarity between each headline and body pair
        """
        all_text = headlines + bodies
        all_vectorizer = TfidfVectorizer().fit(all_text)

        # Use the fitted vocabulary to transform the headlines and bodies
        self.vectorizer_headline.vocabulary_ = all_vectorizer.vocabulary_
        self.vectorizer_body.vocabulary_ = all_vectorizer.vocabulary_

        tfidf_headlines = self.vectorizer_headline.transform(headlines)
        tfidf_bodies = self.vectorizer_body.transform(bodies)

        cos_similarities = [cosine_similarity(tfidf_headlines[i], tfidf_bodies[i])[0][0] for i in range(len(headlines))]

        return tfidf_headlines, tfidf_bodies, cos_similarities

    def transform(self, headlines, bodies):
        """
        Transforms new headlines and bodies into TF-IDF vectors using the fitted model.
        :param headlines: list of headline texts
        :param bodies: list of body texts
        :return: TF-IDF vectors for headlines, bodies, and cosine similarity between each headline and body pair
        """
        tfidf_headlines = self.vectorizer_headline.transform(headlines)
        tfidf_bodies = self.vectorizer_body.transform(bodies)

        cos_similarities = [cosine_similarity(tfidf_headlines[i], tfidf_bodies[i])[0][0] for i in range(len(headlines))]

        return tfidf_headlines, tfidf_bodies, cos_similarities
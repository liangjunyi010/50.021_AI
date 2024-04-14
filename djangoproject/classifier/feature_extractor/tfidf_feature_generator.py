from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm


class TfidfFeatureGenerator(object):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def tfidf_cosine_features(self, headlines, bodies,enable=None):
        """
        Generates TF-IDF cosine similarity features between headlines and bodies.
        :param headlines: List of headline texts.
        :param bodies: List of body texts.
        :return: List of cosine similarity scores between corresponding headlines and bodies.
        """
        texts = headlines + bodies
        self.vectorizer.fit(
            texts)  # Fit the vectorizer to both headlines and bodies together to ensure a unified feature space.

        # Transform headlines and bodies into TF-IDF vectors
        tfidf_headlines = self.vectorizer.transform(headlines)
        tfidf_bodies = self.vectorizer.transform(bodies)

        # Calculate cosine similarities
        cos_similarities = []
        for i in tqdm(range(len(headlines)), desc="Computing TF-IDF cosine similarities"):
            cos_similarity = cosine_similarity(tfidf_headlines[i], tfidf_bodies[i])[0][0]
            cos_similarities.append([cos_similarity])

        return np.array(cos_similarities)
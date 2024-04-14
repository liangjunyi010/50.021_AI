from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm


class SvdFeatureGenerator(object):
    def __init__(self, n_components=100):
        """
        Initialize the SVD feature generator.
        :param n_components: The number of SVD components after dimensionality reduction.
        """
        self.vectorizer = TfidfVectorizer()
        self.n_components = n_components
        self.svd = None  # Delayed initialization

    def fit_transform(self, headlines, bodies, enable=None):
        """
        Perform TF-IDF transformation and SVD dimensionality reduction on text data, then compute cosine similarities.
        :param headlines: List of headline texts.
        :param bodies: List of body texts.
        :return: Cosine similarities between the SVD features of headlines and bodies.
        """
        # Combine texts for TF-IDF
        texts = headlines + bodies
        tfidf_features = self.vectorizer.fit_transform(texts)

        # Initialize SVD with min number of components
        n_features = tfidf_features.shape[1]  # Get the number of features from tfidf
        n_components = min(self.n_components, n_features - 1)  # Adjust components to be less than features
        self.svd = TruncatedSVD(n_components=n_components)

        # Apply SVD dimensionality reduction
        svd_features = self.svd.fit_transform(tfidf_features)

        # Separate SVD features for headlines and bodies
        svd_headlines = svd_features[:len(headlines)]
        svd_bodies = svd_features[len(headlines):]

        # Compute cosine similarities
        cos_similarities = []
        for i in tqdm(range(len(headlines)), desc="Computing SVD cosine similarities"):
            cos_similarity = cosine_similarity([svd_headlines[i]], [svd_bodies[i]])[0][0]
            cos_similarities.append(cos_similarity)

        return np.array(cos_similarities).reshape(-1, 1)
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Word2VecFeatureGenerator(object):
    def __init__(self, model_path='/home/liangjunyi/GitHub/50.021_AI/google_model/GoogleNews-vectors-negative300.bin'):
        """
        Initialize the feature generator with a pre-trained Word2Vec model.
        :param model_path: Path to the pre-trained Word2Vec model.
        """
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    def document_vector(self, doc):
        """
        Compute the vector for a document by averaging the vectors of all words in the document.
        :param doc: A list of words in the document.
        :return: Averaged vector of the document.
        """
        # Filter out words not in the model's vocabulary
        vectors = [self.model[word] for word in doc if word in self.model.key_to_index]

        if len(vectors) == 0:
            return np.zeros(self.model.vector_size)
        else:
            return np.mean(vectors, axis=0)

    def generate_features(self, headlines, bodies, enable=None):
        """
        Generate Word2Vec features for headlines and bodies.
        :param headlines: List of headlines.
        :param bodies: List of bodies.
        :return: Cosine similarities between the Word2Vec features of headlines and bodies.
        """

        headline_vectors = np.array([self.document_vector(doc.split()) for doc in headlines])

        body_vectors = np.array([self.document_vector(doc.split()) for doc in bodies])

        cosine_similarities = np.zeros(len(headline_vectors))

        for i, (h_vec, b_vec) in enumerate(zip(headline_vectors, body_vectors)):
            h_vec_reshaped = h_vec.reshape(1, -1)
            b_vec_reshaped = b_vec.reshape(1, -1)
            cosine_similarities[i] = cosine_similarity(h_vec_reshaped, b_vec_reshaped)

        return cosine_similarities.reshape(-1, 1)
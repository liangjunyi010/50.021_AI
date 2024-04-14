from .data_preprocessing import clean,get_tokenized_lemmas
from tqdm import tqdm
import numpy as np

class PolarityFeaturesGenerator(object):
    def polarity_features(self,headlines, bodies,enable=None):
        _refuting_words = [
            'fake',
            'fraud',
            'hoax',
            'false',
            'deny', 'denies',
            'not',
            'despite',
            'nope',
            'doubt', 'doubts',
            'bogus',
            'debunk',
            'pranks',
            'retract'
        ]

        def calculate_polarity(text):
            tokens = get_tokenized_lemmas(text)
            return sum([t in _refuting_words for t in tokens]) % 2

        X = []
        for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
            clean_headline = clean(headline)
            clean_body = clean(body)
            features = []
            features.append(calculate_polarity(clean_headline))
            features.append(calculate_polarity(clean_body))
            X.append(features)
        return np.array(X)
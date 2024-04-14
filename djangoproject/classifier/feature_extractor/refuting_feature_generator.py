from .data_preprocessing import clean,get_tokenized_lemmas
from tqdm import tqdm
class RefutingFeaturesGenerator(object):
    def refuting_features(self,headlines, bodies,enable=None):
        _refuting_words = [
            'fake',
            'fraud',
            'hoax',
            'false',
            'deny', 'denies',
            # 'refute',
            'not',
            'despite',
            'nope',
            'doubt', 'doubts',
            'bogus',
            'debunk',
            'pranks',
            'retract'
        ]
        X = []
        for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
            clean_headline = clean(headline)
            clean_headline = get_tokenized_lemmas(clean_headline)
            features = [1 if word in clean_headline else 0 for word in _refuting_words]
            X.append(features)
        return X
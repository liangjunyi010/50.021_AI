from tqdm import tqdm
from .data_preprocessing import clean,get_tokenized_lemmas

class WordOverlapFeaturesGenerator(object):
    def word_overlap_features(self,headlines, bodies,enable=None):
        X = []
        for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
            clean_headline = clean(headline)
            clean_body = clean(body)
            clean_headline = get_tokenized_lemmas(clean_headline)
            clean_body = get_tokenized_lemmas(clean_body)
            features = [
                len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
            X.append(features)
        return X
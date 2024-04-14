
from .data_preprocessing import clean,remove_stopwords
from tqdm import tqdm
import numpy as np

class HandFeaturesGenerator(object):
    def ngrams(self,input, n):
        input = input.split(' ')
        output = []
        for i in range(len(input) - n + 1):
            output.append(input[i:i + n])
        return output

    def chargrams(self,input, n):
        output = []
        for i in range(len(input) - n + 1):
            output.append(input[i:i + n])
        return output

    def append_chargrams(self,features, text_headline, text_body, size):
        grams = [' '.join(x) for x in self.chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
        grams_hits = 0
        grams_early_hits = 0
        grams_first_hits = 0
        for gram in grams:
            if gram in text_body:
                grams_hits += 1
            if gram in text_body[:255]:
                grams_early_hits += 1
            if gram in text_body[:100]:
                grams_first_hits += 1
        features.append(grams_hits)
        features.append(grams_early_hits)
        features.append(grams_first_hits)
        return features

    def append_ngrams(self,features, text_headline, text_body, size):
        grams = [' '.join(x) for x in self.ngrams(text_headline, size)]
        grams_hits = 0
        grams_early_hits = 0
        for gram in grams:
            if gram in text_body:
                grams_hits += 1
            if gram in text_body[:255]:
                grams_early_hits += 1
        features.append(grams_hits)
        features.append(grams_early_hits)
        return features

    def hand_features(self,headlines, bodies,enable=None):

        def binary_co_occurence(headline, body):
            # Count how many times a token in the title
            # appears in the body text.
            bin_count = 0
            bin_count_early = 0
            for headline_token in clean(headline).split(" "):
                if headline_token in clean(body):
                    bin_count += 1
                if headline_token in clean(body)[:255]:
                    bin_count_early += 1
            return [bin_count, bin_count_early]

        def binary_co_occurence_stops(headline, body):
            # Count how many times a token in the title
            # appears in the body text. Stopwords in the title
            # are ignored.
            bin_count = 0
            bin_count_early = 0
            for headline_token in remove_stopwords(clean(headline).split(" ")):
                if headline_token in clean(body):
                    bin_count += 1
                    bin_count_early += 1
            return [bin_count, bin_count_early]

        def count_grams(headline, body):
            # Count how many times an n-gram of the title
            # appears in the entire body, and intro paragraph

            clean_body = clean(body)
            clean_headline = clean(headline)
            features = []
            features = self.append_chargrams(features, clean_headline, clean_body, 2)
            features = self.append_chargrams(features, clean_headline, clean_body, 8)
            features = self.append_chargrams(features, clean_headline, clean_body, 4)
            features = self.append_chargrams(features, clean_headline, clean_body, 16)
            features = self.append_ngrams(features, clean_headline, clean_body, 2)
            features = self.append_ngrams(features, clean_headline, clean_body, 3)
            features = self.append_ngrams(features, clean_headline, clean_body, 4)
            features = self.append_ngrams(features, clean_headline, clean_body, 5)
            features = self.append_ngrams(features, clean_headline, clean_body, 6)
            return features

        X = []
        for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
            X.append(binary_co_occurence(headline, body)
                     + binary_co_occurence_stops(headline, body)
                     + count_grams(headline, body))


        return X
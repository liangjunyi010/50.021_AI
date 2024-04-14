from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np


class SentimentFeatureGenerator(object):
    """
    This class uses the Sentiment Analyzer in the NLTK package to assign a sentiment polarity score
    to the headline and body separately.
    """

    def __init__(self):
        """
        Initialize the Sentiment Feature Generator with the NLTK Sentiment Analyzer.
        """
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def generate_features(self, headlines, bodies, enable):
        """
        Generate sentiment polarity scores for headlines and bodies.
        :param headlines: List of headline strings.
        :param bodies: List of body text strings.
        :return: Array of sentiment polarity scores for headlines and bodies.
        """
        # Compute sentiment scores
        headline_sentiments = np.array([self.sentiment_analyzer.polarity_scores(h)['compound'] for h in headlines])
        body_sentiments = np.array([self.sentiment_analyzer.polarity_scores(b)['compound'] for b in bodies])

        # Combine headline and body sentiments into features
        sentiment_features = np.column_stack((headline_sentiments, body_sentiments))

        return sentiment_features
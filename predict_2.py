from joblib import load

from feature_extractor.feature_generator import SingleFeatureGenerator
from utils.score import LABELS

# Load the pre-trained model
model = load('./model/best_model.joblib')


def predict_relationship(headline, body):
    # Create an instance of the feature generator
    feature_generator = SingleFeatureGenerator()

    # Generate features for the input headline and body
    features = feature_generator.generate_features(headline, body)

    # Predict relationship using the trained model
    prediction = model.predict(features)

    # Return the predicted relationship label
    return LABELS[int(prediction[0])]


# Example usage
headline = "Your example headline here"
body = "Your example body text here"
relationship = predict_relationship(headline, body)
print("The predicted relationship is:", relationship)
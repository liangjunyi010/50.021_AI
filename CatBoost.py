from catboost import CatBoostClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from feature_extractor.feature_generator import FeatureGenerator
from utils.system import parse_params, check_version
if __name__ == "__main__":
    check_version()
    parse_params()

    # Load the training dataset and generate folds
    d = DataSet()
    folds, hold_out = kfold_split(d, n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d, folds, hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    fg = FeatureGenerator(competition_dataset.stances, competition_dataset, "competition")
    X_competition, y_competition = fg.generate_features()

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    fg = FeatureGenerator(hold_out_stances, d, "holdout")
    X_holdout, y_holdout = fg.generate_features()
    for fold in fold_stances:
        fg = FeatureGenerator(fold_stances[fold], d, str(fold))
        Xs[fold], ys[fold] = fg.generate_features()

    best_score = 0
    best_fold = None

    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        # Standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = Xs[fold]
        X_test = scaler.transform(X_test)
        y_test = ys[fold]

        # Initialize CatBoost classifier
        clf = CatBoostClassifier(
            loss_function='MultiClass',
            # Additional parameters can be added here as needed
        )

        clf.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            # Additional training parameters can be added here as needed
        )

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score / max_fold_score

        print("Score for fold " + str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf

    # Run on Holdout set and report the final score on the holdout set
    X_holdout = scaler.transform(X_holdout)  # Don't forget to transform the holdout set as well
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual, predicted)
    print("")

    # Run on competition dataset
    X_competition = scaler.transform(X_competition)  # Standardize competition set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual, predicted)

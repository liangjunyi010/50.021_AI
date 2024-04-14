import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from feature_extractor.feature_generator import FeatureGenerator
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from utils.system import parse_params, check_version


if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    print('Before balance')
    original_stance_counts = d.print_stance_counts(d.stances)

    augmented_stances = d.augment_data(d.stances,n_augment=1)
    print('After balance')
    augmented_stance_counts = d.print_stance_counts(augmented_stances)
    d.stances = augmented_stances

    folds, hold_out = kfold_split(d, n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d, folds, hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    print('competition_dataset')
    competition_dataset.print_stance_counts(competition_dataset.stances)
    fg = FeatureGenerator(competition_dataset.stances, competition_dataset, "competition")
    X_competition, y_competition = fg.generate_features()

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    fg = FeatureGenerator(hold_out_stances,d,"holdout")
    X_holdout,y_holdout = fg.generate_features()
    for fold in fold_stances:
        fg = FeatureGenerator(fold_stances[fold],d,str(fold))
        Xs[fold],ys[fold] = fg.generate_features()


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        # smote = SMOTE()
        # X_train, y_train = smote.fit_resample(X_train, y_train)

        X_test = Xs[fold]
        y_test = ys[fold]

        clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf
            dump(best_fold, 'model/best_model.joblib')



    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual,predicted)

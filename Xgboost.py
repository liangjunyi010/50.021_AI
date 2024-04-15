from xgboost import XGBClassifier
import numpy as np
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from feature_extractor.feature_generator import FeatureGenerator
from utils.system import parse_params, check_version

# Merged parameters from the first code snippet
params_xgb = {
    'max_depth': 6,
    'colsample_bytree': 0.6,
    'subsample': 1.0,
    'eta': 0.1,
    'silent': 1,
    'objective': 'multi:softmax',
    'eval_metric':'mlogloss',
    'num_class': 4
}

if __name__ == "__main__":
    check_version()
    parse_params()

    d = DataSet()
    folds, hold_out = kfold_split(d, n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d, folds, hold_out)

    competition_dataset = DataSet("competition_test")
    fg = FeatureGenerator(competition_dataset.stances, competition_dataset, "competition")
    X_competition, y_competition = fg.generate_features()

    Xs = dict()
    ys = dict()

    fg = FeatureGenerator(hold_out_stances, d, "holdout")
    X_holdout, y_holdout = fg.generate_features()
    for fold in fold_stances:
        fg = FeatureGenerator(fold_stances[fold], d, str(fold))
        Xs[fold], ys[fold] = fg.generate_features()

    best_score = 0
    best_fold = None

    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        # Using the merged parameters for the XGBClassifier instance
        clf = XGBClassifier(**params_xgb, n_estimators=200, random_state=14128, verbosity=1, use_label_encoder=False)
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score / max_fold_score

        print("折叠 " + str(fold) + " 的分数是 - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf

    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual, predicted)
    print("\n")

    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual, predicted)

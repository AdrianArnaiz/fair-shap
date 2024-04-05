""" Wrapper for error-parity library.
André F. Cruz and Moritz Hardt. "Unprocessing Seven Years of Algorithmic Fairness." arXiv preprint, 2023.
https://github.com/socialfoundations/error-parity/tree/main
"""

from error_parity import RelaxedThresholdOptimizer

def mh_postpro(X_val, S_val, y_val, model, X_test, S_test, pos_label = None, seed=None):
    
    predictor = lambda X: model.predict_proba(X)[:, -1]
    
    # Given any trained model that outputs real-valued scores
    fair_clf = RelaxedThresholdOptimizer(
        predictor=predictor,   # for sklearn API
        constraint="equalized_odds",
        tolerance=0.05,     # fairness constraint tolerance
        seed = seed
    )

    # Fit the fairness adjustment on some data
    # This will find the optimal _fair classifier_
    fair_clf.fit(X=X_val, y=y_val, group=S_val, pos_label=pos_label)

    # Now you can use `fair_clf` as any other classifier
    # You have to provide group information to compute fair predictions
    y_pred_test = fair_clf(X=X_test, group=S_test)
    return y_pred_test
"""
https://github.com/google-research/google-research/blob/master/label_bias/Label_Bias_EqualOpportunity.ipynb
Implementation for method introduced in: Jiang, H., & Nachum, O. (2019).
Identifying and correcting label bias in machine learning. arXiv preprint arXiv:1901.04966
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

def get_error_and_violations_EOp(y_pred, y, protected_attributes, unpriv_group_idx):
    """Get accuracy and violations of Equal Opportunity 
       MUST GET THE FAVORABLE LABEL AS 1 and unfacorable as 0 (it calculates percentages of positive predictions as np.mean)
    """
    acc = np.mean(y_pred == y)
    violations = []
    for p in protected_attributes:
        #inutition: unprivileged group and favorable label
        protected_idxs = np.where(np.logical_and(p == unpriv_group_idx, y == 1))
        positive_idxs = np.where(y == 1)
        xxx=np.mean(y_pred[positive_idxs]) - np.mean(y_pred[protected_idxs])
        violations.append(xxx)
    pairwise_violations = []
    for i in range(len(protected_attributes)):
        for j in range(i+1, len(protected_attributes)):
            protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
            if len(protected_idxs[0]) == 0:
                continue
            pairwise_violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
    return acc, violations, pairwise_violations


def get_error_and_violations_DP(y_pred, y, protected_attributes, unpriv_gr_idx):
    """Get accuracy and violations of Demographic Parity. 
       MUST GET THE FAVORABLE LABEL AS 1 and unfacorable as 0 (it calculates percentages of positive predictions as np.mean)
    """
    acc = np.mean(y_pred != y)
    violations = []
    for p in protected_attributes:
        protected_idxs = np.where(p == unpriv_gr_idx)
        violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
    pairwise_violations = []
    for i in range(len(protected_attributes)):
        for j in range(i+1, len(protected_attributes)):
            protected_idxs = np.where(np.logical_and(protected_attributes[i] > 0, protected_attributes[j] > 0))
            if len(protected_idxs[0]) == 0:
                continue
            pairwise_violations.append(np.mean(y_pred) - np.mean(y_pred[protected_idxs]))
    return acc, violations, pairwise_violations


def debias_weights(original_labels, protected_attributes, multipliers, unpriv_gr_idx):
    """UNPRIVILEGED GROUP MUST BE 1 (others 0) in exponents -= m * (1-protected_attributes[i])
    """

    exponents = np.zeros(len(original_labels))
    for i, m in enumerate(multipliers):
        if unpriv_gr_idx == 0:
            protected_attrs_vuln_1 = (1-protected_attributes[i].copy())
        else:
            protected_attrs_vuln_1 = protected_attributes[i].copy()
        exponents -= m * protected_attrs_vuln_1
    weights = np.exp(exponents)/ (np.exp(exponents) + np.exp(-exponents))
    weights = np.where(original_labels > 0, 1 - weights, weights)
    return weights

def jiang_weights(X_train, y_train, protected_train, unpriv_group_idx, n_iters=100):
    multipliers = np.zeros(len(protected_train))
    weights = np.array([1] * X_train.shape[0])
    learning_rate = 1.
    for it in range(n_iters):
        model = LogisticRegression()

        model.fit(X_train, y_train, weights)
        y_pred_train = model.predict(X_train)
        
        if (it+1)!=100:
            weights = debias_weights(y_train, protected_train, multipliers, unpriv_group_idx)

        acc, violations, pairwise_violations = get_error_and_violations_DP(y_pred_train, 
                                                                        y_train, 
                                                                        protected_train,
                                                                        unpriv_group_idx)
        multipliers += learning_rate * np.array(violations)
    
    return weights
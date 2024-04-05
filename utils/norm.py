import numpy as np

def indenpendence_probability_norm(ori_weights, labels, attrs, priv_attr, unpriv_attr, fav_lab, unfav_lab):
    weights = ori_weights.copy()
    n = np.sum(weights)
    n_p = np.sum(weights[attrs == priv_attr])
    n_up = np.sum(weights[attrs == unpriv_attr])
    n_fav = np.sum(weights[labels == fav_lab])
    n_unfav = np.sum(weights[labels == unfav_lab])

    n_p_fav = np.sum(weights[(attrs == priv_attr)&(labels == fav_lab)])
    n_p_unfav = np.sum(weights[(attrs == priv_attr)&(labels == unfav_lab)])
    n_up_fav = np.sum(weights[(attrs == unpriv_attr)&(labels == fav_lab)])
    n_up_unfav = np.sum(weights[(attrs == unpriv_attr)&(labels == unfav_lab)])

    # reweighing weights
    weights[(attrs == priv_attr)  &  (labels == fav_lab)]   *= (n_fav*n_p / (n*n_p_fav))
    weights[(attrs == priv_attr)  &  (labels == unfav_lab)] *= (n_unfav*n_p / (n*n_p_unfav))
    weights[(attrs == unpriv_attr) & (labels == fav_lab)]   *= (n_fav*n_up / (n*n_up_fav))
    weights[(attrs == unpriv_attr) & (labels == unfav_lab)] *= (n_unfav*n_up / (n*n_up_unfav))

    return weights
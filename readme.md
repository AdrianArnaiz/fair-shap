
# FairShap: Instance-Level Data Valuation for Algorithmic Fairness 

*Fair Data Valuation using Shapley Values for Algorithmic Fairness through Re-weighting and Data Pruning.*


## Implementation

* NumBa speed-up implementation:
```python
from fairSV.fair_shapley_sklearn import get_SV_matrix_numba_memory, get_sv_arrays

protected_attributes_dict = {'values': A_array, #array of protected attributes,
                             'privileged_protected_attribute': int(priv_attr), #id privileged
                             'unprivileged_protected_attribute': int(unpriv_attr), #id unprivileged
                             'favorable_label':int(fav_lab), 'unfavorable_label':int(unfav_lab)} #id favorable and unfavorable label


SV = get_SV_matrix_numba_memory(X_train, X_valid, y_train, y_valid) #SV Matrix
svs_acc, svs_eop, _, _ = get_sv_arrays(SV, y_valid, protected_attributes_dict, 'all')
```
`SV` refers to $\mathbf{\Phi}$ in the paper. `svs_acc, svs_eop`,... refers to the data valuation $\phi(.)$ for the given value function.

Then use them for your prefered goal: bias mitigation through re-weighting, data generation, exploratoty data analysis, data minimization, data acquisition policies...


### Data Analysis

* Distribution of $\phi()$'s and weights
![SV_EOp distribution](https://github.com/AdrianArnaiz/fair-shap/blob/main/figs/svs_hist_german_age.png "SV_EOp distribution")
![Distribution of Normalized SV_EOp for rewewighting](https://github.com/AdrianArnaiz/fair-shap/blob/main/figs/svs_weights_hist_german_age.png "Distribution of Normalized SV_EOp for rewewighting")


* Embedding space exploration
![Embbeding space of LFWA, FairFace and LFWA with sizes proportinal to SVEop](https://github.com/AdrianArnaiz/fair-shap/blob/main/figs/SVimgembeddingBSUBSET.png "Embbeding space of LFWA, FairFace and LFWA with sizes proportinal to SVEop computed using FairFace as reference dataset")



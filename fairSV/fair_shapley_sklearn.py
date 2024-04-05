__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "adrian@ellisalicante.org"
__version__ = "1.0.0"

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def get_SV_matrix_numba_memory(X_train, X_test, y_train, y_test, K=5):  
    shapley_mat = np.zeros((X_train.shape[0], X_test.shape[0]), dtype=np.float64)
    
    for j in prange(X_test.shape[0]):
        # Get single point shapley
        xt_query = X_test[j]
        y_t_label = y_test[j]
        
        distance1 = np.sum(np.square(X_train - xt_query), axis=1)  # Euclidean distance
        alpha = np.argsort(distance1)

        N = X_train.shape[0]
        for i in range(N-1, -1, -1): 
            if i == N-1:
                shapley_mat[alpha[i], j] = np.float64(int(y_train[alpha[i]] == y_t_label) / N)
            else:
                shapley_mat[alpha[i], j] = np.float64(shapley_mat[alpha[i+1], j] +
                                                     (int(y_train[alpha[i]] == y_t_label) - int(y_train[alpha[i+1]] == y_t_label)) / K * min(K, i+1) / (i+1))

    return shapley_mat



def get_sv_arrays(SV, y_test, protected_attributes_dict, sv_type='acc'):
    """
    Calculate different arrays based on support vectors (SV) and test labels.

    Parameters:
    SV (numpy.ndarray): Support vectors.
    y_test (numpy.ndarray): Test labels.
    protected_attributes_dict (dict): Dictionary containing protected attributes information.
        The dictionary should have the following keys:
            - 'values': A numpy array containing the protected attribute values for each sample.
            - 'privileged_protected_attribute': The privileged value of the protected attribute.
            - 'unprivileged_protected_attribute': The unprivileged value of the protected attribute.
            - 'favorable_label': The label considered as favorable.
            - 'unfavorable_label': The label considered as unfavorable.
    sv_type (str): Type of arrays to calculate. Default is 'acc'.

    Returns:
    numpy.ndarray: Array based on support vectors and test labels.

    Raises:
    NotImplementedError: If sv_type is not implemented.
    """
    if sv_type == 'acc':
        return SV.mean(axis=1) # Original
    else:
        protected_attributes = protected_attributes_dict['values']
        privileged_attr_value = protected_attributes_dict['privileged_protected_attribute']
        unprivileged_attr_value = protected_attributes_dict['unprivileged_protected_attribute']
        fav_label = protected_attributes_dict['favorable_label']
        unfav_label = protected_attributes_dict['unfavorable_label']

        #conditioned tpr and tnr
        f_fav_priv = (y_test==fav_label) & (protected_attributes == privileged_attr_value)
        sv_tpr_p = SV[:,f_fav_priv].mean(axis=1) 

        f_fav_unpriv = (y_test==fav_label) & (protected_attributes == unprivileged_attr_value)
        sv_tpr_u = SV[:,f_fav_unpriv].mean(axis=1) 

        f_unfav_priv = (y_test==unfav_label) & (protected_attributes == privileged_attr_value)
        sv_tnr_p = SV[:,f_unfav_priv].mean(axis=1) 

        f_unfav_unpriv = (y_test==unfav_label) & (protected_attributes == unprivileged_attr_value)
        sv_tnr_u = SV[:,f_unfav_unpriv].mean(axis=1) 

        N=SV.shape[0]
        sv_fpr_p = (1/N) - sv_tnr_p
        sv_fpr_u = (1/N) - sv_tnr_u

        if sv_type =='all':
            sv_eop = sv_tpr_u -  sv_tpr_p
            sv_eod_diff = 0.5*((sv_fpr_u-sv_fpr_p)+(sv_tpr_u-sv_tpr_p))
            sv_eod_abs = 0.5*(np.abs(sv_fpr_u-sv_fpr_p)+np.abs(sv_tpr_u-sv_tpr_p))
            return SV.mean(axis=1), sv_eop, sv_eod_diff, sv_eod_abs
        elif sv_type == 'eop':
            return sv_tpr_u -  sv_tpr_p
        elif sv_type == 'eod_diff':
            return 0.5*((sv_fpr_u-sv_fpr_p)+(sv_tpr_u-sv_tpr_p))
        elif sv_type == 'eod_abs':
            return 0.5*(np.abs(sv_fpr_u-sv_fpr_p)+np.abs(sv_tpr_u-sv_tpr_p))
        else:
            raise NotImplementedError(f'sv_type not implemented: {sv_type}')
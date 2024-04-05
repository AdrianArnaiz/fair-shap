
from aif360.metrics import ClassificationMetric
import numpy as np
from collections import OrderedDict
import pandas as pd
from sklearn.model_selection import train_test_split

from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult, get_distortion_german, get_distortion_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas

def compute_metrics(dataset_true, dataset_pred, 
                    unprivileged_groups, privileged_groups,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                 dataset_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Accuracy"] = classified_metric_pred.accuracy()
    metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate()+
                                             classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = classified_metric_pred.average_odds_difference()
    metrics["Average absoulte odds difference"] = classified_metric_pred.average_abs_odds_difference()
    metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    
    if disp:
        for k in metrics:
            print("\t%s = %.4f" % (k, metrics[k]))
    
    return metrics

def stratified_aif360_split(aif360dataset, num_or_size_splits, shuffle=False, seed=None):

    n = aif360dataset.features.shape[0]
    
    if isinstance(num_or_size_splits, list):
        num_folds = len(num_or_size_splits) + 1
        if num_folds > 1 and all(x <= 1. for x in num_or_size_splits):
            num_or_size_splits = [int(x * n) for x in num_or_size_splits]
    else:
        # num_or_size_splits is an integer
        num_folds = num_or_size_splits
    
    #Order indexes in a shuffle way but stratified (A,Y) in [:num_or_size_splits] and [num_or_size_splits:]
    tr, vl = train_test_split(pd.DataFrame(aif360dataset.features), test_size=n-num_or_size_splits[0], 
                          stratify=np.hstack([aif360dataset.labels,aif360dataset.protected_attributes]), 
                          shuffle=shuffle,
                          random_state=seed)
    order = [int(i) for i in tr.index] + [int(j) for j in vl.index]
    
    #Copy data type - aif360 issue
    folds = [aif360dataset.copy() for _ in range(num_folds)]
    
    #Order all data and filter - aif360 attributes editing to match the order of the folds
    features = np.array_split(aif360dataset.features[order], num_or_size_splits)
    labels = np.array_split(aif360dataset.labels[order], num_or_size_splits)
    scores = np.array_split(aif360dataset.scores[order], num_or_size_splits)
    protected_attributes = np.array_split(aif360dataset.protected_attributes[order], num_or_size_splits)
    instance_weights = np.array_split(aif360dataset.instance_weights[order], num_or_size_splits)
    instance_names = np.array_split(np.array(aif360dataset.instance_names)[order], num_or_size_splits)

    for fold, feats, labs, scors, prot_attrs, inst_wgts, inst_name in zip(
            folds, features, labels, scores, protected_attributes, instance_weights,
            instance_names):

        fold.features = feats
        fold.labels = labs
        fold.scores = scors
        fold.protected_attributes = prot_attrs
        fold.instance_weights = inst_wgts
        fold.instance_names = list(map(str, inst_name))
        fold.metadata = fold.metadata.copy()
        fold.metadata.update({
            'transformer': '{}.split'.format(type(aif360dataset).__name__),
            'params': {'num_or_size_splits': num_or_size_splits,
                       'shuffle': shuffle},
            'previous': [aif360dataset]
        })

    return folds

def make_aif360_dataset_balanced(aifdataset, by='label_attr', strategy='random', seed=42):
    df_label_attr = pd.DataFrame({'names': aifdataset.instance_names,
                                  'label' : aifdataset.labels.ravel(),
                                  'attr': aifdataset.protected_attributes.ravel()}
                                )
    
    if by == 'label_attr': 
        smallest_group_size = df_label_attr[['label', 'attr']].value_counts().min()
        smallest_group_lab, smallest_group_attr = df_label_attr[['label', 'attr']].value_counts(sort=True).index[-1]

        df_fair = df_label_attr[(df_label_attr['label']==smallest_group_lab) & (df_label_attr['attr']==smallest_group_attr)].copy()

        for label in np.unique(aifdataset.labels):
            for attr in np.unique(aifdataset.protected_attributes):
                f = (df_label_attr['label']==label) & (df_label_attr['attr']==attr)
                df_group = df_label_attr[f]
                if not len(df_group) == smallest_group_size:
                    if strategy == 'random':
                        df_fair = pd.concat([df_fair, df_group.sample(smallest_group_size, random_state=seed)])
        
    elif by == 'attr' or by == 'label':
        smallest_group_size = df_label_attr[by].value_counts().min()
        smallest_group_attr = df_label_attr[by].value_counts(sort=True).index[-1]

        df_fair = df_label_attr[(df_label_attr[by]==smallest_group_attr)].copy()

        values = np.unique(aifdataset.protected_attributes) if by == 'attr' else np.unique(aifdataset.labels)
        for x in values:
            df_group = df_label_attr[df_label_attr[by]==x]
            if not len(df_group) == smallest_group_size:
                if strategy == 'random':
                    df_fair = pd.concat([df_fair, df_group.sample(smallest_group_size, random_state=seed)])

    order = list(map(int,np.array(df_fair.index)))
    #Copy data type - aif360 issue
    new_dataset = aifdataset.copy()

    #Order all data and filter - aif360 attributes editing to match the order of the folds
    new_dataset.features =  aifdataset.features[order]
    new_dataset.labels = aifdataset.labels[order]
    new_dataset.scores = aifdataset.scores[order]
    new_dataset.protected_attributes = aifdataset.protected_attributes[order]
    new_dataset.instance_weights = aifdataset.instance_weights[order]
    new_dataset.instance_names = list(map(str,np.array(aifdataset.instance_names)[order]))

    new_dataset.metadata.update({
            'transformer': '{}.split'.format(type(aifdataset).__name__),
            'params': {'undersampled': strategy},
            'previous': [aifdataset]
        })

    return new_dataset
            


def tabular_data_loader(dataset_used, protected_attribute_used):
    if dataset_used == "adult":
        if protected_attribute_used == 1:
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_orig = load_preproc_data_adult(['sex'])
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            dataset_orig = load_preproc_data_adult(['race'])
        optim_options = {
            "distortion_fun": get_distortion_adult,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }
        IF_params = {'li_l2_reg': 2.26,
                     'li_alpha': 1,
                     'li_beta': 0.5,
                     'li_gamma': 0.2}
        
    elif dataset_used == "german":
        if protected_attribute_used == 1:
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_orig = load_preproc_data_german(['sex'])
        else:
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
            dataset_orig = load_preproc_data_german(['age'])
        optim_options = {
            "distortion_fun": get_distortion_german,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }
        IF_params={'li_l2_reg':5.85,
                     'li_alpha':1,
                     'li_beta':0,
                     'li_gamma':0}
        
    elif dataset_used == "compas":
        if protected_attribute_used == 1:
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_orig = load_preproc_data_compas(['sex'])
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            dataset_orig = load_preproc_data_compas(['race'])
        optim_options = {
            "distortion_fun": get_distortion_compas,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }
        IF_params = {'li_l2_reg':37.0,
                     'li_alpha':1,
                     'li_beta':0.2,
                     'li_gamma':0.1}
    else:
        raise ValueError(f"Dataset name invalid: {dataset_used}")
    
    return dataset_orig, privileged_groups, unprivileged_groups, optim_options, IF_params


def standarize_aif360_data(dataset):

    #change metadata
    dataset.metadata['params']['privileged_protected_attributes'] = [np.array([1.])]
    dataset.metadata['params']['unprivileged_protected_attributes'] = [np.array([0.])]

    new_lab_mapping = {}
    for lab, lab_name in dataset.metadata['label_maps'][0].items():
        if lab == dataset.favorable_label:
            new_lab_mapping[1.] = lab_name
        else:
            new_lab_mapping[0.] = lab_name

    dataset.metadata['label_maps'][0] = new_lab_mapping

    new_attr_mapping = {}
    for attr, attr_name in dataset.metadata['protected_attribute_maps'][0].items():
        if attr == dataset.privileged_protected_attributes[0]:
            new_attr_mapping[1.] = attr_name
        else:
            new_attr_mapping[0.] = attr_name
        
    dataset.metadata['protected_attribute_maps'][0] = new_attr_mapping


    dataset.labels = (dataset.labels==dataset.favorable_label).astype(np.float32)
    dataset.favorable_label = 1.
    dataset.unfavorable_label = 0.

    dataset.protected_attributes = (dataset.protected_attributes==dataset.privileged_protected_attributes).astype(np.float32)
    dataset.privileged_protected_attributes = [np.array([1.])]
    dataset.unprivileged_protected_attributes = [np.array([0.])]
    
    privileged_groups = [{dataset.protected_attribute_names[0]: 1}]
    unprivileged_groups = [{dataset.protected_attribute_names[0]: 0}]
    
    return dataset, privileged_groups, unprivileged_groups



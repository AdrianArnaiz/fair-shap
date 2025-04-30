import numpy as np
from tqdm import tqdm
import os
import pickle as pkl


from aif360.algorithms.preprocessing.reweighing import Reweighing
from utils.aif360_utils import stratified_aif360_split, tabular_data_loader
from utils.aif360_utils import standarize_aif360_data
from utils.IFLiLiu.weights_fns import get_IF_weights

from utils.fair_metrics_raw import compute_all_metrics

#from fairSV.fair_shapley import FairShapley
from fairSV.fair_shapley_sklearn import get_SV_matrix_numba_memory, get_sv_arrays


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")


#*#####################################
#*Get arguments
import argparse
parser = argparse.ArgumentParser(description='Delete features for fairSV')
parser.add_argument('--dataset', type=str, help='dataset to use')
parser.add_argument('--atribute', type=int, default=0, help='protected attribute to use', choices=[0,1])
parser.add_argument('--model', type=str, default='GBC', help='model to use', choices=["LR", "GBC"])
parser.add_argument('--seed', type=int, help='seed to use')
parser.add_argument('--percRemove', type=float, default=0.1, help='percentage of data to remove')
parser.add_argument('--percStep', type=float, default=0.01, help='percentage of data to remove')
parser.add_argument('--action', type=str, default='remove', help='add-remove', choices=['add', 'remove'])
parser.add_argument('--order', type=str, default='low', help='add-remove', choices=['low', 'high'])
parser.add_argument("--states", type=str, nargs='+', default=None, help="States used for ACSIncome")

args = parser.parse_args()
dataset_used = args.dataset
if dataset_used == 'acsincome':
    ACS_STATES = args.states
    assert ACS_STATES is not None, "ACS_STATES must be provided for ACSIncome dataset"
protected_attribute_used = args.atribute
MODEL_TYPE = args.model
SEED = args.seed
exp_config_addition = args.action
exp_config_value = args.order

PERC_TO_REMOVE = args.percRemove
PERC_STEP = args.percStep

SAVE_PATH = f"results/tabular/pruning/{dataset_used}_{protected_attribute_used}_{PERC_TO_REMOVE}_{PERC_STEP}/"


#*#####################################
#* Load data
dataset_orig, privileged_groups, unprivileged_groups, optim_options, IF_params= tabular_data_loader(dataset_used,
                                                                                                    protected_attribute_used,
                                                                                                     acsstates=ACS_STATES)
print(f"Data loaded: {dataset_used} with {dataset_orig.features.shape} features")
dataset_orig,_, _  = standarize_aif360_data(dataset_orig)

dataset_orig_train, dataset_orig_vt = stratified_aif360_split(dataset_orig, [0.8], shuffle=True, seed = SEED)
dataset_orig_valid, dataset_orig_test = stratified_aif360_split(dataset_orig_vt, [0.5], shuffle=True, seed = SEED)
X_train = dataset_orig_train.features
y_train = dataset_orig_train.labels.ravel()
X_valid = dataset_orig_valid.features
y_valid = dataset_orig_valid.labels.ravel()

#*#####################################
#* COMPUTE DIFF WEIGHTS
data_and_weights = dict()
#random
data_and_weights['Rand'] = dataset_orig_train.copy()
#groupRW
RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
RW.fit(dataset_orig_train)
data_and_weights['simple'] = RW.transform(dataset_orig_train.copy())
#IF
data_and_weights['IF'] = dataset_orig_train.copy()
data_and_weights['IF'].instance_weights =  get_IF_weights(X_train, y_train, X_valid, y_valid,
                                                          dataset_orig_valid.protected_attributes.ravel(),
                                                          'eop', IF_params['li_l2_reg'], SEED, IF_params['li_alpha'],
                                                          IF_params['li_beta'], IF_params['li_gamma'])
#Shapley Values
protected_attributes_dict = {'values':dataset_orig_valid.protected_attributes.ravel(),
                             'privileged_protected_attribute': int(dataset_orig_valid.privileged_protected_attributes[0][0]),
                             'unprivileged_protected_attribute': int(dataset_orig_valid.unprivileged_protected_attributes[0][0]),
                             'favorable_label':int(dataset_orig_valid.favorable_label),
                             'unfavorable_label':int(dataset_orig_valid.unfavorable_label)
                            }    
SV = get_SV_matrix_numba_memory(X_train, X_valid, y_train, y_valid, K=10)
svs_acc, svs_eop, svs_eod_diff, svs_eod_abs = get_sv_arrays(SV, y_valid, protected_attributes_dict, 'all')
data_and_weights['SVAcc'] = dataset_orig_train.copy()
data_and_weights['SVAcc'].instance_weights = svs_acc
data_and_weights['SVEOp'] = dataset_orig_train.copy()
data_and_weights['SVEOp'].instance_weights = svs_eop
data_and_weights['SVEOdds'] = dataset_orig_train.copy()
data_and_weights['SVEOdds'].instance_weights = svs_eod_diff


#*#####################################
#* Train different models
all_experiments_metrics = OrderedDict()
for REWEIGHING in tqdm(['Rand', 'simple', "SVAcc", "SVEOp", "SVEOdds", 'IF']):
    dataset_orig_train_rw = data_and_weights[REWEIGHING].copy()
           
    # order weights from high to lower or the other way around depending on the experiment
    if exp_config_value == 'high':
        idx_order_weights = dataset_orig_train_rw.instance_weights.argsort()[::-1]
    elif exp_config_value =='low':
        idx_order_weights = dataset_orig_train_rw.instance_weights.argsort()
    else:
        raise ValueError("Invalid high/low strategy: {}".format(exp_config_value))
    # if no weights_ random
    if REWEIGHING=='Rand': #random position
        idx_order_weights = np.random.permutation(len(X_train))

    #order data according to SV value
    ordered_X_train = dataset_orig_train_rw.copy().features[idx_order_weights]
    ordered_y_train = dataset_orig_train_rw.copy().labels.ravel()[idx_order_weights]
    ordered_w_train = dataset_orig_train_rw.copy().instance_weights.ravel()[idx_order_weights]
    
    exp_metrics = OrderedDict()
    exp_metrics["Accuracy"] = []
    exp_metrics["Balanced accuracy"] = []
    exp_metrics["Average odds difference"] = [] 
    exp_metrics["Average absoulte odds difference"] = [] 
    exp_metrics["Equal opportunity difference"] = [] 
    exp_metrics['F1']  = [] 
    exp_metrics['Macro F1']  = [] 
    exp_metrics['Macro F1 ind']  = [] 
    
    
    total_remove = int(ordered_X_train.shape[0]*PERC_TO_REMOVE)
    step_remove = int(ordered_X_train.shape[0]*PERC_STEP)
    
    for idx_limit in tqdm(range(0,total_remove+1,step_remove), position=1, leave=False):
        
        if exp_config_addition == 'add':
            slice_X_train = ordered_X_train.copy()[:idx_limit]
            slice_y_train = ordered_y_train.copy()[:idx_limit]
            slice_w_train = ordered_w_train.copy()[:idx_limit]
        elif exp_config_addition == 'remove':
            slice_X_train = ordered_X_train.copy()[idx_limit:]
            slice_y_train = ordered_y_train.copy()[idx_limit:]
            slice_w_train = ordered_w_train.copy()[idx_limit:]
        else:
            raise ValueError("Invalid add/rem strategy: {}".format(exp_config_addition))

        if MODEL_TYPE == "LR":
            model = LogisticRegression(random_state=SEED)
        elif MODEL_TYPE == 'GBC':
            model = GradientBoostingClassifier(random_state=SEED)
            
        model.fit(slice_X_train, slice_y_train)
        
        pos_ind = np.where(model.classes_ == dataset_orig_train_rw.favorable_label)[0][0] # positive class index
        ## Scores for test set
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        

        X_test = dataset_orig_test_pred.features
            
        y_test = dataset_orig_test_pred.labels
        dataset_orig_test_pred.scores = model.predict_proba(X_test)[:,pos_ind].reshape(-1,1)



        #Compute test metrics in best threshold decision
        fav_inds = dataset_orig_test_pred.scores > 0.5
        dataset_orig_test_pred.labels[fav_inds] = dataset_orig_test_pred.favorable_label
        dataset_orig_test_pred.labels[~fav_inds] = dataset_orig_test_pred.unfavorable_label
        
        priv_attr   = dataset_orig_train.privileged_protected_attributes[0][0]
        unpriv_attr = dataset_orig_train.unprivileged_protected_attributes[0][0]
        metric_test_bef = compute_all_metrics(dataset_orig_test.labels.ravel(),
                                      dataset_orig_test_pred.labels.ravel(),
                                    {'values':dataset_orig_test.protected_attributes.ravel(),
                                    'privileged_protected_attribute': int(priv_attr),
                                    'unprivileged_protected_attribute': int(unpriv_attr),
                                    'favorable_label':int(dataset_orig_test_pred.favorable_label),
                                    'unfavorable_label':int(dataset_orig_test_pred.unfavorable_label)}
                                    )
        
        exp_metrics["Accuracy"].append(metric_test_bef['acc'])        
        exp_metrics["Balanced accuracy"].append(metric_test_bef['ba'])
        exp_metrics['F1'].append(metric_test_bef['f1'])
        exp_metrics['Macro F1'].append(metric_test_bef['macrof1'])
        exp_metrics['Macro F1 ind'].append(metric_test_bef['macrof1_ind'])
        exp_metrics["Average odds difference"].append(metric_test_bef['Diff_EOds'])
        exp_metrics["Average absoulte odds difference"].append(metric_test_bef['Abs_EOds'])
        exp_metrics["Equal opportunity difference"].append(metric_test_bef['EOp'])
        
    #Save experiment metrics:
    if REWEIGHING not in all_experiments_metrics: all_experiments_metrics[REWEIGHING] = {}
    all_experiments_metrics[REWEIGHING] = exp_metrics


exp_name = f"""{dataset_used}_{dataset_orig.protected_attribute_names[0]}"""
exp_name = exp_name + f"""_{MODEL_TYPE}_{exp_config_addition}_{exp_config_value}"""
exp_name = exp_name + f"""_{int(PERC_TO_REMOVE*100)}_{int(PERC_STEP*100)}_S{SEED}"""

#Save results
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
with open(SAVE_PATH + exp_name + '.pkl', 'wb') as f:
    pkl.dump(all_experiments_metrics, f)
print(f"Results saved in {SAVE_PATH + exp_name + '.pkl'}")


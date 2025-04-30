#General and sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from utils.weighted_knn import WeightedKNN

#aif360 specific
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.preprocessing import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
#aif360 personalized wrappers
from utils.aif360_utils import stratified_aif360_split, tabular_data_loader, standarize_aif360_data

#Personalized (FS, baselines and metrics)
from fairSV.fair_shapley_sklearn import get_SV_matrix_numba_memory, get_sv_arrays
from utils.labelbias import jiang_weights
from utils.norm import indenpendence_probability_norm
try:
    #! Only works in a special environment with Gurobi optimization library
    from utils.IFLiLiu.weights_fns import get_IF_weights
except:
    pass
from utils.fair_metrics_raw import compute_all_metrics

#! Only works in a special environment with error-parity library
try:
    from utils.eqoddspostpro import mh_postpro
except:
    pass


# OS based
import os
from tqdm import tqdm
import warnings
import argparse
import time
warnings.filterwarnings("ignore")
exp_time = time.strftime('%d_%m_%y__%H_%M')

############# Global variables #############xº

RELABEL = True

POTENTIAL_METHODS = ['None', 'PExp1', 'PExpW', 'SVacc', 'SVeop', 'SVeodDiff', 'SVeodABS', 'POSTPRO', "calmon", "jiang", "LiLiu"]
N_ITER = 10
SAVE_RESULTS = True

MODEL_NAME = "GBC"
ACS_STATES = ['AL']

FAIR_METHODS = [['None'],
                ['PExpW'],
                ['POSTPRO'], #mh library
                ["jiang"],
                #["calmon"],
                ['LiLiu'], # SV library
                ['SVacc'],
                ['SVeop'],
                ['SVeop', 'PExpW'],
                ['SVeodDiff'],
                ['SVeodDiff', 'PExpW'],
                ['SVeodABS'],
                ['SVeodABS', 'PExpW'],
            ]

#! needs special environment due to Gurobi optimization library compatibility
"""FAIR_METHODS = [["POSTPRO"],
                ["LiLiu"]]"""

############# INPUTS #############

parser = argparse.ArgumentParser()
parser.add_argument("--attr", type=int,
                    default=None,
                    choices=[0, 1],
                    help="Attr used")


parser.add_argument("--dataset", type=str,
                    choices=['compas', 'german', 'adult', 'acsincome'],
                    help='<Required> Set flag', required=True)

parser.add_argument("--states", type=str, nargs='+',
                    default=None,
                    help="States usedfor ACSIncome")


args = parser.parse_args()
DATASET_USED = args.dataset
protected_attribute_used = int(args.attr)
if DATASET_USED == 'acsincome':
    ACS_STATES = args.states
    assert ACS_STATES is not None, "ACS_STATES must be provided for ACSIncome dataset"

print("###########################################")
print(f"DATASET USED: {DATASET_USED}")
if DATASET_USED == 'acsincome':
    print(f"ACS STATES USED: {ACS_STATES}")
print(f"PROTECTED ATTRIBUTE USED: {protected_attribute_used}")
print("###########################################")


##########################
for fair_method in FAIR_METHODS:
    assert isinstance(fair_method, list), f'FAIR_TECHNIIQUE MUST BE A LIST. Now: {fair_method} {type(fair_method)}'
    assert np.all([item in POTENTIAL_METHODS for item in fair_method]), f'{fair_method}'

results = {}

##########################


#* Load dataset
dataset_orig, privileged_groups, unprivileged_groups, optim_options, IF_params = tabular_data_loader(DATASET_USED,
                                                                                                     protected_attribute_used,
                                                                                                     acsstates=ACS_STATES)

np.random.seed(42)
seeds = np.random.randint(12345679,size=N_ITER)
for seed in tqdm(seeds, desc="Processing Seeds"):
    COMPUTE_SV_MATRIX = True
    inner_loop_fair_methods = tqdm(FAIR_METHODS, desc="Initializing...", leave=False)
    for fair_method in inner_loop_fair_methods:
        inner_loop_fair_methods.set_description(f"Seed {seed} - Processing {fair_method}")
        assert isinstance(fair_method, list), f'FAIR_TECHNIIQUE MUST BE A LIST. Now: {fair_method} {type(fair_method)}'
        assert np.all([item in POTENTIAL_METHODS for item in fair_method])
        method_key = '-'.join(fair_method)
        if method_key not in results: 
            results[method_key] = {}
            results[method_key]['acc'] = []
            results[method_key]['ba'] = []
            results[method_key]['f1'] = []
            results[method_key]['macrof1'] = []
            results[method_key]['macrof1_ind'] = []
            results[method_key]['eods_dif'] = []
            results[method_key]['eods_abs'] = []
            results[method_key]['eops'] = []

        #* Load dataset and split (same seed for all methods)
        # loaded here instead of outside to reinitialize the dataset in case of any transformation has been applied)
        dataset_orig_train, dataset_orig_vt = stratified_aif360_split(dataset_orig, [0.7], shuffle=True, seed = seed)
        dataset_orig_valid, dataset_orig_test = stratified_aif360_split(dataset_orig_vt, [0.5], shuffle=True, seed = seed)


        if RELABEL:
            dataset_orig_train, privileged_groups, unprivileged_groups = standarize_aif360_data(dataset_orig_train)
            dataset_orig_valid, _, _ = standarize_aif360_data(dataset_orig_valid)
            dataset_orig_test, _, _ = standarize_aif360_data(dataset_orig_test)


        fav_lab     = dataset_orig_train.favorable_label
        unfav_lab   = dataset_orig_train.unfavorable_label
        priv_attr   = dataset_orig_train.privileged_protected_attributes[0][0]
        unpriv_attr = dataset_orig_train.unprivileged_protected_attributes[0][0]

        scale_orig = StandardScaler()
        X_train = scale_orig.fit_transform(dataset_orig_train.features)
        y_train = dataset_orig_train.labels.ravel()
        X_valid = scale_orig.transform(dataset_orig_valid.features)
        y_valid = dataset_orig_valid.labels.ravel()
        X_test = scale_orig.transform(dataset_orig_test.features)
        y_test = dataset_orig_test.labels.ravel()


        #* PREPROCESSING
        if any(item.startswith('SV') for item in fair_method):
            sv_type = [item for item in fair_method if item.startswith('SV')][0]
            assert sv_type in ['SVacc', 'SVeop', 'SVeodDiff', 'SVeodABS']

            protected_attributes_dict = {'values':dataset_orig_valid.protected_attributes.ravel(),
                                    'privileged_protected_attribute': int(priv_attr),
                                    'unprivileged_protected_attribute': int(unpriv_attr),
                                    'favorable_label':int(fav_lab), 'unfavorable_label':int(unfav_lab)}
            
            sv_arrays = {}
            if COMPUTE_SV_MATRIX:
                SV = get_SV_matrix_numba_memory(X_train, X_valid, y_train, y_valid, K=5)
                COMPUTE_SV_MATRIX = False

            svs_acc, svs_eop, svs_eod_diff, svs_eod_abs = get_sv_arrays(SV, y_valid, protected_attributes_dict, 'all')
            sv_arrays["SVacc"] = svs_acc
            sv_arrays["SVeop"] = svs_eop
            sv_arrays["SVeodDiff"] = svs_eod_diff
            sv_arrays["SVeodABS"] = svs_eod_abs

            weight = sv_arrays[sv_type].copy()
            dataset_orig_train.instance_weights = weight
            dataset_orig_train.instance_weights = (dataset_orig_train.instance_weights - dataset_orig_train.instance_weights.min())/(dataset_orig_train.instance_weights.max()-dataset_orig_train.instance_weights.min())
            dataset_orig_train.instance_weights *= (dataset_orig_train.labels.ravel().shape[0]/dataset_orig_train.instance_weights.sum())

        if 'PExpW' in fair_method:
            dataset_orig_train.instance_weights = indenpendence_probability_norm(dataset_orig_train.instance_weights,
                                                                                labels=dataset_orig_train.labels.ravel(),
                                                                                attrs=dataset_orig_train.protected_attributes.ravel(),
                                                                                priv_attr=priv_attr, unpriv_attr=unpriv_attr,
                                                                              fav_lab=fav_lab, unfav_lab=unfav_lab)
        elif 'PExp1' in fair_method:
            dataset_orig_train.instance_weights *= indenpendence_probability_norm(np.ones(dataset_orig_train.instance_weights.shape[0]),
                                                                                labels=dataset_orig_train.labels.ravel(),
                                                                                attrs=dataset_orig_train.protected_attributes.ravel(),
                                                                                priv_attr=priv_attr, unpriv_attr=unpriv_attr,
                                                                                fav_lab=fav_lab, unfav_lab=unfav_lab)
        elif "calmon" in fair_method:
            OP = OptimPreproc(OptTools, optim_options,
                  unprivileged_groups = unprivileged_groups,
                  privileged_groups = privileged_groups)
            OP = OP.fit(dataset_orig_train)
            
            # Transform training data and align features
            dataset_transf_train = OP.transform(dataset_orig_train, transform_Y=True)
            dataset_orig_train = dataset_orig_train.align_datasets(dataset_transf_train)

            scale_orig = StandardScaler()
            X_train = scale_orig.fit_transform(dataset_orig_train.features)
            y_train = dataset_orig_train.labels.ravel()

        elif "jiang"  in fair_method:
            # Scores always have favorable label as 1 and non-fav as 0, in constrast to labels that change depending on the dataset
            y_for_jiang = (dataset_orig_train.labels.ravel()==fav_lab).astype(np.float32)
            w_lbl_bias = jiang_weights(X_train, y_for_jiang, [dataset_orig_train.protected_attributes.ravel()], unpriv_attr)
            dataset_orig_train.instance_weights = w_lbl_bias

        elif "LiLiu" in fair_method:
            s_valid = dataset_orig_valid.protected_attributes.ravel()
            w = get_IF_weights(X_train, y_train, X_valid, y_valid, s_valid, 'eop',
                               IF_params['li_l2_reg'], seed, IF_params['li_alpha'], IF_params['li_beta'], IF_params['li_gamma'])
            dataset_orig_train.instance_weights = w
            
        #* MODEL training
        if MODEL_NAME == 'GBC':
            model = GradientBoostingClassifier(random_state=seed)
        elif MODEL_NAME == 'KNN':
            model = WeightedKNN(n_neighbors=15)
        else:
            raise NotImplementedError(f"Model {model} not implemented")
        # Fit model
        model.fit(X_train, y_train, sample_weight=dataset_orig_train.instance_weights)

        # positive class index according to model output's index:
        pos_ind = np.where(model.classes_ == dataset_orig_train.favorable_label)[0][0]

        #* MODEL Predictions
        ## Val
        val_pred = dataset_orig_valid.copy()
        val_pred.labels = model.predict(X_valid).reshape((-1, 1))
        val_pred.scores = model.predict_proba(X_valid)[:, pos_ind].reshape(-1,1)
        ## Test
        pred = dataset_orig_test.copy()
        pred.labels = model.predict(X_test).reshape((-1, 1))
        pred.scores = model.predict_proba(X_test)[:, pos_ind].reshape(-1,1)

        #* Post-Processing:
        if 'POSTPRO' in fair_method:
            #? Should I use AIF360 or e-p implementation?
            #eqo = EqOddsPostprocessing(unprivileged_groups, privileged_groups, seed=seed)
            #pred = eqo.fit(dataset_orig_valid, val_pred).predict(pred)
            pred.labels = mh_postpro(X_valid, val_pred.protected_attributes.ravel().astype(int), y_valid,
                                     model,
                                     X_test, pred.protected_attributes.ravel().astype(int),
                                     pos_label=fav_lab,
                                     seed=seed)
            pred.labels = pred.labels.reshape((-1, 1))


        metrics = compute_all_metrics(dataset_orig_test.labels.ravel(), pred.labels.ravel(),
                                    {'values':dataset_orig_test.protected_attributes.ravel(),
                                    'privileged_protected_attribute': int(priv_attr),
                                    'unprivileged_protected_attribute': int(unpriv_attr),
                                    'favorable_label':int(fav_lab), 'unfavorable_label':int(unfav_lab)}
                                    )
        results[method_key]['acc'].append(metrics['acc'])
        results[method_key]['ba'].append(metrics['ba'])
        results[method_key]['f1'].append(metrics['f1'])
        results[method_key]['macrof1'].append(metrics['macrof1'])
        results[method_key]['macrof1_ind'].append(metrics['macrof1_ind'])
        results[method_key]['eods_dif'].append(metrics['Diff_EOds'])
        results[method_key]['eods_abs'].append(metrics['Abs_EOds'])
        results[method_key]['eops'].append(metrics['EOp'])
        

df_rows_mean = []
df_rows_std = []
for fair_method in FAIR_METHODS:
    method_key = '-'.join(fair_method)
    accs = results[method_key]['acc']
    bas = results[method_key]['ba']
    f1s = results[method_key]['f1']
    macrof1s = results[method_key]['macrof1']
    macrof1_inds = results[method_key]['macrof1_ind']
    eods_dif = results[method_key]['eods_dif']
    eods_abs = results[method_key]['eods_abs']
    eops = results[method_key]['eops']

    #save arrays as csvs in folder called dataset/attribute/individual/method
    if SAVE_RESULTS:
        path = f"results/tabular/{MODEL_NAME}-{DATASET_USED}-{N_ITER}it-{exp_time}/{dataset_orig.protected_attribute_names[0]}/{method_key}"
        if not os.path.exists(path):
            os.makedirs(path)
        np.savetxt(f"{path}/accs.csv", accs, delimiter=",")
        np.savetxt(f"{path}/bas.csv", bas, delimiter=",")
        np.savetxt(f"{path}/f1s.csv", f1s, delimiter=",")
        np.savetxt(f"{path}/macrof1s.csv", macrof1s, delimiter=",")
        np.savetxt(f"{path}/macrof1_inds.csv", macrof1_inds, delimiter=",")
        np.savetxt(f"{path}/eods_dif.csv", eods_dif, delimiter=",")
        np.savetxt(f"{path}/eods_abs.csv", eods_abs, delimiter=",")
        np.savetxt(f"{path}/eops.csv", eops, delimiter=",")
        print(f"Results saved in {path}")


        dict_for_table_mean = {
        "Method": method_key,
        "Acc": np.mean(accs),
        "BA": np.mean(bas),
        "F1": np.mean(f1s),
        "Macro F1": np.mean(macrof1s),
        "Macro F1 inds": np.mean(macrof1_inds),
        "EOds Diff": np.mean(eods_dif),
        "EOds Abs": np.mean(eods_abs),
        "EOp Diff": np.mean(eops),
        "EOp Abs": np.mean(np.abs(eops)),
        }
        dict_for_table_std = {
        "Method": method_key,
        "Acc": np.std(accs),
        "BA": np.std(bas),
        "F1": np.std(f1s),
        "Macro F1": np.std(macrof1s),
        "Macro F1 inds": np.std(macrof1_inds),
        "EOds Diff": np.std(eods_dif),
        "EOds Abs": np.std(eods_abs),
        "EOp Diff": np.std(eops),
        "EOp Abs": np.std(np.abs(eops)),
        }

        df_rows_mean.append(dict_for_table_mean)
        df_rows_std.append(dict_for_table_std)
    
    
    print(f"""
    #### {DATASET_USED}-{protected_attribute_used}:{dataset_orig.protected_attribute_names[0]}-{method_key} ####
    * Acc  : {np.mean(accs)*100:.2f}+-{np.std(accs):.3f}
    * BA   : {np.mean(bas)*100:.2f}+-{np.std(bas):.3f}
    * F1   : {np.mean(f1s):.3f}+-{np.std(f1s):.3f}
    * MF1  : {np.mean(macrof1s):.3f}+-{np.std(macrof1s):.3f}
    * MF1-I: {np.mean(macrof1_inds):.3f}+-{np.std(macrof1_inds):.3f}
    
    * EOds_d : {np.mean(eods_dif):.3f}+-{np.std(eods_dif):.3f}
    * EOds_a : {np.mean(eods_abs):.3f}+-{np.std(eods_abs):.3f}
    
    * EOp_d  : {np.mean(eops):.3f}+-{np.std(eops):.3f}
    * EOp_a  : {np.mean(np.abs(eops)):.3f}+-{np.std(np.abs(eops)):.3f}
    """)




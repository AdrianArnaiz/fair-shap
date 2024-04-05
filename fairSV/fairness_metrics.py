"""
Fairness metrics
"""

__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "adrian@ellisalicante.org"
__version__ = "1.0.0"


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def TPR(df, val=False):
    """
    Sensitivity 
    TPR = TP / (TP+FN) 
    TPR = P(Y^=1|Y=1)
    TPR = 1 - FNR
    EO-related
    """
    aux = "val_" if val else ""
    TP = df[aux+"true_positives"]
    FN = df[aux+"false_negatives"]
    return TP / (TP+FN)

def FNR(df, val=False):
    """
    FNR = FN / (TP+FN) 
    FNR = P(Y^=0|Y=1)
    FNR = 1 - TPR
    EO-related
    """
    aux = "val_" if val else ""
    TP = df[aux+"true_positives"]
    FN = df[aux+"false_negatives"]
    return FN / (TP+FN)

def TNR(df, val=False):
    """
    Specificity 
    TNR = TN / (TN+FP)
    TNR = P(Y^=0|Y=0)
    TNR = 1 - FPR
    EO-related
    """
    aux = "val_" if val else ""
    TN = df[aux+"true_negatives"]
    FP = df[aux+"false_positives"]
    return TN / (TN+FP)

def FPR(df, val=False):
    """
    FPR = FP / (TN+FP) 
    FPR = P(Y^=1|Y=0)
    FPR = 1 - TNR
    EO-related
    """
    aux = "val_" if val else ""
    TN = df[aux+"true_negatives"]
    FP = df[aux+"false_positives"]
    return FP / (TN+FP)

def PPV(df, val=False):
    """
    PPV = TP / (TP+FP) 
    PPV = P(Y=1|Y^=1)
    PPV = 1 - FDR
    PP-related
    """
    aux = "val_" if val else ""
    TP = df[aux+"true_positives"]
    FP = df[aux+"false_positives"]
    return TP / (TP+FP)

def max_accuracy_disparity(df, val=False):
    """ 
    Metric when A=Y, then TPR and TNR are accuracies of each group
    max(log(TPR,TNR), log(TNR,TPR))
    """
    rates = [TPR(df, val), TNR(df, val)]
    return np.log2(np.max(rates)/np.min(rates))

def get_rates(df, val=False, all=False):
    aux = "val_" if val else ""
    res = {#'loss':df[aux+'loss'],
           #'binary_accuracy':df[aux+'binary_accuracy'],
           'TPR':TPR(df, val),
           'FNR':FNR(df, val),
           'TNR':TNR(df, val),
           'FPR':FPR(df, val),
           'Max_Acc_Disparity': max_accuracy_disparity(df, val),
           'EqualOp': TPR(df, val)+TNR(df, val)-1,
           'EqualOp_bounded': (TPR(df, val)+TNR(df, val))/2,
           }

    if all:
        for c in [cc for cc in df.columns if not 'val_' in cc]:
            res[c]=df[aux+c]
    return res


def plot_metrics(df, metrics=['accuracy'], path="./", save=False):
    plt.style.use("ggplot")
    plt.figure(figsize=(10,8))
    for m in metrics:
        plt.plot(df[m], label="train_"+m)
    plt.title(f"""Training {', '.join(metrics)}""")
    plt.xlabel("Epoch #")
    plt.ylabel(', '.join(metrics))
    plt.legend(loc="lower left")
    if save: plt.savefig(f"""{path}/plot_{'_'.join(metrics)}""")


def plot_same_metric_diff_df(df_list, metrics=['accuracy'], name=['t','v'], path='./', save=False):
    plt.style.use("ggplot")
    plt.figure(figsize=(10,8))
    for m in metrics:
        for i, df in enumerate(df_list):
            plt.plot(df[m], label=name[i]+m)
    plt.title(f"""Training {', '.join(metrics)}""")
    plt.xlabel("Epoch #")
    plt.ylabel(', '.join(metrics))
    plt.legend(loc="lower left")
    if save: plt.savefig(f"""{path}/plot_{'_'.join(metrics)}""")


def get_roc_curve(label, pred, path='./', plot_fig = False):
    """
    https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
    return FPR, FNR, THRESHOLD, ACCURACY_ DISPARITY for the threshold that minimizes the latter
    """
    # get fp, tpr for each threshold
    fpr, tpr, thresholds = roc_curve(label, pred)
    auc = roc_auc_score(label, pred)
    
    # get threshold that minimizes mac_acc_disparity
    e=0.00001
    max_accuracy_disparities = np.abs(np.log2(tpr/(1-fpr+e)))
    ix = np.argmin(max_accuracy_disparities)
    best_threshold = thresholds[ix]
    
    # plot
    if plot_fig:
        plt.figure(figsize=(10,8))
        plt.plot([0,1], [0,1], linestyle='--')
        plt.plot(fpr, tpr, label='Model')
        plt.xlabel('FPR')
        plt.ylabel('TNR')
        plt.title(f"""AUC: {auc:.3f}""")
        plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Fairest', zorder=2)
        plt.legend()
        plt.savefig(path+"Roc_Curve") 


    return best_threshold, {'FPR':fpr[ix], 
                            'FNR':1-tpr[ix],
                            'TPR':tpr[ix],
                            'TNR':1-fpr[ix], 
                            'Max_Acc_Disparity':max_accuracy_disparities[ix],
                            'AUC':auc,
                            'EqualOp': tpr[ix]+(1-fpr[ix])-1,
                            'EqualOp_bounded': (tpr[ix]+(1-fpr[ix]))/2,
                            }

import numpy as np

def basic_metrics(real_labels, predicted_labels, label=None):
    """
    label is the label of interest which will be considered as the positive label.
    All the rest of the labels are considered as negative labels ans combined as if they were one.
    """
    label = real_labels[0] if label is None else label
    TP = np.sum((real_labels == label) & (predicted_labels == label))
    TN = np.sum((real_labels != label) & (predicted_labels != label))
    FP = np.sum((real_labels != label) & (predicted_labels == label))
    FN = np.sum((real_labels == label) & (predicted_labels != label))
    return TP, TN, FP, FN

def accuracy_p(real_labels=None, predicted_labels=None, basics=None, label=None):
    if basics==None:
        TP, TN, FP, FN = basic_metrics(real_labels, predicted_labels, label=label)
    else:
        TP, TN, FP, FN = basics
    return (TP+ TN) /(TP+ TN+ FP+ FN)


def f1_p(real_labels=None, predicted_labels=None, basics=None, label=None):
    # Calculate Precision and Recall
    if basics==None:
        TP, _, FP, FN = basic_metrics(real_labels, predicted_labels, label=label)
    else:
        TP, _, FP, FN = basics
    precision = TP / (TP + FP) #PPV
    recall = TP / (TP + FN) #TPR
    
    f1 = TP/ (TP+ 0.5*(FP+FN)) #2/(1/precision + 1/recall)
    return f1
    
def macro_f1_p(real_labels=None, predicted_labels=None):
    labels_unique = np.unique(real_labels)
    all_f1=[]
    for label in labels_unique:
        all_f1.append(f1_p(real_labels,predicted_labels, label=label))
    return np.mean(all_f1)



def true_class_rate(labels, predictions, class_label, attributes = None, attribute=None):
    """True class rate. Equals to TPR if class_label is the positive or TNR if class_label is the negative.
    In the case of fairness, it equals True Favorable Rate when class_label is the favorable label and true unfavorable rate when class_label is the unfavorable label.

    Args:
        labels (array): original labels
        predictions (array): model predictions
        class_label (int): class of inteterest

    Returns:
        float: true class rate
    """
    #check that labels, predictions and attrivutes are numpy arrays
    labels = np.array(labels)
    predictions = np.array(predictions)

    if attributes is None:
        return np.mean(predictions[labels==class_label]==class_label)
    else:
        attributes = np.array(attributes)
        return np.mean(predictions[(labels==class_label) & (attributes==attribute)]==class_label)


def false_class_rate(labels, predictions, class_label, attributes = None, attribute=None):
    """
    Computes the false classification rate for a given class label. Equals to FNR if class_label is the positive or FPR if class_label is the negative.

    Args:
        labels (numpy.ndarray): The true labels.
        predictions (numpy.ndarray): The predicted labels.
        class_label (int): The class label for which to compute the false classification rate.

    Returns:
        float: The false classification rate for the given class label.
    """
    labels = np.array(labels)
    predictions = np.array(predictions)
    if attributes is None:
        return np.mean(predictions[labels==class_label]!=class_label)
    else:
        attributes = np.array(attributes)
        return np.mean(predictions[(labels==class_label) & (attributes==attribute)]!=class_label)
    

def precision(labels, predictions, class_label, attributes = None, attribute=None):
    """
    Computes the precision for a given class label. PPV

    Args:
        labels (numpy.ndarray): The true labels.
        predictions (numpy.ndarray): The predicted labels.
        class_label (int): The class label for which to compute the precision.

    Returns:
        float: The precision for the given class label.
    """
    labels = np.array(labels)
    predictions = np.array(predictions)
    if attributes is None:
        return np.mean(labels[predictions==class_label]==class_label)
    else:
        attributes = np.array(attributes)
        return np.mean(labels[predictions==class_label & (attributes==attribute)]==class_label)
    
def f1_p2(labels, predictions, class_label, attributes = None, attribute=None):
    prec = precision(labels, predictions, class_label, attributes, attribute=attribute)
    tpr = true_class_rate(labels, predictions, class_label, attributes, attribute=attribute)
    return 2*((prec*tpr)/(prec+tpr)) 

def macro_f1_p2(labels, predictions, attributes = None, attribute=None, individual_nan=False):
    labels_unique = np.unique(labels)
    all_f1=[]
    for label in labels_unique:
        label_f1 = f1_p2(labels, predictions, class_label=label, attributes=attributes, attribute=attribute)
        #label_f1 = f1_p(labels, predictions, label=label)
        if individual_nan:
            label_f1 = label_f1 if not np.isnan(label_f1) else 0
        all_f1.append(label_f1)
    return np.mean(all_f1)


def compute_all_metrics(orig_labels, pred_labels, protected_attributes_dict):

    protected_attributes = protected_attributes_dict['values'] 
    fav_lab = protected_attributes_dict['favorable_label']
    unfav_lab = protected_attributes_dict['unfavorable_label']
    priv_attr = protected_attributes_dict['privileged_protected_attribute']
    unpriv_attr = protected_attributes_dict['unprivileged_protected_attribute']

    Acc = np.mean(pred_labels==orig_labels)
    
    f1 = f1_p2(orig_labels, pred_labels, class_label=fav_lab)
    f1 = 0 if np.isnan(f1) else f1
    macrof1 = macro_f1_p2(orig_labels, pred_labels)
    macrof1 = 0 if np.isnan(macrof1) else macrof1
    macrof1_ind = macro_f1_p2(orig_labels, pred_labels, individual_nan=True)
    
    TFavR = true_class_rate(orig_labels, pred_labels,fav_lab)
    TUnfavR = true_class_rate(orig_labels, pred_labels,unfav_lab)
    
    ba = (TFavR+TUnfavR)*0.5

    F_Unfav_R = false_class_rate(orig_labels, pred_labels,fav_lab)
    F_Fav_R = false_class_rate(orig_labels, pred_labels, unfav_lab)
    
    
    TFavR_Priv = true_class_rate(orig_labels, pred_labels,
                                 class_label = fav_lab, attribute = priv_attr,
                                 attributes = protected_attributes)
                                 
                                 
    TFavR_Un = true_class_rate(orig_labels, pred_labels,
                               fav_lab,
                               protected_attributes,
                               unpriv_attr)

                               
    TUnfavR_Priv = true_class_rate(orig_labels, pred_labels,
                                   unfav_lab,
                                   protected_attributes,
                                 priv_attr)
             
    TUnfavR_Un = true_class_rate(orig_labels, pred_labels,
                                 unfav_lab,
                                 protected_attributes,
                                 unpriv_attr)
    
    EOp = TFavR_Un-TFavR_Priv
    Diff_EOds = 0.5*( ((1-TUnfavR_Un)-(1-TUnfavR_Priv)) + (TFavR_Un-TFavR_Priv))
    Abs_EOds = 0.5*( np.abs((1-TUnfavR_Un)-(1-TUnfavR_Priv)) + np.abs(TFavR_Un-TFavR_Priv))

    return {'acc': Acc, 'f1': f1, 'macrof1': macrof1, 'macrof1_ind':macrof1_ind, 'ba': ba, 'EOp': EOp, 'Diff_EOds': Diff_EOds, 'Abs_EOds': Abs_EOds}
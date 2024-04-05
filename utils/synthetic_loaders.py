"""
Code from https://github.com/mbilalzafar/fair-classification/blob/master/disparate_impact/synthetic_data_demo/generate_synthetic_data.py
Fairness Beyond Disparate Treatment & Disparate Impact: Learning Classification without Disparate Mistreatment
Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Rodriguez, Krishna P. Gummadi.
26th International World Wide Web Conference (WWW), Perth, Australia, April 2017.
"""

import math
import numpy as np
import matplotlib.pyplot as plt # for plotting stuff
from random import seed, shuffle
from scipy.stats import multivariate_normal # generating synthetic data
SEED = 1122334455
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

def generate_synthetic_data_zafar(n_samples, plot_data=False):

    """
        Code for generating the synthetic data.
        We will have two non-sensitive features and one sensitive feature.
        A sensitive feature value of 0.0 means the example is considered to be in protected group (e.g., female) and 1.0 means it's in non-protected group (e.g., male).
    """

    disc_factor = math.pi / 4.0 # this variable determines the initial discrimination in the data -- decraese it to generate more discrimination

    def gen_gaussian(mean_in, cov_in, class_label):
        nv = multivariate_normal(mean = mean_in, cov = cov_in)
        X = nv.rvs(n_samples)
        y = np.ones(n_samples, dtype=float) * class_label
        return nv,X,y

    """ Generate the non-sensitive features randomly """
    # We will generate one gaussian cluster for each class
    mu1, sigma1 = [2, 2], [[5, 1], [1, 5]]
    mu2, sigma2 = [-2,-2], [[10, 1], [1, 3]]
    nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1) # positive class
    nv2, X2, y2 = gen_gaussian(mu2, sigma2, -1) # negative class

    # join the posisitve and negative class clusters
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    # shuffle the data
    perm = list(range(0,n_samples*2))
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    
    rotation_mult = np.array([[math.cos(disc_factor), -math.sin(disc_factor)], [math.sin(disc_factor), math.cos(disc_factor)]])
    X_aux = np.dot(X, rotation_mult)


    """ Generate the sensitive feature here """
    x_control = [] # this array holds the sensitive feature value
    for i in range (0, len(X)):
        x = X_aux[i]

        # probability for each cluster that the point belongs to it
        p1 = nv1.pdf(x)
        p2 = nv2.pdf(x)
        
        # normalize the probabilities from 0 to 1
        s = p1+p2
        p1 = p1/s
        p2 = p2/s
        
        r = np.random.uniform() # generate a random number from 0 to 1

        if r < p1: # the first cluster is the positive class
            x_control.append(1.0) # 1.0 means its male
        else:
            x_control.append(0.0) # 0.0 -> female

    x_control = np.array(x_control)

    """ Show the data """
    if plot_data:
        num_to_draw = 200 # we will only draw a small number of points to avoid clutter
        x_draw = X[:num_to_draw]
        y_draw = y[:num_to_draw]
        x_control_draw = x_control[:num_to_draw]

        X_s_0 = x_draw[x_control_draw == 0.0]
        X_s_1 = x_draw[x_control_draw == 1.0]
        y_s_0 = y_draw[x_control_draw == 0.0]
        y_s_1 = y_draw[x_control_draw == 1.0]
        plt.scatter(X_s_0[y_s_0==1.0][:, 0], X_s_0[y_s_0==1.0][:, 1], color='green', marker='x', s=30, linewidth=1.5, label= "Prot. +ve")
        plt.scatter(X_s_0[y_s_0==-1.0][:, 0], X_s_0[y_s_0==-1.0][:, 1], color='red', marker='x', s=30, linewidth=1.5, label = "Prot. -ve")
        plt.scatter(X_s_1[y_s_1==1.0][:, 0], X_s_1[y_s_1==1.0][:, 1], color='green', marker='o', facecolors='none', s=30, label = "Non-prot. +ve")
        plt.scatter(X_s_1[y_s_1==-1.0][:, 0], X_s_1[y_s_1==-1.0][:, 1], color='red', marker='o', facecolors='none', s=30, label = "Non-prot. -ve")

        
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off') # dont need the ticks to see the data distribution
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.legend(loc=2, fontsize=15)
        plt.xlim((-15,10))
        plt.ylim((-10,15))

    x_control = {"s1": x_control} # all the sensitive features are stored in a dictionary
    return X,y,x_control


def make_unfair_classif(n = 500, n_features = 5, n_classes = 3, n_clusters = 2, p = 0.75):
    """
    the first half of classes will have p of probability having S = 1, the other half will have 1-p having S = 1
    """
    # rmks : all variances here are identities
    
    # centroids
    #centroids = np.random.uniform(-n_classes/2, n_classes/2, (n_classes, n_features))
    centroids = np.random.uniform(-1, 1, (n_classes, n_features))
    
    for cl in range(n_classes):
        
        # generate means based on centroids
        means = np.zeros((n_clusters, n_features))
        for comp in range(n_clusters):
            #means[comp] = centroids[cl] + (n_classes/3)*np.random.randn(n_features)
            means[comp] = centroids[cl] + np.random.randn(n_features)
            
        # Weight of each component, in this case all of them are equal
        weights = np.ones(n_clusters) / n_clusters
        
        # A stream of indices from which to choose the component
        mixture_idx = np.random.choice(n_clusters, size=n, replace=True, p=weights)
        
        # Data
        M = np.zeros((len(mixture_idx), n_features))
        for i in range(n_clusters):
            #M[mixture_idx == i] = np.random.multivariate_normal(means[i], (n_classes/3)*np.eye(n_features), sum(mixture_idx == i))
            M[mixture_idx == i] = np.random.multivariate_normal(means[i], np.eye(n_features), sum(mixture_idx == i))
        
        ym = cl*np.ones(len(mixture_idx))
        
        if cl == 0:
            M_all = M
            y_all = ym
        else:
            M_all = np.concatenate((M_all, M), axis = 0)
            y_all = np.concatenate((y_all, ym), axis = 0)
    
    # prepare contaminations
    M_all = np.concatenate((M_all, -1*np.ones((len(M_all), 1))), axis=1)
    
    # add contaminations(p) for the first half classes, and keep contaminations(1-p) for the rest
    for cl in np.arange(n_classes):
        if cl < n_classes//2:
            index = (y_all == cl)
            n_index = np.sum(index)
            M_all[index, -1] = 2*np.random.binomial(1, p, n_index)-1
        else:
            index = (y_all == cl)
            n_index = np.sum(index)
            M_all[index, -1] = 2*np.random.binomial(1, 1-p, n_index)-1
    
    # shuffle
    shuffle_indexes = np.arange(len(y_all))
    np.random.shuffle(shuffle_indexes)
    M_all = M_all[shuffle_indexes]
    y_all = y_all[shuffle_indexes]
        
    return(M_all, y_all)

def make_unfair_poolclassif(n, n_features, n_classes, n_clusters, n_pool = 0, p = 0.7):
    """
    make unfair classification dataset (labeled dataset) with pool data (unlabeled dataset)
    """
    X, y = make_unfair_classif(n = n + n_pool, n_features = n_features, n_classes = n_classes, n_clusters = n_clusters, p = p)
    if n_pool > 0:
        X_pool = X[:n_classes*n_pool]
    else:
        X_pool = None
    X = X[n_classes*n_pool:]
    y = y[n_classes*n_pool:]
    
    return(X, y, X_pool)

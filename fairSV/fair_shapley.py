__author__ = "Adrian Arnaiz-Rodriguez"
__email__ = "adrian@ellisalicante.org"
__version__ = "1.0.0"

import os
from matplotlib import style
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.manifold import TSNE


class FairShapley():
    """
    Class for computing and visualizing shapley values
    """

    def __init__(self, X_train, Y_train, X_test, Y_test, protected_attributes_dict=None, save_folder=None, show_plot=None, SV=None, calculate_2dim=True) -> None:
        """_summary_

        Args:
            X_train (array): _description_
            Y_train (array): _description_
            X_test (array): _description_
            Y_test (array): _description_
            protected_attributes_dict (dict, optional): Protected attribute from TEST SET to calculate conditional probs. If None then A=Y. Defaults to None.
                Keys: 'values' (list of values of the protected attribute), 'privileged_protected_attribute' (privileged value of the protected attribute), 'unprivileged_protected_attribute' (unprivileged value of the protected attribute),
                'favorable_label' (value of favorable label), 'unfavorable_label' (value of unfavorable label)
            save_folder (_type_, optional): _description_. Defaults to None.
            show_plot (_type_, optional): _description_. Defaults to None.
            SV (_type_, optional): _description_. Defaults to None.
            calculate_2dim (bool, optional): _description_. Defaults to True.
        """
        self.x_train = self.X_tr_2dim = X_train
        self.y_train = Y_train
        self.x_test = self.X_tst_2dim = X_test
        self.y_test = Y_test

        self.protected_attributes_dict = protected_attributes_dict #array of protected attributes

        #x2dim = TSNE(n_components=2, learning_rate='auto').fit_transform(np.vstack([X_train,X_test]))
        if X_train.shape[1]>2 and calculate_2dim:
            print('# Calculating TSNE')
            tsne = TSNE(n_components=2, verbose=0, perplexity=45, n_iter=450)
            tsne_results = tsne.fit_transform(np.vstack([X_train,X_test]))
            self.X_tr_2dim = tsne_results[:len(X_train)]
            self.X_tst_2dim = tsne_results[len(X_train):]


        self.N = self.x_train.shape[0]
        self.N_tst = self.x_test.shape[0]

        self.SV = SV            
        self.sv_tpr = None
        self.sv_tnr = None
        self.sv_fnr = None
        self.sv_fpr = None
        self.sv_acc = None
        self.sv_max_acc_disp_diff = None
        self.sv_max_acc_disp_log = None 
        self.sv_tp_diff = None
        self.sv_eop = None
        self.sv_eop_bounded = None

        if SV is not None:
            self.get_sv_arrays()

        self.show_plot = show_plot
        self.save_folder = save_folder
        if self.save_folder is not None and not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        sns.set_theme(style="darkgrid", palette="deep", font='serif')
        sns.set_context("paper", font_scale=1.5)
        #sns.set(rc={'axes.facecolor':'white'})
        #plt.figure(facecolor='white') 

    
    def _single_point_shapley(self, K, xt_query, y_t_label):
        """
        Calculation of SV of each training data point wrt one test point

        Args:
            K (int): K parameter for KNN
            xt_query (np.array): test data point
            y_t_label (np.array - int): train data point

        Returns:
            np.array: R^n. SV of each training data point wrt one test point
        """
        distance1 = np.sum(np.square(self.x_train-xt_query), axis=1) # euclidean
        alpha = np.argsort(distance1)
        y = self.y_train
        #pred = [self.y_train[alpha[i]] for i in range(K)]
        #prob = pred.count(y_t_label)/len(pred)
        shapley_arr = np.zeros(self.N, dtype='float64')
        for i in range(self.N-1, -1, -1): 
            if i == self.N-1:
                shapley_arr[alpha[i]] = np.float64(int(y[alpha[i]] == y_t_label) /self.N)
            else:
                shapley_arr[alpha[i]] = np.float64(shapley_arr[alpha[i+1]] + \
                ( int(y[alpha[i]]==y_t_label) - int(y[alpha[i+1]]==y_t_label) )/K * min(K,i+1)/(i+1))
        return shapley_arr


    def get_SV_matrix(self, K):
        """
        Calculation of SV of each training data point wrt each test point

        Args:
            K (int): K parameter for KNN

        Returns:
            _type_: R^(n x m)
        """
        N_tst = self.x_test.shape[0]
        shapley_mat = np.zeros((self.N, N_tst), dtype='float64')
        for j in tqdm(range(N_tst), desc='Calculating SV Matrix'):
            point_test = self.x_test[j]
            label_test = self.y_test[j]
            sv_xtest_array = self._single_point_shapley(K, point_test, label_test)
            #print('Array for test point',j, point_test, label_test,':',sv_xtest_array)
            shapley_mat[:,j] = sv_xtest_array
        self.SV = shapley_mat

        #sv arrays
        self.get_sv_arrays() #tpr, tnr and so on

        return shapley_mat


    def get_sv_arrays(self):
        """
        Generate sv array for tpr and tnr
        Raises:
            Exception: if SV matrix is not already calculated
        """
        if self.SV is None: raise Exception('SV not calculated. Please call get_SV_matrix')
        self.sv_acc = self.SV.mean(axis=1) # Original  
        
        if not self.protected_attributes_dict:
            # When protected attributes are not given, then A=Y
            self.sv_tpr = self.SV[:,self.y_test==1].mean(axis=1)
            self.sv_fnr = (1/self.N) - self.sv_tpr
            self.sv_tnr = self.SV[:,self.y_test==0].mean(axis=1)
            self.sv_fpr = (1/self.N) - self.sv_tnr       
            tpr = self.sv_tpr.sum()
            tnr = self.sv_tnr.sum()
            self.sv_max_acc_disp_diff = self.sv_tpr - self.sv_tnr if tpr>tnr else self.sv_tnr - self.sv_tpr

            self.sv_tp_diff = self.sv_tpr - self.sv_tnr
            self.sv_tp_diff[self.y_train==0] = self.sv_tp_diff[self.y_train==0] * -1

            self.sv_eop = self.sv_tpr + self.sv_tnr - (1/self.N) #[EOp [-1,1]]
            self.sv_eop_bounded = (self.sv_tpr + self.sv_tnr)/2  #[EOp [ 0,1]]

            new_tpr = tpr - self.sv_tpr
            new_tnr = tnr - self.sv_tnr
            if tpr>tnr:
                acc_disp_log = np.log2(tpr/tnr)
                self.sv_max_acc_disp_log = acc_disp_log - np.log2(new_tpr/new_tnr)
                #self.sv_max_acc_disp = np.log2(np.abs(self.sv_tpr/self.sv_tnr)) ## LOG
            else:
                acc_disp_log = np.log2(tnr/tpr)
                self.sv_max_acc_disp_log = acc_disp_log - np.log2(new_tnr/new_tpr)
                #self.sv_max_acc_disp = np.log2(np.abs(self.sv_tnr/self.sv_tpr)) ## LOG
        else:
            """
            Statistical parity difference
            Disparate impact
            Average odds difference
            Average absoulte odds difference
            Equal opportunity difference
            """
            #values
            protected_attributes = self.protected_attributes_dict['values']
            #protected attrs value por privileged and unprivileged group
            privileged_attr_value = self.protected_attributes_dict['privileged_protected_attribute']
            unprivileged_attr_value = self.protected_attributes_dict['unprivileged_protected_attribute']
            #labels: what is favorable and unfavorable label
            fav_label = self.protected_attributes_dict['favorable_label']
            unfav_label = self.protected_attributes_dict['unfavorable_label']

            # General tpr and tnr
            self.sv_tpr = self.SV[:,self.y_test==fav_label].mean(axis=1)
            self.sv_fnr = (1/self.N) - self.sv_tpr
            self.sv_tnr = self.SV[:,self.y_test==unfav_label].mean(axis=1)
            self.sv_fpr = (1/self.N) - self.sv_tnr 

            #conditioned tpr and tnr
            f_fav_priv = (self.y_test==fav_label) & (protected_attributes == privileged_attr_value)
            self.sv_tpr_p = self.SV[:,f_fav_priv].mean(axis=1) 

            f_fav_unpriv = (self.y_test==fav_label) & (protected_attributes == unprivileged_attr_value)
            self.sv_tpr_u = self.SV[:,f_fav_unpriv].mean(axis=1) 

            f_unfav_priv = (self.y_test==unfav_label) & (protected_attributes == privileged_attr_value)
            self.sv_tnr_p = self.SV[:,f_unfav_priv].mean(axis=1) 

            f_unfav_unpriv = (self.y_test==unfav_label) & (protected_attributes == unprivileged_attr_value)
            self.sv_tnr_u = self.SV[:,f_unfav_unpriv].mean(axis=1) 

            self.sv_fpr_p = (1/self.N) - self.sv_tnr_p
            self.sv_fpr_u = (1/self.N) - self.sv_tnr_u
            self.sv_fnr_p = (1/self.N) - self.sv_tpr_p
            self.sv_fnr_u = (1/self.N) - self.sv_tpr_u
            
            #self.sv_statistical_parity_diff = None #Not strictley based on TPR

            #self.sv_disparate_impact = None

            # 0.5*( (FPR_u - FPR_p) + (TPR_u - TPR_p) )
            self.sv_average_odds_difference = 0.5*((self.sv_fpr_u-self.sv_fpr_p)+(self.sv_tpr_u-self.sv_tpr_p))

            self.sv_average_absoulte_odds_difference = 0.5*(np.abs(self.sv_fpr_u-self.sv_fpr_p)+np.abs(self.sv_tpr_u-self.sv_tpr_p))

            self.sv_equal_opportunity_difference = self.sv_tpr_u -  self.sv_tpr_p



    def get_best_K(self, max_k=30):
        """
        Calculation of KNN with different k and return the one with higher probability accuracy.
        
        Args:
            plot (bool, optional): print Acc-K graph. Defaults to True.

        Returns:
            best_k: k with most accuracy
            best_pred_prob_acc: probaccuracy of best k: mean of probabilities for the actual class for all test set
            best_pred_probs: probability of actual class for each test point
        
        """
        k_to_test = range(2, min(self.N+1, max_k))
        best_k = 0
        best_pred_prob_acc = 0
        best_pred_probs = 0
        preds_array = []
        for k in tqdm(k_to_test, desc='Finding best k'):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.x_train,self.y_train)
            pred_probs_k = knn.predict_proba(self.x_test) ## n_test x n_clases- probs for all classes for each sample
            preds_actual_class = pred_probs_k[range(self.N_tst),self.y_test] ##probs of each sample for the actual class
            pred_prob_acc_k = np.mean(preds_actual_class) ##mean of probs for the actual class
            preds_array.append(pred_prob_acc_k)
            if pred_prob_acc_k>=best_pred_prob_acc:
                best_pred_prob_acc= pred_prob_acc_k
                best_pred_probs = preds_actual_class
                best_k = k
        
        if self.show_plot:
            plt.figure(figsize=(10,6))
            plt.plot(k_to_test, preds_array, color='blue', linestyle='dashed', marker='o',
                                markerfacecolor='red', markersize=10)
            plt.title(f'Prob_ACC vs. K - K:{best_k} - Prob_acc={best_pred_prob_acc:.2e}')
            plt.xlabel('K')
            plt.ylabel('Prob_ACC Rate')
            plt.show()
            plt.close()
        return best_k, best_pred_prob_acc, best_pred_probs
    

    def calculate_tp_fn_tn_fp(self,df, th=0.5, rates_prob=True):
        tp = len(df[(df['Pred_prob_lbl_1']>=th) & (df['Actual_lbl']==1)])
        fn = len(df[(df['Pred_prob_lbl_1']<th) & (df['Actual_lbl']==1)])

        tn = len(df[(df['Pred_prob_lbl_1']<th) & (df['Actual_lbl']==0)])
        fp = len(df[(df['Pred_prob_lbl_1']>=th) & (df['Actual_lbl']==0)])

        return tp, fn, tn, fp

    def get_clf_metrics(self, actual_lables, pred_prob_c1, th):
        """
        Calculate classification metric that depends on a selected threshold
        e.g.: TPR= TP/(TP+FN) but TP= |Y^>t|Y=1|

        Args:
            actual_lables (np.array): actual labels
            pred_prob_c1 (np.array): probability of prediction for class 1 (positive)
            th (0<float<1): decission threshold

        Returns:
            dict: clf metrics at thresolhd th
        """
        df_actual_and_pred = np.column_stack((actual_lables, pred_prob_c1))
        df_actual_and_pred = pd.DataFrame(df_actual_and_pred, columns=['Actual_lbl', 'Pred_prob_lbl_1'])

        tp, fn, tn, fp = self.calculate_tp_fn_tn_fp(df_actual_and_pred, th=th)        
        clf_metrics = {'acc':(tp+tn)/(tp+tn+fp+fn),
                       'TPR': tp/(tp+fn), 
                       'TNR': tn/(tn+fp),
                       'FPR': fp/(tn+fp),
                       'FNR': fn/(tp+fn),
                       'EqualOp': tp/(tp+fn) + tn/(tn+fp) - 1,
                       'EqualOp_bounded': (tp/(tp+fn) +  tn/(tn+fp))/2,
                       'Max_acc_disp_diff': np.abs((tp/(tp+fn)) -  (tn/(tn+fp)))
                       }
        try:
            clf_metrics['Max_acc_disp_log'] = np.abs( np.log2( clf_metrics['TPR']/clf_metrics['TNR'] ) )
        except ZeroDivisionError:
            clf_metrics['Max_acc_disp_log'] = 'inf'
        try:
            clf_metrics['Max_acc_disp_ratio'] = (np.max([clf_metrics['TPR'],clf_metrics['TNR']]) / np.min([clf_metrics['TPR'],clf_metrics['TNR']]))-1
        except ZeroDivisionError:
            clf_metrics['Max_acc_disp_ratio'] = 'inf'

        return clf_metrics


    def get_th_independent_clf_metrics(self, pred_prob_actual_class):
        mean_prob_acc_k = np.mean(pred_prob_actual_class) # R
        tpr_prob = np.mean(pred_prob_actual_class[self.y_test==1])
        fnr_prob = 1- tpr_prob
        tnr_prob = np.mean(pred_prob_actual_class[self.y_test==0])
        fpr_prob = 1 - tnr_prob
        max_acc_disp_prob = np.abs(np.log2(tpr_prob/tnr_prob))

        metrics = {'acc':mean_prob_acc_k,
                    'TPR':tpr_prob,
                    'TNR':tnr_prob,
                    'FPR':fpr_prob,
                    'FNR':fnr_prob,
                    'EqualOp': tpr_prob+tnr_prob-1,
                    'EqualOp_bounded': (tpr_prob+tnr_prob)/2,
                    'Max_acc_disp_diff': np.abs(tpr_prob-tnr_prob),
                    'Max_acc_disp_ratio': np.max([tpr_prob,tnr_prob])/np.min([tpr_prob,tnr_prob]) - 1,
                    'Max_acc_disp_log': max_acc_disp_prob}

        return metrics
        # mean of probs for the actual (right) class
        

    def do_knn(self, k, model='knn'):
        """
        Classification using knn and returning all emasyres and optimized mesaures (regarding max_acc_disp)
        Args:
            k (int): k for knn
            th (float, optional): threshold. Defaults to 0.5.

        Returns:
            clf_metrics: dict with classification results
        """
        # Train and test
        if model=='knn':
            knn = KNeighborsClassifier(n_neighbors=k)
        elif model == 'GBC':
            knn= GradientBoostingClassifier()
            
        knn.fit(self.x_train,self.y_train)
        # probs for all classes for each sample - n_test x n_clases 
        pred_probs_k = knn.predict_proba(self.x_test) #R^m x |classes|
        # prob of each sample for the actual (right) class
        preds_actual_class = pred_probs_k[range(self.N_tst),self.y_test] # R^m

        # get measures fot th=0.5, th=opt and th_independent
        ## th=0.5
        def_th_clf_metrics = self.get_clf_metrics(self.y_test, pred_probs_k[:,1], th=0.5)
        ## th=opt
        best_th, auc = self.get_roc_curve(self.y_test, pred_probs_k[:,1])
        best_th_clf_metrics = self.get_clf_metrics(self.y_test, pred_probs_k[:,1], th=best_th)
        ## th independent
        indep_th_clf_metrics = self.get_th_independent_clf_metrics(preds_actual_class) 
        indep_th_clf_metrics['AUC'] = def_th_clf_metrics['AUC'] = best_th_clf_metrics['AUC'] = auc
               
        clf_metrics = {'def': def_th_clf_metrics,
                        f'opt_{best_th:.2f}':best_th_clf_metrics,
                        'indep':indep_th_clf_metrics,
                        }
        

        return clf_metrics, pred_probs_k


    def get_roc_curve(self, label, pred):
        """
        https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
        return FPR, FNR, THRESHOLD, ACCURACY_ DISPARITY for the threshold that minimizes the latter
        """
        # get fp, tpr for each threshold
        fpr, tpr, thresholds = roc_curve(label, pred)
        auc = roc_auc_score(label, pred)
        
        # get threshold that minimizes max_acc_disparity
        e=0.000001
        max_accuracy_disparities = np.abs(np.log2(tpr/(1-fpr+e)))
        ix = np.argmin(max_accuracy_disparities)
        best_threshold = thresholds[ix]
        return best_threshold, auc


    def plot_data_sns(self, sv_array = None, size=60, test=False, size_test=30, protected_attr=None, save=False, title=''):
        """plot train and test data in 2 dimensions, with different size options:
            - normal
            - custom
            - proportional to positive SV
            - proportional to negative SV

        Args:
            test (bool, optional): Whether plot or no test data
            size (int, optional): size of dots. Defaults to 30.

        Raises:
            Exception: data is not 2 dimensional
        """
        if sv_array is None and type(size)==str: 
            sv_array = self.sv_acc.copy()
        if not sv_array is None: 
            original_array = sv_array.copy()

        fig = plt.figure()
        X_tr = self.X_tr_2dim.copy()
        X_tst = self.X_tst_2dim.copy()
            
        colors = sns.color_palette()[:2]
        df_train = pd.DataFrame({'x_cord':X_tr[:,0], 'y_cord':X_tr[:,1],'lab':self.y_train})
        
        fig_title = title
        if type(size)==str:
            max_sv = np.max(original_array)
            
            sv_array = (sv_array/max_sv) * 400 # From [0.8,0,-0.2] to [1,0,-0.25] to [400,0,-100]
            if size=='sv_pos':
                sv_array[sv_array<0] = 0 # do not print negative SV
                sv_array[sv_array>0] = sv_array[sv_array>0] + 20 #minimum size of positive SV
                size = sv_array.astype('float64')
                fig_title = f"{title} - Positive SV\n"
                edgecolor='lime'
            elif size=='sv_neg':
                sv_array[sv_array>0] = 0  #do not print positive SV
                sv_array[sv_array<0] = sv_array[sv_array<0] -20 #minimum size of positive SV
                size = (sv_array*-1).astype('float64')
                fig_title = f"{title} - Negative SV\n"
                edgecolor='darkred'
            else:
                raise Exception("Not valid size for nodes")

            sv_array[sv_array>500]=500 # limit for extra big points
            fig_title = fig_title + f"""Max $SV=${np.max(original_array):.2e} - min $SV=${np.min(original_array):.2e}"""
        
        if protected_attr:
            style_markers = ['o','x']
            markers = [style_markers[i] for i in protected_attr]
            axs = sns.jointplot(data=df_train, x='x_cord', y='y_cord', hue='lab',
                                kind='scatter', marker=markers, s=size, height=8)
            axs.fig.suptitle(fig_title)
        else:
            axs = sns.jointplot(data=df_train, x='x_cord', y='y_cord', hue='lab',
                                kind='scatter', marker='.', s=size, height=8)
            axs.fig.suptitle(fig_title)
        
        if type(size)!=int:
            idx_max = np.argsort(size)[::-1][:min(50, len(self.x_test)-1)]
            axs.ax_joint.scatter(X_tr[idx_max,0], X_tr[idx_max,1], marker='.',
                                s=size[idx_max], facecolor="none", edgecolors=edgecolor, linewidth=2)

        if test:
            colors_test = [colors[lab] for lab in self.y_test]

            if protected_attr:
                style_markers = ['o','x']
                markers = [style_markers[i] for i in protected_attr]
                axs = sns.jointplot(data=df_train, x='x_cord', y='y_cord', hue='lab',
                                    kind='scatter', marker=markers, s=size, height=8)
                axs.fig.suptitle(fig_title)
            else:
                axs.ax_joint.scatter(X_tst[:,0], X_tst[:,1], c=colors_test, marker='1', s=size_test)
                           
        if save and not self.save_folder is None: plt.savefig(f"""{self.save_folder}/plot_{title}""")
        if self.show_plot: plt.show()
        plt.close()

    
    def plot_sv_hist(self, sv_array=None, log=True, title='', save=False):
        """
        Plot hist of the array given as parameter. Designed for SV histogram by class
        """
        df = pd.DataFrame({
            'SV': sv_array,
            'lab': self.y_train
        })
        plt.figure()
        sns.histplot(data=df, x='SV',hue='lab', bins=int(np.sqrt(df.shape[0])),
                     log_scale=(False, log)).set_title("SV hist - "+title.upper())
        if save and not self.save_folder is None: plt.savefig(f"""{self.save_folder}/plot_{title}""")
        if self.show_plot: plt.show()
        plt.close()


    def get_sv_summary(self):
        sv_arrays = [s for s in dir(self) if s .startswith('sv_')]
        data = {}
        for sv_arr_str in sv_arrays:
            data[sv_arr_str] = eval(f"""self.{sv_arr_str}.sum()""")
        return pd.DataFrame.from_dict(data, orient='index', columns = ['sum']) 
 
      
    def whole_process(self, save=False, show_plot=False):
        self.show_plot = show_plot
        # Get k with best probability
        k,prob_acc,probs=self.get_best_K()
        print('Best K:',k)
        # Get SV matrix with previous k
        SV_m = self.get_SV_matrix(K=k)
        #Get graph of feature space
        self.plot_data_sns(test=True, save=save, title='FeatureSpace')

        # get graphs of positive and negative SV for each measure
        # get histogram of sv for each measure
        sv_arrays = sorted([s for s in dir(self) if s .startswith('sv_')])
        for sv_arr_str in sv_arrays:
            array = eval(f'self.{sv_arr_str}')
            self.plot_data_sns(sv_array = array, size='sv_pos',test=True, save=save, title=sv_arr_str.upper()+'_pos')
            self.plot_data_sns(sv_array = array, size='sv_neg',test=True, save=save, title=sv_arr_str.upper()+'_neg')
            self.plot_sv_hist(sv_array = array, title=sv_arr_str, save=save)
  
        df_sv_arrays_sum = self.get_sv_summary()
        print(df_sv_arrays_sum)
        metrics,_= self.do_knn(k=k)
        self.metrics = metrics
        return metrics


if __name__ == "__main__":
    import sklearn.datasets as dt
    X_train, y_train = dt.make_blobs(n_samples=[1000,1000], centers=[[-1,0],[1,0]], 
                    cluster_std=(.5,.5), n_features=2, random_state=42)
    X_test, y_test = dt.make_blobs(n_samples=[200,200], centers=[[-1,0],[1,0]], 
                        cluster_std=(.5,.5), n_features=2, random_state=42)

    fsv_b_synth = FairShapley(X_train, y_train, X_test, y_test, save_folder='test_figures_SV/', show_plot=False)
    metrics = fsv_b_synth.whole_process(save=True)
    print(pd.DataFrame(metrics).T)
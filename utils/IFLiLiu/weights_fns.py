# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import random
import numpy as np
from typing import Sequence
import gurobipy as gp

#from dataset import fetch_data, DataTemplate
#from eval import Evaluator
from .model import LogisticRegression
from .fair_fn import grad_ferm, grad_dp, loss_ferm, loss_dp


def lp(fair_infl: Sequence, util_infl: Sequence, fair_loss: float, alpha: float, beta: float,
       gamma: float) -> np.ndarray:
    num_sample = len(fair_infl)
    max_fair = sum([v for v in fair_infl if v < 0.])
    max_util = sum([v for v in util_infl if v < 0.])

    #print("Maximum fairness promotion: %.5f; Maximum utility promotion: %.5f;" % (max_fair, max_util))

    all_one = np.array([1. for _ in range(num_sample)])
    fair_infl = np.array(fair_infl)
    util_infl = np.array(util_infl)
    model = gp.Model()
    x = model.addMVar(shape=(num_sample,), lb=0, ub=1)

    if fair_loss >= -max_fair:
        #print("=====> Fairness loss exceeds the maximum availability")
        model.addConstr(util_infl @ x <= 0. * max_util, name="utility")
        model.addConstr(all_one @ x <= alpha * num_sample, name="amount")
        model.setObjective(fair_infl @ x)
        model.optimize()
    else:
        model.addConstr(fair_infl @ x <= beta * -fair_loss, name="fair")
        model.addConstr(util_infl @ x <= gamma * max_util, name="util")
        model.setObjective(all_one @ x)
        model.optimize()

    #print("Total removal: %.5f; Ratio: %.3f%%\n" % (sum(x.X), (sum(x.X) / num_sample) * 100))

    return 1 - x.X


def get_IF_weights(x_train, y_train, x_val, y_val, s_val,
                   metric,
                   l2_reg, seed, alpha, beta, gamma):
    """
    https://github.com/brandeis-machine-learning/influence-fairness/blob/main/main.py
    """

    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)

    """ initialization"""
    model = LogisticRegression(l2_reg=l2_reg)
    #val_evaluator, test_evaluator = Evaluator(s_val, "val"), Evaluator(data.s_test, "test")

    """ vanilla training """

    model.fit(x_train, y_train)
    if metric == "eop":
        ori_fair_loss_val = loss_ferm(model.log_loss, x_val, y_val, s_val)
    elif metric == "dp":
        pred_val, _ = model.pred(x_val)
        ori_fair_loss_val = loss_dp(x_val, s_val, pred_val)
    else:
        raise ValueError
    ori_util_loss_val = model.log_loss(x_val, y_val)

    """ compute the influence and solve lp """

    pred_train, _ = model.pred(x_train)

    train_total_grad, train_indiv_grad = model.grad(x_train, y_train)
    util_loss_total_grad, acc_loss_indiv_grad = model.grad(x_val, y_val)
    if metric == "eop":
        fair_loss_total_grad = grad_ferm(model.grad, x_val, y_val, s_val)
    elif metric == "dp":
        fair_loss_total_grad = grad_dp(model.grad_pred, x_val, s_val)
    else:
        raise ValueError

    hess = model.hess(x_train)
    util_grad_hvp = model.get_inv_hvp(hess, util_loss_total_grad)
    fair_grad_hvp = model.get_inv_hvp(hess, fair_loss_total_grad)

    util_pred_infl = train_indiv_grad.dot(util_grad_hvp)
    fair_pred_infl = train_indiv_grad.dot(fair_grad_hvp)

    sample_weight = lp(fair_pred_infl, util_pred_infl, ori_fair_loss_val, alpha, beta, gamma)

    return sample_weight
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

from abc import ABC, abstractmethod
from typing import Sequence, Tuple
import numpy as np
import sklearn.neural_network
import sklearn.linear_model
import sklearn.metrics
from scipy.linalg import cho_solve, cho_factor

"""import torch
from torch import nn
from torch import Tensor"""

#from eval import Evaluator
#from utils import nearest_pd


class IFBaseClass(ABC):
    """ Abstract base class for influence function computation on logistic regression """

    @staticmethod
    def set_sample_weight(n: int, sample_weight: np.ndarray or Sequence[float] = None) -> np.ndarray:
        if sample_weight is None:
            sample_weight = np.ones(n)
        else:
            if isinstance(sample_weight, np.ndarray):
                assert sample_weight.shape[0] == n
            elif isinstance(sample_weight, (list, tuple)):
                assert len(sample_weight) == n
                sample_weight = np.array(sample_weight)
            else:
                raise TypeError

            assert min(sample_weight) >= 0.
            assert max(sample_weight) <= 2.

        return sample_weight

    @staticmethod
    def check_pos_def(M: np.ndarray) -> bool:
        pos_def = np.all(np.linalg.eigvals(M) > 0)
        print("Hessian positive definite: %s" % pos_def)
        return pos_def

    @staticmethod
    def get_inv_hvp(hessian: np.ndarray, vectors: np.ndarray, cho: bool = True) -> np.ndarray:
        if cho:
            return cho_solve(cho_factor(hessian), vectors)
        else:
            hess_inv = np.linalg.inv(hessian)
            return hess_inv.dot(vectors.T)

    @abstractmethod
    def log_loss(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None,
                 l2_reg: bool = False) -> float:
        raise NotImplementedError

    @abstractmethod
    def grad(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None,
             l2_reg: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """ Return the sum of all gradients and every individual gradient """
        raise NotImplementedError

    @abstractmethod
    def grad_pred(self, x: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None) -> Tuple[
        np.ndarray, np.ndarray]:
        """ Return the sum of all gradients and every individual gradient """
        raise NotImplementedError

    @abstractmethod
    def hess(self, x: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None,
             check_pos_def: bool = False) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def pred(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Return the predictive probability and class label """
        raise NotImplementedError

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray or Sequence[float] = None) -> None:
        raise NotImplementedError


class LogisticRegression(IFBaseClass):
    """
    Logistic regression: pred = sigmoid(weight^T @ x + bias)
    Currently only support binary classification
    Borrowed from https://github.com/kohpangwei/group-influence-release
    """

    def __init__(self, l2_reg: float, fit_intercept: bool = False):
        self.l2_reg = l2_reg
        self.fit_intercept = fit_intercept
        self.model = sklearn.linear_model.LogisticRegression(
            penalty="l2",
            C=(1. / l2_reg),
            fit_intercept=fit_intercept,
            tol=1e-8,
            solver="lbfgs",
            max_iter=2048,
            multi_class="ovr",
            warm_start=False,
        )

    def log_loss(self, x, y, sample_weight=None, l2_reg=False, eps=1e-16):
        sample_weight = self.set_sample_weight(x.shape[0], sample_weight)

        pred, _, = self.pred(x)
        log_loss = - y * np.log(pred + eps) - (1. - y) * np.log(1. - pred + eps)
        log_loss = sample_weight @ log_loss
        if l2_reg:
            log_loss += self.l2_reg * np.linalg.norm(self.weight, ord=2) / 2.

        return log_loss

    def grad(self, x, y, sample_weight=None, l2_reg=False):
        """
        Compute the gradients: grad_wo_reg = (pred - y) * x
        """

        sample_weight = np.array(self.set_sample_weight(x.shape[0], sample_weight))

        pred, _ = self.pred(x)

        indiv_grad = x * (pred - y).reshape(-1, 1)
        reg_grad = self.l2_reg * self.weight
        weighted_indiv_grad = indiv_grad * sample_weight.reshape(-1, 1)
        if self.fit_intercept:
            weighted_indiv_grad = np.concatenate([weighted_indiv_grad, (pred - y).reshape(-1, 1)], axis=1)
            reg_grad = np.concatenate([reg_grad, np.zeros(1)], axis=0)

        total_grad = np.sum(weighted_indiv_grad, axis=0)

        if l2_reg:
            total_grad += reg_grad

        return total_grad, weighted_indiv_grad

    def grad_pred(self, x, sample_weight=None):
        """
        Compute the gradients w.r.t predictions: grad_wo_reg = pred * (1 - pred) * x
        """

        sample_weight = np.array(self.set_sample_weight(x.shape[0], sample_weight))

        pred, _ = self.pred(x)
        indiv_grad = x * (pred * (1 - pred)).reshape(-1, 1)
        weighted_indiv_grad = indiv_grad * sample_weight.reshape(-1, 1)
        total_grad = np.sum(weighted_indiv_grad, axis=0)

        return total_grad, weighted_indiv_grad

    def hess(self, x, sample_weight=None, check_pos_def=False):
        """
        Compute hessian matrix: hessian = pred * (1 - pred) @ x^T @ x + lambda
        """

        sample_weight = np.array(self.set_sample_weight(x.shape[0], sample_weight))

        pred, _ = self.pred(x)

        factor = pred * (1. - pred)
        indiv_hess = np.einsum("a,ai,aj->aij", factor, x, x)
        reg_hess = self.l2_reg * np.eye(x.shape[1])

        if self.fit_intercept:
            off_diag = np.einsum("a,ai->ai", factor, x)
            off_diag = off_diag[:, np.newaxis, :]

            top_row = np.concatenate([indiv_hess, np.transpose(off_diag, (0, 2, 1))], axis=2)
            bottom_row = np.concatenate([off_diag, factor.reshape(-1, 1, 1)], axis=2)
            indiv_hess = np.concatenate([top_row, bottom_row], axis=1)

            reg_hess = np.pad(reg_hess, [[0, 1], [0, 1]], constant_values=0.)

        hess_wo_reg = np.einsum("aij,a->ij", indiv_hess, sample_weight)
        total_hess_w_reg = hess_wo_reg + reg_hess

        if check_pos_def:
            self.check_pos_def(total_hess_w_reg)

        return total_hess_w_reg

    def fit(self, x, y, sample_weight=None, verbose=False):
        sample_weight = self.set_sample_weight(x.shape[0], sample_weight)

        self.model.fit(x, y, sample_weight=sample_weight)
        self.weight: np.ndarray = self.model.coef_.flatten()
        if self.fit_intercept:
            self.bias: np.ndarray = self.model.intercept_

        if verbose:
            pred, _ = self.pred(x)
            train_loss_wo_reg = self.log_loss(x, y, sample_weight)
            reg_loss = np.sum(np.power(self.weight, 2)) * self.l2_reg / 2.
            train_loss_w_reg = train_loss_wo_reg + reg_loss

            print("Train loss: %.5f + %.5f = %.5f" % (train_loss_wo_reg, reg_loss, train_loss_w_reg))

        return

    def pred(self, x):
        return self.model.predict_proba(x)[:, 1], self.model.predict(x)





"""if __name__ == "__main__":
    data = fetch_data("german")

    model = LogisticRegression(l2_reg=data.l2_reg)
    # model = NN(input_dim=data.dim, l2_reg=1e-3)
    # model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-3)

    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    model.fit(data.x_train, data.y_train)
    log_loss = model.log_loss(data.x_train, data.y_train)
    print("Training loss: ", log_loss)

    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)
    val_evaluator(data.y_val, pred_label_val)
    test_evaluator(data.y_test, pred_label_test)"""

    # total_grad, weighted_indiv_grad = model.grad(data.x_train, data.y_train)
    # print(total_grad.shape)
    # print(weighted_indiv_grad.shape)
    # hess = model.hess(data.x_train, data.y_train, check_pos_def=True)
    # total_grad, weighted_indiv_grad = model.grad(data.x_train, data.y_train)
    # print(hess.shape)
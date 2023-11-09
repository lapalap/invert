import torch
from torch.utils.data import Dataset
import json
from scipy import stats
import sklearn.metrics as metrics
import numpy
import torchmetrics
torchmetrics.functional.classification.binary_auroc
from torchmetrics.functional.classification import multilabel_auroc


#TODO: all metrics should return one float (pearson returns 2 variables)
class Metric:
    def __init__(
        self,
        metric="auc-roc",  # change measure
    ):
        self.metric = metric

    def __call__(
        self,
        a,
        l,
    ):
        return torchmetrics.functional.classification.binary_auroc(a, l)
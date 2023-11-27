from torch.utils.data import Dataset
import torchmetrics
from torchmetrics.functional.classification import multilabel_auroc

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
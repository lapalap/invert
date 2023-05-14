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



# #TODO: all metrics should return one float (pearson returns 2 variables)
# class Rho:
#     def __init__(
#         self,
#         measure="fast-auc-roc",  # change measure
#     ):
#         self.measure = measure

#         if self.measure == "fast-auc-roc":
#             self.metric = fast_auc_roc
#         elif self.measure == "auc-pr":
#             self.metric = auc_pr
#         elif self.measure == "auc-roc":
#             self.metric = auc_roc
#         elif self.measure == "pearson":
#             self.metric = pearson
#         elif self.measure == "spearman":
#             self.metric = spearman
#         elif self.measure == "kendalltau":
#             self.metric = kendalltau

#     def __call__(
#         self,
#         x1,
#         x2,
#     ):
#         return self.metric(x1, x2)


# #  using following implementation https://gist.github.com/mattsgithub/dedaa017adc1f30d9833175a5c783221
# def fast_auc_roc(y_true, y_score):

#     """
#     Metric for computing fast version of AUC ROC (sklearn-free)

#     :param y_true:
#     :param y_score:
#     :return:
#     """
#     y_true = y_true.long()
#     # Total number of observations
#     N = y_true.shape[0]

#     # Number of positive observations
#     N_pos = torch.sum(y_true)

#     # Number of negative observations
#     N_neg = N - N_pos

#     # Sort true labels according to scores
#     I = torch.argsort(y_score, descending = True)
#     y_pred = y_true[I]

#     # Index vector
#     I = torch.arange(1, N + 1)

#     return 1. + ((N_pos + 1.) / (2. * N_neg)) - (1. / (N_pos * N_neg)) * I.dot(y_pred)

# def auc_roc(y_true, y_score):

#     """
#     Metric for computing standard version of AUC ROC from sklearn

#     :param y_true:
#     :param y_score:
#     :return:
#     """
#     fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
#     return metrics.auc(fpr, tpr)

# def auc_pr(y_true, y_score):

#     """
#     Metric for computing AUC Precision-Recall from sklearn

#     :param y_true:
#     :param y_score:
#     :return:
#     """
#     precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
#     return metrics.auc(recall, precision)

# def pearson(y_1, y_2):
#     #TODO: write comment
#     return stats.pearsonr(y_1, y_2)

# def spearman(y_1, y_2):
#     # TODO: write comment
#     return stats.spearmanr(y_1, y_2)

# def kendalltau(y_1, y_2):
#     # TODO: write comment
#     return stats.kendalltau(y_1, y_2)



# class LabelDataset(Dataset):
#     def __init__(
#         self,
#         dataset: str,
#     ):
#         """

#         :param dataset:
#         """
#         with open(dataset, "r") as f:
#             self.data = json.load(f)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         image_path, label, mask_path = self.data[idx]
#         label = torch.tensor(label)

#         return label


# def get_predicted_mean_scores(
#     labels, # 1D tensor, int
#     scores, # 1D tensor, float
#     cat_ids: list
# ):
#     """
#     Takes multiple occurences of labels and their according predicted scores of one image as input and returns
#     the mean predicted scores of each label.

#     :param labels: Tensor of classes occuring in one image, 1D.
#     :param scores: Tensor of predicted scores for classes occuring in one image, 1D.
#     :param cat_ids: List of all class ids.
#     :return predicted_scores: Tensor of predicted mean score for each class, [n_classes].
#     """

#     values, indices = torch.sort(labels, 0)
#     unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
#     scores_sorted = torch.gather(scores, 0, indices)

#     # collect each label and its frequency
#     _count = 0
#     group_labels = []
#     for i in labels_count:
#         group_labels.append(scores_sorted[_count:_count + i])
#         _count += i

#     # get the mean score for each label
#     mean_scores = [float(torch.mean(i)) for i in group_labels]
#     mean_scores_idx = dict(zip(unique_labels.tolist(), mean_scores))
#     labels_idx = dict(zip(cat_ids, [0 for i in range(len(cat_ids))]))
#     predicted_scores = {k: (mean_scores_idx[k] if k in unique_labels.tolist() else v)
#                         for (k, v) in labels_idx.items()}
#     predicted_scores = torch.tensor(list(predicted_scores.values()))

#     return predicted_scores


# # TODO: remove if not using AUC thresholds
# def auc_data(x1, x2, threshold):
#     """
#     Collects the highest and lowest scores of given dataset and a threshold.

#     :param x1: tensor of labels, N x K
#     :param x2: tensor of activations, N x K
#     :param threshold: int for threshold.
#     :return A_Phi: sorted and trimmed tensor of labels, threshold x K
#     :return A_F: sorted and trimmed tensor of activations, threshold x K
#     """

#     values, indices = torch.sort(x2, 0)
#     A_F = torch.cat((values[:threshold, :], values[-threshold:, :]), 0)
#     A_F_indices = torch.cat((indices[:threshold, :], indices[-threshold:, :]), 0)
#     A_Phi_sorted = torch.gather(x1, 0, A_F_indices) 
#     A_Phi = torch.cat((A_Phi_sorted[:threshold, :], A_Phi_sorted[-threshold:, :]), 0)

#     return A_Phi, A_F

# class QuantileVector:
#     """
#     Code from https://github.com/CSAILVision/NetDissect-Lite

#     Streaming randomized quantile computation for numpy.
#     Add any amount of data repeatedly via add(data).  At any time,
#     quantile estimates (or old-style percentiles) can be read out using
#     quantiles(q) or percentiles(p).
#     Accuracy scales according to resolution: the default is to
#     set resolution to be accurate to better than 0.1%,
#     while limiting storage to about 50,000 samples.
#     Good for computing quantiles of huge data without using much memory.
#     Works well on arbitrary data with probability near 1.
#     Based on the optimal KLL quantile algorithm by Karnin, Lang, and Liberty
#     from FOCS 2016.  http://ieee-focs.org/FOCS-2016-Papers/3933a071.pdf
#     """

#     def __init__(self, depth=1, resolution=24 * 1024, buffersize=None,
#             dtype=None, seed=None):
#         self.resolution = resolution
#         self.depth = depth
#         # Default buffersize: 128 samples (and smaller than resolution).
#         if buffersize is None:
#             buffersize = min(128, (resolution + 7) // 8)
#         self.buffersize = buffersize
#         self.samplerate = 1.0
#         self.data = [numpy.zeros(shape=(depth, resolution), dtype=dtype)]
#         self.firstfree = [0]
#         self.random = numpy.random.RandomState(seed)
#         self.extremes = numpy.empty(shape=(depth, 2), dtype=dtype)
#         self.extremes.fill(numpy.NaN)
#         self.size = 0

#     def add(self, incoming):
#         assert len(incoming.shape) == 2
#         assert incoming.shape[1] == self.depth
#         self.size += incoming.shape[0]
#         # Convert to a flat numpy array.
#         if self.samplerate >= 1.0:
#             self._add_every(incoming)
#             return
#         # If we are sampling, then subsample a large chunk at a time.
#         self._scan_extremes(incoming)
#         chunksize = numpy.ceil[self.buffersize / self.samplerate]
#         for index in range(0, len(incoming), chunksize):
#             batch = incoming[index:index+chunksize]
#             sample = batch[self.random.binomial(1, self.samplerate, len(batch))]
#             self._add_every(sample)

#     def _add_every(self, incoming):
#         supplied = len(incoming)
#         index = 0
#         while index < supplied:
#             ff = self.firstfree[0]
#             available = self.data[0].shape[1] - ff
#             if available == 0:
#                 if not self._shift():
#                     # If we shifted by subsampling, then subsample.
#                     incoming = incoming[index:]
#                     if self.samplerate >= 0.5:
#                         print('SAMPLING')
#                         self._scan_extremes(incoming)
#                     incoming = incoming[self.random.binomial(1, 0.5,
#                         len(incoming - index))]
#                     index = 0
#                     supplied = len(incoming)
#                 ff = self.firstfree[0]
#                 available = self.data[0].shape[1] - ff
#             copycount = min(available, supplied - index)
#             self.data[0][:,ff:ff + copycount] = numpy.transpose(
#                     incoming[index:index + copycount,:])
#             self.firstfree[0] += copycount
#             index += copycount

#     def _shift(self):
#         index = 0
#         # If remaining space at the current layer is less than half prev
#         # buffer size (rounding up), then we need to shift it up to ensure
#         # enough space for future shifting.
#         while self.data[index].shape[1] - self.firstfree[index] < (
#                 -(-self.data[index-1].shape[1] // 2) if index else 1):
#             if index + 1 >= len(self.data):
#                 return self._expand()
#             data = self.data[index][:,0:self.firstfree[index]]
#             data.sort()
#             if index == 0 and self.samplerate >= 1.0:
#                 self._update_extremes(data[:,0], data[:,-1])
#             offset = self.random.binomial(1, 0.5)
#             position = self.firstfree[index + 1]
#             subset = data[:,offset::2]
#             self.data[index + 1][:,position:position + subset.shape[1]] = subset
#             self.firstfree[index] = 0
#             self.firstfree[index + 1] += subset.shape[1]
#             index += 1
#         return True

#     def _scan_extremes(self, incoming):
#         # When sampling, we need to scan every item still to get extremes
#         self._update_extremes(
#                 numpy.nanmin(incoming, axis=0),
#                 numpy.nanmax(incoming, axis=0))

#     def _update_extremes(self, minr, maxr):
#         self.extremes[:,0] = numpy.nanmin(
#                 [self.extremes[:, 0], minr], axis=0)
#         self.extremes[:,-1] = numpy.nanmax(
#                 [self.extremes[:, -1], maxr], axis=0)

#     def minmax(self):
#         if self.firstfree[0]:
#             self._scan_extremes(self.data[0][:,:self.firstfree[0]].transpose())
#         return self.extremes.copy()

#     def _expand(self):
#         cap = self._next_capacity()
#         if cap > 0:
#             # First, make a new layer of the proper capacity.
#             self.data.insert(0, numpy.empty(
#                 shape=(self.depth, cap), dtype=self.data[-1].dtype))
#             self.firstfree.insert(0, 0)
#         else:
#             # Unless we're so big we are just subsampling.
#             assert self.firstfree[0] == 0
#             self.samplerate *= 0.5
#         for index in range(1, len(self.data)):
#             # Scan for existing data that needs to be moved down a level.
#             amount = self.firstfree[index]
#             if amount == 0:
#                 continue
#             position = self.firstfree[index-1]
#             # Move data down if it would leave enough empty space there
#             # This is the key invariant: enough empty space to fit half
#             # of the previous level's buffer size (rounding up)
#             if self.data[index-1].shape[1] - (amount + position) >= (
#                     -(-self.data[index-2].shape[1] // 2) if (index-1) else 1):
#                 self.data[index-1][:,position:position + amount] = (
#                         self.data[index][:,:amount])
#                 self.firstfree[index-1] += amount
#                 self.firstfree[index] = 0
#             else:
#                 # Scrunch the data if it would not.
#                 data = self.data[index][:,:amount]
#                 data.sort()
#                 if index == 1:
#                     self._update_extremes(data[:,0], data[:,-1])
#                 offset = self.random.binomial(1, 0.5)
#                 scrunched = data[:,offset::2]
#                 self.data[index][:,:scrunched.shape[1]] = scrunched
#                 self.firstfree[index] = scrunched.shape[1]
#         return cap > 0

#     def _next_capacity(self):
#         cap = numpy.ceil(self.resolution * numpy.power(0.67, len(self.data)))
#         if cap < 2:
#             return 0
#         return max(self.buffersize, int(cap))

#     def _weighted_summary(self, sort=True):
#         if self.firstfree[0]:
#             self._scan_extremes(self.data[0][:,:self.firstfree[0]].transpose())
#         size = sum(self.firstfree) + 2
#         weights = numpy.empty(
#             shape=(size), dtype='float32') # floating point
#         summary = numpy.empty(
#             shape=(self.depth, size), dtype=self.data[-1].dtype)
#         weights[0:2] = 0
#         summary[:,0:2] = self.extremes
#         index = 2
#         for level, ff in enumerate(self.firstfree):
#             if ff == 0:
#                 continue
#             summary[:,index:index + ff] = self.data[level][:,:ff]
#             weights[index:index + ff] = numpy.power(2.0, level)
#             index += ff
#         assert index == summary.shape[1]
#         if sort:
#             order = numpy.argsort(summary)
#             summary = summary[numpy.arange(self.depth)[:,None], order]
#             weights = weights[order]
#         return (summary, weights)

#     def quantiles(self, quantiles, old_style=False):
#         if self.size == 0:
#             return numpy.full((self.depth, len(quantiles)), numpy.nan)
#         summary, weights = self._weighted_summary()
#         cumweights = numpy.cumsum(weights, axis=-1) - weights / 2
#         if old_style:
#             # To be convenient with numpy.percentile
#             cumweights -= cumweights[:,0:1]
#             cumweights /= cumweights[:,-1:]
#         else:
#             cumweights /= numpy.sum(weights, axis=-1, keepdims=True)
#         result = numpy.empty(shape=(self.depth, len(quantiles)))
#         for d in range(self.depth):
#             result[d] = numpy.interp(quantiles, cumweights[d], summary[d])
#         return result

#     def integrate(self, fun):
#         result = None
#         for level, ff in enumerate(self.firstfree):
#             if ff == 0:
#                 continue
#             term = numpy.sum(
#                     fun(self.data[level][:,:ff]) * numpy.power(2.0, level),
#                     axis=-1)
#             if result is None:
#                 result = term
#             else:
#                 result += term
#         if result is not None:
#             result /= self.samplerate
#         return result

#     def percentiles(self, percentiles):
#         return self.quantiles(percentiles, old_style=True)

#     def readout(self, count, old_style=True):
#         return self.quantiles(
#                 numpy.linspace(0.0, 1.0, count), old_style=old_style)



# import time
# # An adverarial case: we keep finding more numbers in the middle
# # as the stream goes on.
# amount = 100000
# percentiles = 100
# data = numpy.arange(float(amount))
# data[1::2] = data[-1::-2] + (len(data) - 1)
# data /= 2
# depth = 5
# alldata = data[:,None] + (numpy.arange(depth) * amount)[None, :]
# print(alldata.shape)
# actual_sum = numpy.sum(alldata * alldata, axis=0)
# amt = amount // depth
# for r in range(depth):
#     numpy.random.shuffle(alldata[r*amt:r*amt+amt,r])
# # data[::2] = data[-2::-2]
# # numpy.random.shuffle(data)
# starttime = time.time()
# qc = QuantileVector(depth=depth, resolution=8 * 1024)
# qc.add(alldata)
# ro = qc.readout(101)
# # print(ro)
# endtime = time.time()
# # print 'ro', ro
# # print ro - numpy.linspace(0, amount, percentiles+1)
# gt = numpy.linspace(0, amount, percentiles+1)[None,:] + (
#         numpy.arange(qc.depth) * amount)[:,None]
#
# print(gt.shape)
# print("Maximum relative deviation among %d perentiles:" % percentiles, (
#         numpy.max(abs(ro - gt) / amount) * percentiles))
# print("Minmax eror %f, %f" % (
#     max(abs(qc.minmax()[:,0] - numpy.arange(qc.depth) * amount)),
#     max(abs(qc.minmax()[:, -1] - (numpy.arange(qc.depth)+1) * amount + 1))))
# print("Integral error:", numpy.max(numpy.abs(
#         qc.integrate(lambda x: x * x)
#         - actual_sum) / actual_sum))
# print("Count error: ", (qc.integrate(lambda x: numpy.ones(x.shape[-1])
#         ) - qc.size) / (0.0 + qc.size))
# print("Time", (endtime - starttime))

# from tqdm import tqdm
# import time
# import matplotlib.pyplot as plt
#
# np.random.seed(42)
# N = np.arange(start=10, stop=1000000, step=10000)
#
# t_sklearn = []
# t_dot = []
#
# for n in tqdm(N):
#     N_pos = np.random.randint(low=1, high=n + 1)
#     y_true = torch.tensor(np.concatenate((np.ones(N_pos), np.zeros(n - N_pos))))
#     y_score = torch.tensor(np.random.random(size=n))
#
#     # Timeit
#     t0 = time.time()
#     y1 = auc_roc(y_true=y_true, y_score=y_score)
#     t1 = time.time()
#     t_sklearn.append(t1 - t0)
#
#     # Timeit
#     t0 = time.time()
#     y2 = fast_auc_roc(y_true=y_true, y_score=y_score)
#     t1 = time.time()
#     t_dot.append(t1 - t0)
#
#     # Proves their equality
#     # Raises error if not almost equal (up to 14 decimal places)
#     np.testing.assert_almost_equal(y1, y2, decimal=4)
#
# plt.scatter(N, t_dot, marker='o', color='blue', alpha=0.3, label='Mann-Whitney U implementation')
# plt.scatter(N, t_sklearn, marker='o', color='gray', alpha=0.3, label='scikit-learn implementation')
# plt.title('Comparing Times')
# plt.ylabel('Time (seconds)')
# plt.xlabel('Number of Observations')
# plt.legend()
# plt.show()
import os
import warnings

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from phi import Phi
from metrics import Metric

from operator import itemgetter
import sympy

from tqdm import tqdm

import torchmetrics

warnings.simplefilter("default")
# iou = JaccardIndex(num_classes=2)


class Invert:
    def __init__(
        self,
        F: torch.nn.Module,
        storage_dir=".invert/",
        device="cpu",
    ):
        # TODO: fix description

        self.device = device

        self.F = F

        if storage_dir[-1] == "/":
            storage_dir = storage_dir[:-1]

        self.storage_dir = storage_dir
        self.make_folder_if_it_doesnt_exist(name=storage_dir)

    def load_activations(self, A: torch.Tensor, Labels: torch.Tensor, description: dict):

        # dict {i: 'name', 'description'}
        self.A = A.clone().to(self.device)
        self.Labels = Labels.clone().to(self.device)

        self.concepts = {}
        for i, k in enumerate(description):
            self.concepts[i] = description[k]
            self.concepts[i]['symbol'] = sympy.Symbol(
                self.concepts[i]['offset'])

    def explain_representation(self,
                               r: int,
                               L: int,
                               B: int,
                               metric: Metric,
                               threshold=0.):

        N = self.A.shape[0]

        # creating set of concepts
        univariate_formulas = [Phi(expr=self.concepts[i]['symbol'],
                                   concepts=[self.concepts[k]['symbol']
                                             for k in self.concepts],
                                   concepts_to_indices={
                                       self.concepts[key]['name']: i for i, key in enumerate(self.concepts)},
                                   boolean=True,
                                   device=self.device,
                                   buffer=(self.Labels[:, i] == 1).to(self.device)) for i in range(len(self.concepts))]

        # adding the inverse ones
        univariate_formulas += [~formula for formula in univariate_formulas]

        # start beam search
        formula_length = 1
        # evaluate_univariate formulas and take best beam_search_size

        top_formulas = [{"formula": formula,
                         "length": formula_length,
                         "metric": metric(self.A[:, r], formula.buffer),
                         "concept_fraction": min(formula.buffer.sum(), N - formula.buffer.sum())/N
                         } for formula in univariate_formulas]
        top_formulas = sorted(
            top_formulas, key=itemgetter("metric"), reverse=True)
        top_formulas = [
            formula for formula in top_formulas if formula['concept_fraction'] >= threshold][:B]

        formula_length = 2
        while formula_length <= L:
            for i in tqdm(range(min(B, len(top_formulas)))):
                for j in range(len(univariate_formulas)):
                    conjunction = top_formulas[i]["formula"] & univariate_formulas[j]
                    if conjunction is not None:
                        _metric = metric(self.A[:, r], conjunction.buffer)
                        _sum = conjunction.buffer.sum()
                        _concept_fraction = min(_sum, N - _sum)/N

                        if _concept_fraction >= threshold:
                            top_formulas.append({"formula": conjunction,
                                                 "length": formula_length,
                                                 "metric": _metric,
                                                 "concept_fraction": _concept_fraction})

                    disjunction = top_formulas[i]["formula"] | univariate_formulas[j]
                    if disjunction is not None:
                        _metric = metric(self.A[:, r], disjunction.buffer)
                        _sum = disjunction.buffer.sum()
                        _concept_fraction = min(_sum, N - _sum)/N

                        if _concept_fraction >= threshold:
                            top_formulas.append({"formula": disjunction,
                                                 "length": formula_length,
                                                 "metric": _metric,
                                                 "concept_fraction": _concept_fraction})

            top_formulas = sorted(
                top_formulas, key=itemgetter("metric"), reverse=True)
            top_formulas = top_formulas[:min(B, len(top_formulas))]

            formula_length += 1

        # filter for threshold
        return top_formulas
    

    def explain_representation_fast(self,
                               r: int,
                               L: int,
                               B: int,
                               metric: Metric,
                               threshold=0.):

        N = self.A.shape[0]
        _k =self.Labels.shape[1] #number of concepts

        univariate_formulas = torch.cat((self.Labels == 1, torch.logical_not(self.Labels == 1)), dim = 1)
        scores = torch.zeros([2*_k, 2]).to(self.device) # auc, fraction

        # start beam search
        formula_length = 1
        # evaluate_univariate formulas and take best beam_search_size

        # TODO: make in batches
        scores[:, 0] = torchmetrics.functional.classification.multilabel_auroc(self.A[:, [r]].repeat([1, 2*_k]),
                                                                                univariate_formulas,
                                                                                num_labels=_k*2,
                                                                                average=None,
                                                                                thresholds=None)
        scores[:, 1] = torch.min(univariate_formulas.sum(axis = 0), N - univariate_formulas.sum(axis = 0))/N

        top = torch.argsort(scores[:, 0], descending = True)
        print(top)
        top = top[scores[top, 1] >= threshold][:B]

        buffer = univariate_formulas[:, top].clone().to(self.device)
        scores_buffer = scores[top, :]
        formulas = []

        for i in top.tolist():
            if i // _k == 0:
                formulas.append(self.concepts[i]['symbol'])
            else:
                formulas.append(~self.concepts[i]['symbol'])

        formula_length = 2
        while formula_length <= L:
            for i in tqdm(range(min(B, buffer.shape[1]))):
                conjunction = torch.logical_and(buffer[:, [i]].repeat(1, 2*_k), univariate_formulas)
                disjunction = torch.logical_or(buffer[:, [i]].repeat(1, 2*_k), univariate_formulas)

                _all = torch.cat((conjunction, disjunction), dim = 1)
                _scores = torch.zeros([_all.shape[1], 2]) # auc, fraction
                _scores[:, 0] = torchmetrics.functional.classification.multilabel_auroc(self.A[:, [r]].repeat([1, 4*_k]),
                                                                                         _all,
                                                                                         num_labels=_k*4,
                                                                                         average=None,
                                                                                         thresholds=None)
                _scores[:, 1] = torch.min(_all.sum(axis = 0), N - _all.sum(axis = 0))/N

                _top = torch.argsort(_scores[:, 0], descending = True)
                _top = _top[(_scores[_top, 1] >= threshold ) * ((_all[:, _top].T != buffer[:, [i]].T).any(dim=1))][:B]

                buffer = torch.cat((buffer, _all[:, _top]), dim = 1)
                scores_buffer = torch.cat((scores_buffer, _scores[_top, :]), dim = 0)

                for j in _top.tolist():
                    if j // 2*_k == 0:
                        if j // _k == 0:
                            formulas.append(formulas[i] & self.concepts[i]['symbol'])
                        else:
                            formulas.append(formulas[i] &  ~self.concepts[i]['symbol'])
                    else:
                        if j // _k == 2:
                            formulas.append(formulas[i] | self.concepts[i]['symbol'])
                        else:
                            formulas.append(formulas[i] |  ~self.concepts[i]['symbol'])

            print(buffer.shape)
            buffer, inverse_indices = torch.unique(buffer,  return_inverse=True, dim = 1)
            print(buffer.shape)
            inverse_indices = inverse_indices[:buffer.shape[1]]
            print(inverse_indices)
            scores_buffer = scores_buffer[inverse_indices, :]
            formulas = formulas[inverse_indices.tolist()]

            top = torch.argsort(scores_buffer[:, 0], descending = True)[:B]
            buffer = buffer[:, top]
            scores_buffer = scores_buffer[top, :]
            formulas = formulas[top]

            formula_length += 1

        return formulas, scores_buffer

    # def __check_unique(self, formula, top_formulas):
    #     for i in top_formulas:
    #         if torch.all(formula.buffer.eq(i["formula"].buffer)):
    #             return False
    #     return True

    def __get_filenames_in_a_folder(self, folder: str):
        """
        returns the list of paths to all the files in a given folder
        """

        files = os.listdir(folder)
        files = [f"{folder}/" + x for x in files]
        return files

    def make_folder_if_it_doesnt_exist(self, name):

        if name[-1] == "/":
            name = name[:-1]

        folder_exists = os.path.exists(name)

        if folder_exists == True:
            num_files = len(self.__get_filenames_in_a_folder(folder=name))
            if num_files > 0:
                UserWarning(
                    f"Folder: {name} already exists and has {num_files} items")
        else:
            os.mkdir(name)

    def get_activations(self):
        raise NotImplementedError

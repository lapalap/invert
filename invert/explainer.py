import os
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from invert.phi import Phi
from invert.metrics import Metric

from operator import itemgetter
import sympy

from tqdm import tqdm

from scipy.stats import mannwhitneyu

import torchmetrics
warnings.simplefilter("default")


class Invert:
    def __init__(
        self,
        storage_dir=".invert/",
        device="cpu",
    ):
        self.device = device

        if storage_dir[-1] == "/":
            storage_dir = storage_dir[:-1]

        self.storage_dir = storage_dir
        self.make_folder_if_it_doesnt_exist(name=storage_dir)

    def load_activations(self, A: torch.Tensor, Labels: torch.Tensor, description: dict, dataset: str):

        self.A = A.clone().to(self.device)
        self.Labels = Labels.clone().to(self.device)
        self.dataset = dataset

        if self.dataset == "imagenet":
            self.concepts = {}
            for i, k in enumerate(description):
                self.concepts[i] = description[k]
                self.concepts[i]["symbol"] = sympy.Symbol(
                    self.concepts[i]["offset"])
        elif self.dataset == "coco":
            self.concepts = {}
            for i, k in enumerate(description):
                self.concepts[i] = {"name" : description[k],
                            "symbol" : sympy.Symbol(str(k))
                            }

    def explain_representation(self,
                               r: int,
                               L: int,
                               B: int,
                               metric: Metric,
                               min_fraction=0.,
                               max_fraction=0.5,
                               mode = "positive",
                               memorize_states = False):

        N = self.A.shape[0]

        # creating set of concepts
        univariate_formulas = [Phi(expr=self.concepts[i]["symbol"],
                                   concepts=[self.concepts[k]["symbol"]
                                             for k in self.concepts],
                                   concepts_to_indices={
                                       self.concepts[key]["name"]: i for i, key in enumerate(self.concepts)},
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
                         "concept_fraction": formula.buffer.sum()/N
                         } for formula in univariate_formulas]
        
        if mode == "positive":
            top_formulas = sorted(top_formulas, key=itemgetter("metric"), reverse=True)
        elif mode == "negative":
            top_formulas = sorted(top_formulas, key=itemgetter("metric"), reverse=False)

        top_formulas = [
            formula for formula in top_formulas if (formula["concept_fraction"] <= max_fraction) & (formula["concept_fraction"] >= min_fraction)][:B]
        
        if memorize_states:
            states = {}
            states["1"] = top_formulas.copy()

        formula_length = 2
        while formula_length <= L:
            for i in range(min(B, len(top_formulas))):
                for j in range(len(univariate_formulas)):
                    conjunction = top_formulas[i]["formula"] & univariate_formulas[j]
                    if conjunction is not None:
                        _metric = metric(self.A[:, r], conjunction.buffer)
                        _sum = conjunction.buffer.sum()
                        _concept_fraction = _sum/N

                        if (_concept_fraction >= min_fraction) & (_concept_fraction <= max_fraction):
                            top_formulas.append({"formula": conjunction,
                                                 "length": formula_length,
                                                 "metric": _metric,
                                                 "concept_fraction": _concept_fraction})

                    disjunction = top_formulas[i]["formula"] | univariate_formulas[j]
                    if disjunction is not None:
                        _metric = metric(self.A[:, r], disjunction.buffer)
                        _sum = disjunction.buffer.sum()
                        _concept_fraction = _sum/N

                        if (_concept_fraction >= min_fraction) & (_concept_fraction <= max_fraction):
                            top_formulas.append({"formula": disjunction,
                                                 "length": formula_length,
                                                 "metric": _metric,
                                                 "concept_fraction": _concept_fraction})
            
            if mode == "positive":
                top_formulas = sorted(top_formulas, key=itemgetter("metric"), reverse=True)
            elif mode == "negative":
                top_formulas = sorted(top_formulas, key=itemgetter("metric"), reverse=False)

            top_formulas = top_formulas[:min(B, len(top_formulas))]

            if memorize_states:
                states[str(formula_length)] = top_formulas.copy()

            formula_length += 1

        if memorize_states:
                return states
        return top_formulas

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
    
    def get_p_value(self, r: int, explanation: Phi, alternative='two-sided'):
        U1, p = mannwhitneyu(self.A[explanation.buffer.cpu() == 0, r], A[explanation.buffer.cpu() == 1, r],
                             alternative = alternative)

        return p
        

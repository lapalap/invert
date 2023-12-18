import os
import warnings

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from invert.phi import Phi
from invert.metrics import Metric

from operator import itemgetter
import sympy

import json

from scipy.stats import mannwhitneyu

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
                
    def load_concept_labels(self,
                             labels_path: str,
                             num_samples: int,
                             num_concepts: int,
                             description_path: str,
                             dataset = "imagenet"):
        
        self.labels = torch.load(labels_path).to(self.device)
        # WIP: not working
        # self.labels = torch.BoolTensor(torch.BoolStorage.from_file(labels_path,
        #                                                             shared=True,
        #                                                             size=[num_samples, num_concepts])).reshape(num_samples,
        #                                                                                                     num_concepts)
        self.num_samples = num_samples
        self.num_concepts = num_concepts
        with open(description_path, 'r') as fp:
            self.description  = json.load(fp)

        self.concepts = [sympy.Symbol(self.description[k]["offset"]) for k in self.description]
        
        #big one
        self.memdict = {concept.name :self.labels[:, i] for i,concept in enumerate(self.concepts)}

                
    def explain_representation_test(self,
                               A: torch.Tensor,
                               L: int,
                               B: int,
                               metric: Metric,
                               limit_search = None,
                               min_fraction=0.,
                               max_fraction=0.5,
                               mode = "positive",
                               memorize_states = False):
        # start beam search
        formula_length = 1
        # evaluate_univariate formulas and take best beam_search_size
        ATOMIC_CONCEPTS = []
        for i in range(2*self.num_concepts):
            q = i % self.num_concepts
            if i // self.num_concepts == 0:
                formula = Phi(expr=self.concepts[q],
                              device = self.device)
                buffer = self.memdict[self.concepts[q].name]
            else:
                formula = Phi(expr=~self.concepts[q],
                              device = self.device)
                buffer = ~self.memdict[self.concepts[q].name]

            concept_fraction = buffer.sum()/self.num_samples

            ATOMIC_CONCEPTS.append({"formula": formula,
                         "length": formula_length,
                         "metric": metric(A, buffer),
                         "concept_fraction": concept_fraction})
        
        if mode == "positive":
            ATOMIC_CONCEPTS = sorted(ATOMIC_CONCEPTS, key=itemgetter("metric"), reverse=True)
        elif mode == "negative":
            ATOMIC_CONCEPTS = sorted(ATOMIC_CONCEPTS, key=itemgetter("metric"), reverse=False)

        if limit_search is None:
            limit_search = len(ATOMIC_CONCEPTS)

        BEAM = [
            formula for formula in ATOMIC_CONCEPTS if (formula["concept_fraction"] <= max_fraction) & (formula["concept_fraction"] >= min_fraction)][:B]

        if memorize_states:
            states = {}
            states["1"] = BEAM.copy()

        formula_length = 2

        while formula_length <= L:
            for i in tqdm(range(min(B, len(BEAM)))):
                for j in range(limit_search):
                    conjunction = BEAM[i]["formula"] & ATOMIC_CONCEPTS[j]["formula"]
                    if conjunction is not None:
                        _formula_buffer = conjunction(self.memdict)
                        _metric = metric(A, _formula_buffer)
                        _concept_fraction = _formula_buffer.sum()/self.num_samples

                        if (_concept_fraction >= min_fraction) & (_concept_fraction <= max_fraction):
                            BEAM.append({"formula": conjunction,
                                                 "length": formula_length,
                                                 "metric": _metric,
                                                 "concept_fraction": _concept_fraction})

                    disjunction = BEAM[i]["formula"] | ATOMIC_CONCEPTS[j]["formula"]
                    if disjunction is not None:
                        _formula_buffer = disjunction(self.memdict)
                        _metric = metric(A, _formula_buffer)
                        _concept_fraction = _formula_buffer.sum()/self.num_samples

                        if (_concept_fraction >= min_fraction) & (_concept_fraction <= max_fraction):
                            BEAM.append({"formula": disjunction,
                                                 "length": formula_length,
                                                 "metric": _metric,
                                                 "concept_fraction": _concept_fraction})
            
            if mode == "positive":
                BEAM = sorted(BEAM, key=itemgetter("metric"), reverse=True)
            elif mode == "negative":
                BEAM = sorted(BEAM, key=itemgetter("metric"), reverse=False)

            BEAM = BEAM[:min(B, len(BEAM))]

            if memorize_states:
                states[str(formula_length)] = BEAM.copy()

            formula_length += 1

        if memorize_states:
                return states
        return BEAM

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
        U1, p = mannwhitneyu(self.A[explanation.buffer.cpu() == 0, r].cpu(), self.A[explanation.buffer.cpu() == 1, r].cpu(),
                             alternative = alternative)

        return p
        

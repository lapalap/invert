import os
import warnings
import torchmetrics
import torch
import sympy
import json

from tqdm import tqdm
from invert.phi import Phi
from operator import itemgetter
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
                             description_path: str):
        
        self.labels = torch.load(labels_path)
        self.num_samples = num_samples
        self.num_concepts = num_concepts
        with open(description_path, 'r') as fp:
            self.description  = json.load(fp)

        self.concepts = [sympy.Symbol(self.description[k]["offset"]) for k in self.description]
        
        #big one
        self.memdict = {concept.name :self.labels[:, i] for i,concept in enumerate(self.concepts)}

    def _metric(self, a: torch.Tensor, b: torch.Tensor):
        return torchmetrics.functional.classification.binary_auroc(a, b)

    @torch.no_grad()            
    def explain_representation_test(self,
                               A: torch.Tensor,
                               L: int,
                               B: int,
                               limit_search = None,
                               min_fraction=0.,
                               max_fraction=0.5,
                               mode = "positive",
                               memorize_states = False):
        # start beam search
        formula_length = 1
        # evaluate_univariate formulas and take best beam_search_size
        ATOMIC_CONCEPTS = []
        for i in tqdm(range(2*self.num_concepts)):
            q = i % self.num_concepts
            if i // self.num_concepts == 0:
                _formula = Phi(expr=self.concepts[q],
                              device = self.device)
                _buffer = self.memdict[self.concepts[q].name].to(self.device)
                _metric = self._metric(A, _buffer)
                _concept_fraction = _buffer.sum()/self.num_samples
            else:
                _formula = Phi(expr=~self.concepts[q],
                              device = self.device)
                _buffer = ~self.memdict[self.concepts[q].name]
                _metric = 1 - ATOMIC_CONCEPTS[q]["metric"]
                _concept_fraction = 1 - ATOMIC_CONCEPTS[q]["concept_fraction"]

            ATOMIC_CONCEPTS.append({"formula": _formula,
                         "length": formula_length,
                         "metric": _metric,
                         "differentiability": 2*abs(0.5 - _metric),
                         "concept_fraction": _concept_fraction})
        del _buffer
        
        if mode == "positive":
            ATOMIC_CONCEPTS = sorted(ATOMIC_CONCEPTS, key=itemgetter("metric"), reverse=True)
        elif mode == "negative":
            ATOMIC_CONCEPTS = sorted(ATOMIC_CONCEPTS, key=itemgetter("metric"), reverse=False)

        BEAM = [
            formula for formula in ATOMIC_CONCEPTS if (formula["concept_fraction"] <= max_fraction) & (formula["concept_fraction"] >= min_fraction)][:B]
            
        if memorize_states:
            states = {}
            states["1"] = BEAM.copy()
            
        ATOMIC_CONCEPTS = sorted(ATOMIC_CONCEPTS, key=itemgetter("differentiability"), reverse=True)
        if limit_search is None:
            limit_search = len(ATOMIC_CONCEPTS)
        ATOMIC_CONCEPTS = ATOMIC_CONCEPTS[:limit_search]

        formula_length = 2

        while formula_length <= L:
            for i in tqdm(range(min(B, len(BEAM)))):
                for j in range(limit_search):
                    conjunction = BEAM[i]["formula"] & ATOMIC_CONCEPTS[j]["formula"]
                    if conjunction is not None:
                        _formula_buffer = conjunction(self.memdict)
                        _metric = self._metric(A, _formula_buffer)
                        _concept_fraction = _formula_buffer.sum()/self.num_samples

                        if (_concept_fraction >= min_fraction) & (_concept_fraction <= max_fraction):
                            BEAM.append({"formula": conjunction,
                                                 "length": formula_length,
                                                 "metric": _metric,
                                                 "differentiability": 2*abs(0.5 - _metric),
                                                 "concept_fraction": _concept_fraction})

                    disjunction = BEAM[i]["formula"] | ATOMIC_CONCEPTS[j]["formula"]
                    if disjunction is not None:
                        _formula_buffer = disjunction(self.memdict)
                        _metric = self._metric(A, _formula_buffer)
                        _concept_fraction = _formula_buffer.sum()/self.num_samples

                        if (_concept_fraction >= min_fraction) & (_concept_fraction <= max_fraction):
                            BEAM.append({"formula": disjunction,
                                                 "length": formula_length,
                                                 "metric": _metric,
                                                 "differentiability": 2*abs(0.5 - _metric),
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
    
    def get_p_value(self, A: torch.Tensor, explanation: Phi, alternative='two-sided'):
        _buffer = explanation(self.memdict)
        
        U1, p = mannwhitneyu(A[~_buffer].cpu(), self.A[_buffer].cpu(),
                             alternative = alternative)

        return p
        

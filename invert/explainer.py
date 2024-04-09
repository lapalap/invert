import os
import warnings
import torchmetrics
import torch
import sympy
import json
from operator import itemgetter
from scipy.stats import mannwhitneyu
from typing import List, Dict, TypedDict

from invert.utils import find_by_offset
from invert.phi import Phi


class Explanation(TypedDict):
    formula: Phi
    length: int
    metric: float
    differentiability: float
    concept_fraction: float


warnings.simplefilter("default")


class Invert:
    def __init__(
        self,
        device="cpu",
    ) -> None:
        """
        Initialize Invert class

        Args:
            device (str, optional): Device to perform computations on. Defaults to "cpu".
        """
        self.device = device

    def load_concept_labels(self,
                            labels_path: str,
                            description_path: str) -> None:
        """
        Loading the dataset labels as one-hot-encoded matrix and label description.

        Args:
            labels_path (str): Path to the labels matrix in a form of one-hot encoded tensor.
            Tensor should have n rows, with k columns, where n number of samples and k is number of concepts.
            description_path (str): Path to the json file with label descriptions
        """

        self.labels = torch.load(labels_path)
        self.num_samples = self.labels.shape[0]
        self.num_concepts = self.labels.shape[1]

        with open(description_path, 'r') as fp:
            self.description = json.load(fp)

        self.concepts = [sympy.Symbol(self.description[k]["offset"])
                         for k in self.description]

        # big one
        self.memdict = {concept.name: self.labels[:, i]
                        for i, concept in enumerate(self.concepts)}

    def _metric(self, a: torch.Tensor, b: torch.Tensor):
        return torchmetrics.functional.classification.binary_auroc(a, b)

    @torch.no_grad()
    def explain_representation(self,
                               A: torch.Tensor,
                               L: int,
                               B: int,
                               limit_search=None,
                               min_fraction=0.,
                               max_fraction=0.5,
                               mode="positive",
                               memorize_states=False) -> List[Explanation] | Dict[str, List[Explanation]]:
        """
        Returns the top B explanations for the provided activations.

        Args:
            A (torch.Tensor): Neural activations in the shape of n, where n is the number of samples.
            L (int): Maximum formula length.
            B (int): Beam size.
            limit_search (type, optional): Parameter that limits the number of intermediate formulas to improve computation speed. Defaults to None.
            min_fraction (type, optional): Minimum fraction of the dataset where the explanation should be positive. Defaults to 0.
            max_fraction (float, optional): Maximum fraction of the dataset where the explanation should be positive. Defaults to 0.5.
            mode (str, optional): Determines whether to search for explanations maximizing or minimizing the AUC metric. Defaults to "positive."
            memorize_states (bool, optional): If True, memorizes all intermediate states. Defaults to False.

        Returns:
            List[Dict[str, Union[Phi, int, float]]]: A list of B (beam size) best explanations, sorted in descending order by AUC. Each explanation is represented as a dictionary with the following keys:
                - "formula": Phi class of the explanation.
                - "length": Length of the Phi formula.
                - "metric": AUC score.
                - "differentiability": 2 * abs(0.5 - AUC).
                - "concept_fraction": Size of the positive image of a concept divided by the size of the dataset.

        If memorize_states is True, the function returns a dictionary where keys correspond to different formula lengths, and each value is a list of B best explanations.
        """
        # put A to self.device
        A = A.to(self.device)

        # start beam search
        formula_length = 1
        # evaluate_univariate formulas and take best beam_search_size
        ATOMIC_CONCEPTS = []
        for i in range(2*self.num_concepts):
            q = i % self.num_concepts
            if i // self.num_concepts == 0:
                _formula = Phi(expr=self.concepts[q],
                               device=self.device)
                _buffer = self.memdict[self.concepts[q].name].to(self.device)
                _metric = self._metric(A, _buffer)
                _concept_fraction = _buffer.sum()/self.num_samples
            else:
                _formula = Phi(expr=~self.concepts[q],
                               device=self.device)
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
            ATOMIC_CONCEPTS = sorted(
                ATOMIC_CONCEPTS, key=itemgetter("metric"), reverse=True)
        elif mode == "negative":
            ATOMIC_CONCEPTS = sorted(
                ATOMIC_CONCEPTS, key=itemgetter("metric"), reverse=False)

        BEAM = [
            formula for formula in ATOMIC_CONCEPTS if ((formula["concept_fraction"] <= max_fraction) &
            (formula["concept_fraction"] >= min_fraction) &
            (formula["concept_fraction"] > 0.) &
            (formula["concept_fraction"] < 1.))][:B]

        if memorize_states:
            states = {}
            states["1"] = BEAM.copy()

        ATOMIC_CONCEPTS = sorted(ATOMIC_CONCEPTS, key=itemgetter(
            "differentiability"), reverse=True)
        if limit_search is None:
            limit_search = len(ATOMIC_CONCEPTS)
        ATOMIC_CONCEPTS = ATOMIC_CONCEPTS[:limit_search]

        formula_length = 2

        while formula_length <= L:
            for i in range(min(B, len(BEAM))):
                for j in range(limit_search):
                    conjunction = BEAM[i]["formula"] & ATOMIC_CONCEPTS[j]["formula"]
                    if conjunction is not None:
                        _formula_buffer = conjunction(self.memdict)
                        _metric = self._metric(A, _formula_buffer)
                        _concept_fraction = _formula_buffer.sum()/self.num_samples

                        if ((_concept_fraction >= min_fraction) &
                        (_concept_fraction <= max_fraction) &
                        (_concept_fraction < 1.) &
                        (_concept_fraction > 0.)):
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

                        if ((_concept_fraction >= min_fraction) &
                        (_concept_fraction <= max_fraction) &
                        (_concept_fraction < 1.) &
                        (_concept_fraction > 0.)):
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

    def get_p_value(self, A: torch.Tensor, explanation: Phi, alternative='two-sided') -> float:
        """
        Returns a p-value of Mann-Whitney U test.

        Args:
            A (torch.Tensor): Neural activations
            explanation (Phi): Explanation
            alternative (str, optional): Alternative hypothesis in Mann-Whitney U test. Potential choices: /'one-sided/' or /'two-sided/'.  Defaults to 'two-sided'

        Returns:
            float: p-value of the  Mann-Whitney U test
        """
        A = A.to(self.device)
        _buffer = explanation(self.memdict)

        U1, p = mannwhitneyu(A[~_buffer].cpu(), A[_buffer].cpu(),
                             alternative=alternative)

        return p

    def explain_formula(self, explanation: Phi):

        human_readable_explanation = str(explanation)
        output = {}
        elements = []

        for element in explanation._distinct_concepts:
            label = find_by_offset(str(element), self.description)
            elements.append(
                {'wordnet': element, 'description': label['name'], 'full_description': label['definition']})
            human_readable_explanation = human_readable_explanation.replace(
                str(element), label['name'])

        output = {'label': human_readable_explanation,
                  'details': elements}

        return output
        
        

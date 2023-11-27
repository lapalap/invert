import sympy
import torch

class Phi:
    @torch.no_grad()
    def __init__(
            self,
            expr: sympy.core.expr.Expr,
            concepts: list,
            concepts_to_indices: dict,
            boolean: bool,
            device,
            buffer: torch.Tensor = None
    ):

        """
        Base class for the explanation

        :param expr:
        :param concepts:
        :param concepts_to_indices:
        :param boolean:
        :param device:
        """
        # we support only logic for now
        assert boolean

        self.expr = sympy.logic.boolalg.simplify_logic(
            expr, form="cnf"
        )  # expression itself
        self.concepts = concepts
        self.concepts_to_indices = concepts_to_indices
        self.boolean = boolean  # if the formula is binary or not
        self.device = device
        self.buffer = buffer

        self.compute_graph = False

        # update information
        self.__update_info()

    def __update_info(self):
        self.info = {}

        # create a pytorch model
        self.info["str"] = str(self.expr).replace(" ", "")
        self.info["disjunction_terms"] = self.info["str"].split("&")
        self.info["n_disjunction_terms"] = len(self.info["disjunction_terms"])
        self.info["distinct_concepts"] = self.expr.free_symbols
        self.info["n_distinct_concepts"] = len(self.info["distinct_concepts"])
        self.info["n_concepts"] = len(self.concepts)

        if self.compute_graph and self.info["n_distinct_concepts"] > 1:
            # create a pytorch model only in case of multivariate formula
            self.disjunction_layer = torch.nn.Linear(
                self.info["n_concepts"], self.info["n_disjunction_terms"], bias=False
            ).to(self.device)
            self.disjunction_layer.weight = torch.nn.Parameter(
                torch.zeros_like(self.disjunction_layer.weight)
            )
            self.negation_indices = torch.zeros(self.info["n_concepts"]).to(self.device)

            for i, disjunction_term in enumerate(self.info["disjunction_terms"]):
                disjunction_term = disjunction_term.replace("(", "").replace(")", "")
                terms = disjunction_term.split("|")
                for term in terms:
                    negation = False
                    if term[0] == "~":
                        term = term[1:]
                        negation = True
                    term_index = self.concepts_to_indices[term]
                    if negation:
                        self.negation_indices[term_index] = 1.0
                    self.disjunction_layer.weight[i, term_index] = 1.0

    def __and__(self, phi):
        assert self.concepts == phi.concepts
        assert self.boolean is not None

        result = torch.logical_and(self.buffer, phi.buffer)

        # check if intersection is empty or everything
        if (result.sum() == 0) or (result.sum() == result.shape[0]):
            return None

        # check if new formula doesn't change anything
        if torch.equal(self.buffer, result):
            return None

        return Phi(expr=self.expr & phi.expr,
                   concepts = self.concepts,
                   concepts_to_indices = self.concepts_to_indices,
                   boolean = True,
                   device = self.device,
                   buffer = result)

    def __or__(self, phi):
        assert self.concepts == phi.concepts
        assert self.boolean is not None

        result = torch.logical_or(self.buffer, phi.buffer)

        # check if intersection is empty or everything
        if (result.sum() == 0) or (result.sum() == result.shape[0]):
            return None

        # check if new formula doesn't change anything
        if torch.equal(self.buffer, result):
            return None

        return Phi(expr= self.expr | phi.expr,
                   concepts=self.concepts,
                   concepts_to_indices=self.concepts_to_indices,
                   boolean=True,
                   device=self.device,
                   buffer=result)

    def __invert__(self):
        assert self.boolean is not None
        return Phi(expr= ~self.expr,
                   concepts=self.concepts,
                   concepts_to_indices=self.concepts_to_indices,
                   boolean=True,
                   device=self.device,
                   buffer=torch.logical_not(self.buffer))

    def __call__(self, X: torch.Tensor):
        if self.info["n_distinct_concepts"] > 1:
            X = X + self.negation_indices  # performs negation operation
            X = self.disjunction_layer(X)  # performs disjunction
            X = torch.clip(X, min=0, max=1)  # clips disjunction output between 0 and 1
            X = torch.prod(X, 1)  # performs conjunction
            return X
        else:
            distinct_concept = list(self.info["distinct_concepts"])[0]
            index = self.concepts_to_indices[str(distinct_concept)]
            return X[..., index]

    def __repr__(self):
        """
        Used for printing the explanation

        :return:
        """
        return self.expr.__str__()
    
    def describe(self, concept_description: dict):
         self.info["str"]


# #### PHI EXAMPLE START #####
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# N_S = 80
# text_symbols = ["c{i}".format(i=i) for i in range(80)]
# S = sympy.symbols(text_symbols)
#
# concepts_to_indices = {text_symbols[i]: i for i in range(80)}
# formula = ~S[0]
#
# phi = Phi(
#     expr=formula,
#     concepts=S,
#     concepts_to_indices=concepts_to_indices,
#     boolean=True,
#     device=device,
# )
# print(phi.info)

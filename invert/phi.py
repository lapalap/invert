from typing import Any
import sympy
import torch
import sympytorch
import functools as ft

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

from sympytorch.sympy_module import _global_func_lookup

ExprType = TypeVar("ExprType", bound=sympy.Expr)
T = TypeVar("T")

def _reduce(fn: Callable[..., T]) -> Callable[..., T]:
    def fn_(*args: Any) -> T:
        return ft.reduce(fn, args)

    return fn_

_global_func_lookup[sympy.And] = _reduce(torch.mul)
_global_func_lookup[sympy.Or] = _reduce(torch.add)


_updated_func_lookup: Dict[
    Union[Type[sympy.Basic], Callable[..., Any]], Callable[..., torch.Tensor]
] = {
    sympy.And: _reduce(torch.mul),
    sympy.Or: _reduce(torch.add),
}

class Phi:
    @torch.no_grad()
    def __init__(
            self,
            expr: sympy.core.expr.Expr,
            device
    ):

        """
        Base class for the explanation

        :param expr:
        :param concepts:
        """
        self.expr = expr
        self._distinct_concepts = list(self.expr.free_symbols)
        self.device = device
        self._pytorch_expr = sympytorch.SymPyModule(expressions=[self.expr],
                                            extra_funcs=_updated_func_lookup).to(self.device)

    @torch.no_grad()
    def __call__(self, memdict: dict) -> torch.tensor:
        # for binary labels
        subset = {c.name: memdict.get(c.name, None).to(self.device) for c in self._distinct_concepts}
        return self._pytorch_expr(**subset)[:, 0]

    def __and__(self, phi):
        return Phi(expr = self.expr & phi.expr,
                   device = self.device)

    def __or__(self, phi):
        return Phi(expr = self.expr | phi.expr,
                   device = self.device)

    def __invert__(self):
        return Phi(expr =~self.expr,
                   device = self.device)

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

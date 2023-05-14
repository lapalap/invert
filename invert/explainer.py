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

        #dict {i: 'name', 'description'}
        self.A = A.clone()
        self.Labels = Labels.clone()
        
        self.concepts = {}
        for i, k in enumerate(description):
            self.concepts[i] = description[k]
            self.concepts[i]['symbol'] = sympy.Symbol(self.concepts[i]['offset'])

    def explain_representation(self,
                r: int,
                L:int,
                B:int,
                metric: Metric,
                threshold = 0.):
        
        N = self.A.shape[0]

        #creating set of concepts
        univariate_formulas = [Phi(expr=self.concepts[i]['symbol'],
                                   concepts = [self.concepts[k]['symbol'] for k in self.concepts],
                                   concepts_to_indices = {self.concepts[key]['name']: i for i, key in enumerate(self.concepts)},
                                   boolean=True,
                                   device=self.device,
                                   buffer = (self.Labels[:, i] == 1).to(self.device)) for i in range(len(self.concepts))]
        
        # adding the inverse ones
        univariate_formulas += [~formula for formula in univariate_formulas]

        #start beam search
        formula_length = 1
        # evaluate_univariate formulas and take best beam_search_size

        top_formulas = [{"formula": formula,
                    "length": formula_length,
                    "metric": metric(self.A[:, r], formula.buffer),
                    "concept_fraction": min(formula.buffer.sum(), N - formula.buffer.sum())/N
                    } for formula in univariate_formulas]
        top_formulas = sorted(top_formulas, key=itemgetter("metric"), reverse=True)
        top_formulas = [formula for formula in top_formulas if formula['concept_fraction'] > threshold][:B]

        formula_length = 2
        while formula_length <= L:
            for i in tqdm(range(B)):
                for j in range(len(univariate_formulas)):
                    conjunction = top_formulas[i]["formula"] & univariate_formulas[j]
                    if conjunction is not None:
                        _metric = metric(self.A[:, r], conjunction.buffer)
                        _sum = conjunction.buffer.sum()
                        _concept_fraction = min(_sum, N - _sum)/N

                        if _concept_fraction > threshold:
                            top_formulas.append({"formula": conjunction,
                                                 "length": formula_length,
                                                 "metric": _metric,
                                                 "concept_fraction": _concept_fraction})

                    disjunction = top_formulas[i]["formula"] | univariate_formulas[j]
                    if disjunction is not None:
                        _metric = metric(self.A[:, r], disjunction.buffer)
                        _sum = disjunction.buffer.sum()
                        _concept_fraction = min(_sum, N - _sum)/N

                        if _concept_fraction > threshold:
                            top_formulas.append({"formula": disjunction,
                                                 "length": formula_length,
                                                 "metric": _metric,
                                                 "concept_fraction": _concept_fraction})

            top_formulas = sorted(top_formulas, key=itemgetter("metric"), reverse=True)
            top_formulas = top_formulas[:B]
            print(top_formulas)

            formula_length += 1

        #filter for threshold
        return top_formulas
    
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
                UserWarning(f"Folder: {name} already exists and has {num_files} items")
        else:
            os.mkdir(name)

    def get_activations(self):
        raise NotImplementedError

    # def beam_search(self,
    #                   A,
    #                   concepts,
    #                   concepts_to_indices: dict,
    #                   labels: torch.Tensor,
    #                   measure: Rho,
    #                   length_limit=3,
    #                   beam_search_size = 20,
    #                   beam_search_limit = None,
    #                   threshold = 0.
    #                   ):
    #     """

    #     :param length_limit:
    #     :param beam_search_size:
    #     :param beam_search_limit:
    #     :return:
    #     """

    #     N = A.shape[0]

    #     #creating set of concepts
    #     univariate_formulas = [Phi(expr=concepts[i],
    #                                concepts = concepts,
    #                                concepts_to_indices = concepts_to_indices,
    #                                boolean=True,
    #                                device=self.device,
    #                                buffer = labels[..., i]) for i in range(len(concepts))]
    #     # adding the inverse ones
    #     univariate_formulas += [~formula for formula in univariate_formulas]

    #     #start beam search
    #     formula_length = 1
    #     # evaluate_univariate formulas and take best beam_search_size

    #     scores = [{"formula": formula,
    #                "auc": measure(formula.buffer, A),
    #                "dt_fraction": min(formula.buffer.sum(), N - formula.buffer.sum())/N
    #                }
    #               for formula in univariate_formulas]
    #     # scores = [[formula,
    #     #            measure(formula.buffer, A),
    #     #            min(formula.buffer.sum(), N - formula.buffer.sum())/N] for formula in univariate_formulas]
    #     scores = sorted(scores, key=itemgetter("auc"), reverse=True)

    #     top_formulas = [formula for formula in scores if formula['dt_fraction'] > threshold][:beam_search_size]

    #     while formula_length < length_limit:
    #         for i in range(beam_search_size):
    #             for j in range(len(univariate_formulas)):
    #                 conjunction = top_formulas[i]["formula"] & univariate_formulas[j]
    #                 if conjunction is not None and top_formulas[i]["formula"].expr !=univariate_formulas[j].expr:
    #                     _auc = measure(conjunction.buffer, A)
    #                     _sum = conjunction.buffer.sum()
    #                     _dt_fraction = min(_sum, N - _sum)/N

    #                     if _dt_fraction > threshold:
    #                         top_formulas.append({"formula": conjunction,
    #                                              "auc": _auc,
    #                                              "dt_fraction": _dt_fraction})

    #                     # top_formulas.append([conjunction,
    #                     #                      measure(conjunction.buffer, A),
    #                     #                      min(conjunction.buffer.sum(), N - conjunction.buffer.sum())/N])

    #                 disjunction = top_formulas[i]["formula"] | univariate_formulas[j]
    #                 if disjunction is not None and top_formulas[i]["formula"].expr != univariate_formulas[j].expr:

    #                     _auc = measure(disjunction.buffer, A)
    #                     _sum = disjunction.buffer.sum()
    #                     _dt_fraction = min(_sum, N - _sum)/N

    #                     if _dt_fraction > threshold:
    #                         top_formulas.append({"formula": disjunction,
    #                                              "auc": _auc,
    #                                              "dt_fraction": _dt_fraction})
    #                     # top_formulas.append([disjunction,
    #                     #                      measure(disjunction.buffer, A),
    #                     #                      min(disjunction.buffer.sum(), N - disjunction.buffer.sum())/N])

    #         top_formulas = sorted(top_formulas, key=itemgetter("auc"), reverse=True)
    #         top_formulas = top_formulas[:beam_search_size]
    #         print(top_formulas)

    #         formula_length += 1

    #     #filter for threshold
    #     return top_formulas, scores



        #         formula_length = max(formulas.keys()) + 1
        #
        #         if formula_length > length_limit:
        #             return formulas
        #
        #         if beam_search_limit is None:
        #             beam_search_limit = len(formulas[1])
        #
        #         formulas[formula_length] = []
        #
        #         for i, formula in formulas[formula_length-1][:beam_search_size]:
        #             for j, unit_formula in formulas[formula_length-1][:beam_search_limit]:
        #                 mask_and = formula['mask'] * unit_formula['mask']
        #                 formulas[formula_length].append({'formula': formula['formula'] & unit_formula['formula'],
        #                                     'mask': mask_and,
        #                                     'iou': iou(E_f_b, mask_and)})
        #
        #                 mask_or = torch.clamp(formula['mask'] + unit_formula['mask'], max=1)
        #                 formulas[formula_length].append({'formula': formula['formula'] | unit_formula['formula'],
        #                                     'mask': mask_or,
        #                                     'iou': iou(E_f_b, mask_or)})
        #
        #         sorted(formulas[formula_length], key=itemgetter('iou'))
        #         if not store_all:
        #             formulas[formula_length] = formulas[formula_length][:beam_search_size]
        #
        #         return self.__beam_step(E_f_b, formulas, length_limit, beam_search_size, beam_search_limit, store_all)



    # def connect(self,
    #             data,
    #             metamodel: nn.Module,
    #             dataloader_batch_size = 256):
    #
    #     #STEP 1
    #     # Setting up a dataloader
    #     dataloader = DataLoader(data, batch_size=dataloader_batch_size, shuffle=False)
    #
    #
    #     A_F = torch.load('A_F.tnsr')
    #     A_G = torch.load('A_G.tnsr')
    #     optimizer = torch.optim.AdamW(metamodel.parameters(), lr=0.001)
    #     loss_form = torch.nn.MSELoss()
    #     metamodel.to(self.device).train()
    #     for epoch in tqdm(range(2001)):
    #         optimizer.zero_grad()
    #         y_pred = metamodel(A_G)
    #         loss = loss_form(y_pred, A_F)
    #         loss.backward()
    #         optimizer.step()
    #
    #     return metamodel

# #### CONNECTOR START #####
# model = torchvision.models.resnet18(pretrained = True)
#
# connector = Connector(model, K=1000)
#
# transform=torchvision.transforms.Compose([
#                            torchvision.transforms.ToTensor(),
#                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                        ])
#
# A = torch.load('/Users/kirillbykov/Documents/Work/DORA/notebooks/feature_extractor/features/kaggle/resnet18/resnet18_vals_full/A50k_fcresnet18_val.tnsr')
# B = torch.load('/Users/kirillbykov/Documents/Work/DORA/notebooks/feature_extractor/theory/labels_ILSVRC2012_val.tnsr')
# descriptions = torch.load('/Users/kirillbykov/Documents/Work/DORA/notebooks/feature_extractor/theory/columns_ILSVRC2012_val.tnsr')
#
# print(connector.beam_search(
#                       A = A[..., 254],
#                       concepts = sympy.symbols(["c{i}".format(i=i) for i in range(len(descriptions))]),
#                       concepts_to_indices = {"c{i}".format(i=i): i for i in range(len(descriptions))},
#                       labels =  B,
#                       measure =  Rho(),
#                       length_limit=3,
#                       beam_search_size = 3,
#                       beam_search_limit = None,
#                       ))
#
#
# # dataset = COCODataset(
# #     split="val2017",
# #     #split="train2017",
# #     download=False,
# #     path= "/Users/kirillbykov/Documents/GitHub/representation-connector/connector/coco",
# #     # path="/home/laura/ida-cluster/compexp/vision/cocoapi/coco",  # CPU TODO: remove
# #     #path="/home/lkopf/compexp/vision/cocoapi/coco",  # GPU TODO: remove
# # )
#
# k = 1000




#
# metamodel = torch.nn.Sequential(
#     torch.nn.Linear(len(dataset.concepts), k))
#
# metamodel = connector.connect(dataset, metamodel)
# torch.save(metamodel, 'metamodel.mdl')



#### CONNECTOR END #####


        # # Initialize the linear model
        # lr = LinearRegression()
        # # training the model
        # lr.fit(A_G.T, A_f.T)
        #
        # coefficients = lr.coef_

#
#     def __get_filenames_in_a_folder(self, folder: str):
#         """
#         returns the list of paths to all the files in a given folder
#         """
#
#         files = os.listdir(folder)
#         files = [f"{folder}/" + x for x in files]
#         return files
#
#     def __transform_explanations(self,
#                                 explanations: torch.Tensor):
#         """
#         transforms the explanation to absolute values and sums up color channels
#         """
#         return explanations.abs().sum(axis = 1)
#
#     def __get_focal_field_multipliers(self,
#                                      scores: torch.Tensor,
#                                      type = 'one-tail'):
#         """
#         get focal field multiplier from the tensor of representation scores
#         """
#
#         if type == 'one-tail':
#             positions = torch.argsort(scores, descending=True)
#             return positions/positions.max()
#         elif  type == 'two-tail':
#             scores_transformed = (scores - scores.mean()).abs()
#             positions = torch.argsort(scores_transformed, descending=True)
#             return positions / positions.max()
#
#     def __normalise_attribution_batch(self,
#                                      explanations: torch.Tensor):
#         """
#         normalise attribution maps between 0 and 1
#         """
#         s = explanations.max(axis=0).view([explanations.shape[0], 1, 1])
#         return explanations / s
#
#
#     def __smooth_explanations(self,
#                              explanations: torch.Tensor):
#         """
#         smooth explanations to make nice binary masks
#         """
#         explanations = self.blurrer(explanations)
#         return explanations
#
#     def __binarise_explanations(self,
#                                 explanations: torch.Tensor,
#                                 scores: torch.Tensor,
#                                 threshold = 0.05):
#         """
#         binarise explanation maps
#         """
#
#         # STEP 1: Normalise explanations
#         explanations = self.__normalise_attribution_batch(explanations)
#
#         # STEP 2: Smooth explanations
#         explanations = self.__smooth_explanations(explanations)
#
#         # STEP 3: Limit focal field
#         focal_multipliers = self.__get_focal_field_multiplier(scores)
#
#         low_percentile = 1.- focal_multipliers
#         # TODO: check if this is optimal
#         quantiles = torch.quantile(explanations.view([explanations.shape[0], -1]), low_percentile, dim = 1)
#         quantiles = torch.diagonal(quantiles, 0)
#
#         explanations = explanations * (explanations > quantiles.view([explanations.shape[0], 1, 1]))
#
#         # STEP 4: thresholding and binarisation
#         explanations = 1. * (explanations >= threshold)
#
#         return explanations
#
#     def __beam_step(self,
#                      E_f_b: torch.Tensor,
#                      formulas: dict,
#                      length_limit = 3,
#                      beam_search_size = 20,
#                      beam_search_limit = None,
#                      store_all = False):
#         formula_length = max(formulas.keys()) + 1
#
#         if formula_length > length_limit:
#             return formulas
#
#         if beam_search_limit is None:
#             beam_search_limit = len(formulas[1])
#
#         formulas[formula_length] = []
#
#         for i, formula in formulas[formula_length-1][:beam_search_size]:
#             for j, unit_formula in formulas[formula_length-1][:beam_search_limit]:
#                 mask_and = formula['mask'] * unit_formula['mask']
#                 formulas[formula_length].append({'formula': formula['formula'] & unit_formula['formula'],
#                                     'mask': mask_and,
#                                     'iou': iou(E_f_b, mask_and)})
#
#                 mask_or = torch.clamp(formula['mask'] + unit_formula['mask'], max=1)
#                 formulas[formula_length].append({'formula': formula['formula'] | unit_formula['formula'],
#                                     'mask': mask_or,
#                                     'iou': iou(E_f_b, mask_or)})
#
#         sorted(formulas[formula_length], key=itemgetter('iou'))
#         if not store_all:
#             formulas[formula_length] = formulas[formula_length][:beam_search_size]
#
#         return self.__beam_step(E_f_b, formulas, length_limit, beam_search_size, beam_search_limit, store_all)
#
#     def __beam_search(self,
#                       E_f_b: torch.Tensor,
#                       E_g_b: torch.Tensor,
#                       semantic_representations_names: list,
#                       length_limit=3,
#                       beam_search_size = 20,
#                       beam_search_limit = None,
#                       store_all = False
#                       ):
#
#         #transform names into logical elements
#         semantic_symbols = symbols(semantic_representations_names)
#
#         formulas = {1: []}
#         for j, unit in enumerate(semantic_symbols):
#             formulas[1].append({'formula': unit,
#                               'mask': E_g_b[j],
#                               'iou': iou(E_f_b, E_g_b[j])})
#             formulas[1].append({'formula': ~unit,
#                               'mask': 1. - E_g_b[j],
#                               'iou': iou(E_f_b, 1. - E_g_b[j])})
#
#         # Start recursive procedure
#         return self.__beam_step(E_f_b, formulas, length_limit, beam_search_size, beam_search_limit, store_all)
#
#     #TODO: fix that
#     def connect_individual(self,
#                              data: Dataset,
#                              function: Callable,
#                              semantic_representations: nn.Module,
#                              explanation_method,
#                              semantic_objective_fn: Callable or None,
#                              semantic_representations_names: list or None,
#                              explanation_method_init_params: dict or None,
#                              explanation_method_params: dict or None,
#                              dataloader_batch_size = 256
#                              ):
#
#         #TODO: make this adaptive
#         INPUT_DIMENTIONS = [3, 224, 224]
#
#         if semantic_objective_fn is None:
#             semantic_objective_fn = lambda x: x
#
#         # TODO: custom input dimensions
#         with torch.no_grad():
#             x = torch.randn(1, 3, 224, 224)
#             layer_dimensions = function(x)[0]
#             assert (len(layer_dimensions.shape) > 1), "Expected scalar output of explained neuron"
#
#             semantic_dimensions = semantic_objective_fn(semantic_representations(x))[0].shape
#             assert (len(semantic_dimensions) > 1), "Expected scalar output of connected representations, not {shape}".format(shape = semantic_dimensions)
#
#         # if names are not provided
#         if semantic_representations_names is None:
#             semantic_representations_names = ["Unit #{unit}".format(unit = unit) for unit in range(semantic_dimensions)]
#
#         # Setting up a dataloader
#         dataloader = DataLoader(data, batch_size=dataloader_batch_size, shuffle=False)
#
#         # STEP 1: collecting explanations wrt explaining function
#         explainer = explanation_method(model=function, *explanation_method_init_params)
#         E_f = torch.zeros([len(data), 224, 224])
#         Y_f = torch.zeros([len(data)])
#
#         _counter = 0
#         for i, x in tqdm(enumerate(dataloader)):
#             x = x.float().to(self.device)
#             with torch.no_grad():
#                 Y_f[_counter: _counter + _batch_size] = function(x)[:].data
#
#             _batch_size = x.shape[0]
#             attributions = explainer.attribute(x, target=0)
#             attributions = self.__transform_explanations(attributions)
#
#             E_f[_counter: _counter + _batch_size] = attributions.data
#             _counter += _batch_size
#
#         # STEP 2: collecting explanations wrt semantic representations
#         _composite_function = semantic_objective_fn(semantic_representations)
#         explainer = explanation_method(model=_composite_function, *explanation_method_init_params)
#         E_g = torch.zeros([semantic_dimensions, len(data), 224, 224])
#         Y_g = torch.zeros([semantic_dimensions, len(data)])
#
#         for i, x in tqdm(enumerate(dataloader)):
#             x = x.float().to(self.device)
#
#             _batch_size = x.shape[0]
#             for j in range(semantic_dimensions):
#                 with torch.no_grad():
#                     Y_g[j, _counter: _counter + _batch_size] = _composite_function(x)[:, j].data
#
#                 attributions = explainer.attribute(x, target=j)
#                 attributions = self.__transform_explanations(attributions)
#
#                 E_f[j, _counter: _counter + _batch_size] = attributions.data
#                 _counter += _batch_size
#
#         # STEP 3: Binarising explanation maps
#
#         E_f_b = self.__binarise_explanations(E_f, scores = Y_f)
#         E_g_b = torch.zeros_like(E_g)
#         for j in range(semantic_dimensions):
#             E_g_b[j] = self.__binarise_explanations(E_g[j], scores = Y_g[j])
#
#
#         # STEP 4: Beam search
#
#         formulas = self.__beam_search(E_f_b,
#                                       E_g_b,
#                                       semantic_representations_names,
#                                       length_limit = 3,
#                                       beam_search_size=20,
#                                       beam_search_limit=None,
#                                       store_all=False)
#
#         return formulas
#
#     def connect_individual_linear(self,
#                              data: Dataset,
#                              function: Callable,
#                              semantic_collection: SemanticCollection,
#                              ):
#
#         # STEP 1
#         # Setting up a dataloader
#         dataloader = DataLoader(data, batch_size=dataloader_batch_size, shuffle=False)
#
#         A_f = torch.zeros(len(data))
#         A_G = torch.zeors([len(data), len(semantic_collection),])
#
#         _counter = 0
#         for i, x in tqdm(enumerate(dataloader)):
#             x = x.float().to(self.device)
#             _batch_size = x.shape[0]
#
#             A_f[_counter: _counter + _batch_size] = self.__get_activations(x, function)
#             A_G[_counter: _counter + _batch_size] = self.__get_activations(x, semantic_collection)
#
#         # Initialize the linear model
#         lr = LinearRegression()
#         # training the model
#         lr.fit(A_G.T, A_f.T)
#
#         coefficients = lr.coef_


















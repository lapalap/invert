from invert.explainer import Invert
from invert.metrics import Metric
import torch
import pandas as pd
import json

df = pd.read_csv('/Users/kirillbykov/Documents/Work/DORA/notebooks/feature_extractor/theory/ILSVRC2012_val_labels.csv')
one_hot_labels = torch.tensor(df[df.columns[1:]].values)

data = open('assets/imagenet/ILSVRC2012_classes.json')
data = data.read()
data = json.loads(data)

model = torch.hub.load("pytorch/vision", "densenet161", weights="DEFAULT")

explainer = Invert(
        model,
        storage_dir=".invert/",
        device="cpu",
    )

explainer.load_activations(torch.load("/Users/kirillbykov/Documents/Work/INVERT/data/imagenet_val/features50k_densenet161_val.tnsr"),
                           one_hot_labels,
                           data
                           )

explainer.explain_representation(254, 5, 5, Metric())
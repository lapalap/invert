from invert.explainer import Invert
from invert.metrics import Metric
import torch
import pandas as pd
import json

# download image labes by following link 
# https://www.dropbox.com/s/kpqdvkj03blales/ILSVRC2012_val_labels.csv?dl=0
df = pd.read_csv('...')
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
# download DenseNet activations by following link
# https://www.dropbox.com/s/8fnjmanjounjrwj/features50k_densenet161_val.tnsr?dl=0
A = torch.load("...")
explainer.load_activations(A,
                           one_hot_labels,
                           data
                           )

explainer.explain_representation(254, 5, 5, Metric())
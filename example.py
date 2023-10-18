from invert.explainer import Invert
from invert.metrics import Metric
import torch
import pandas as pd
import json
from datetime import datetime

if torch.cuda.is_available():        
    #device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# download image labes by following link 
# https://www.dropbox.com/s/kpqdvkj03blales/ILSVRC2012_val_labels.csv?dl=0
#df = pd.read_csv('...')
df = pd.read_csv("/mnt/scratch/bio/lkopf/broden1_224/coco20k_labels.csv")
one_hot_labels = torch.tensor(df[df.columns[1:]].values)

# data = open('assets/imagenet/ILSVRC2012_classes.json')
# data = data.read()
# data = json.loads(data)

S2I = {
  "__background__": 0,
  "person": 1,
  "bicycle": 2,
  "car": 3,
  "motorcycle": 4,
  "airplane": 5,
  "bus": 6,
  "train": 7,
  "truck": 8,
  "boat": 9,
  "traffic light": 10,
  "fire hydrant": 11,
  "na_12" : 12,
  "stop sign": 13,
  "parking meter": 14,
  "bench": 15,
  "bird": 16,
  "cat": 17,
  "dog": 18,
  "horse": 19,
  "sheep": 20,
  "cow": 21,
  "elephant": 22,
  "bear": 23,
  "zebra": 24,
  "giraffe": 25,
  "na_26": 26,
  "backpack": 27,
  "umbrella": 28,
  "na_29": 29,
  "na_30": 30,
  "handbag": 31,
  "tie": 32,
  "suitcase": 33,
  "frisbee": 34,
  "skis": 35,
  "snowboard": 36,
  "sports ball": 37,
  "kite": 38,
  "baseball bat": 39,
  "baseball glove": 40,
  "skateboard": 41,
  "surfboard": 42,
  "tennis racket": 43,
  "bottle": 44,
  "na_45": 45,
  "wine glass": 46,
  "cup": 47,
  "fork": 48,
  "knife": 49,
  "spoon": 50,
  "bowl": 51,
  "banana": 52,
  "apple": 53,
  "sandwich": 54,
  "orange": 55,
  "broccoli": 56,
  "carrot": 57,
  "hot dog": 58,
  "pizza": 59,
  "donut": 60,
  "cake": 61,
  "chair": 62,
  "couch": 63,
  "potted plant": 64,
  "bed": 65,
  "na_66": 66,
  "dining table": 67,
  "na_68": 68,
  "na_69": 69,
  "toilet": 70,
  "na_71": 71,
  "tv": 72,
  "laptop": 73,
  "mouse": 74,
  "remote": 75,
  "keyboard": 76,
  "cell phone": 77,
  "microwave": 78,
  "oven": 79,
  "toaster": 80,
  "sink": 81,
  "refrigerator": 82,
  "na_83": 83,
  "book": 84,
  "clock": 85,
  "vase": 86,
  "scissors": 87,
  "teddy bear": 88,
  "hair drier": 89,
  "toothbrush": 90
}

#S2I = {f"{k}-s": v for k, v in S2I.items()}

I2S = {v: k for k, v in S2I.items()}

data = I2S

#model = torch.hub.load("pytorch/vision", "densenet161", weights="DEFAULT")

model = None

explainer = Invert(
        model,
        storage_dir=".invert/",
        #device="cpu",
        device=device,
    )
# download DenseNet activations by following link
# https://www.dropbox.com/s/8fnjmanjounjrwj/features50k_densenet161_val.tnsr?dl=0
#A = torch.load("...")
A = torch.load("/mnt/scratch/bio/lkopf/logits_fc/maskrcnn_resnet50_fpn/class_logits_area_all.pt")
explainer.load_activations(A,
                           one_hot_labels,
                           data
                           )
FORMULA_LENGTH = 1
B = 1 # number of top formulas
i = 25

#explainer.explain_representation(254, 5, 5, Metric())
explanation = explainer.explain_representation_fast(i, FORMULA_LENGTH, B, Metric())

print(explanation)
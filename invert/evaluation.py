import os
import cv2
import torch
import torch.nn as nn
from torch import nn, Tensor
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.datasets.vision import VisionDataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from collections import OrderedDict
from pycocotools.mask import encode, decode


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SemanticSegmentation(nn.Module):
    """ SemanticSegmenttion preset from PyTorch
        https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py
    """

    def __init__(
        self,
        *,
        resize_size: Optional[int],
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation = InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.resize_size = [resize_size] if resize_size is not None else None
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img: Tensor) -> Tensor:
        if isinstance(self.resize_size, list):
            img = F.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``. "
            f"Finally the values are first rescaled to ``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and "
            f"``std={self.std}``."
        )
    

class CocoDataset(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform = None,
        target_transform = None,
        transforms = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cat_ids = self.coco.getCatIds()

    def _load_image(self, id: int) -> Image.Image:
        path = os.path.join(self.root,self.coco.loadImgs(id)[0]["file_name"])
        input_img = np.array(Image.open(path).convert("RGB"))       
        resized_img = cv2.resize(input_img, (224, 224), interpolation=cv2.INTER_NEAREST)
        resized_img = Image.fromarray(resized_img)
        image = F.pil_to_tensor(resized_img)
        image = F.convert_image_dtype(image, torch.float)
        return image

    def _load_mask(self, id: int):
        img = self.coco.imgs[id]
        anns_ids = self.coco.getAnnIds(id)
        anns = self.coco.loadAnns(anns_ids)
        mask_tens = torch.zeros([91,224,224])
        if len(anns) > 0:
            anns_ids = self.coco.getAnnIds(imgIds=img["id"], catIds=self.cat_ids, iscrowd=None)
            anns_dict = dict()
            for i, ann in enumerate(anns):
                anns_dict[i] = ann
            anns_ordered = OrderedDict(sorted(anns_dict.items(), key = lambda x: x[1]["category_id"]))		
            n = dict()
            # iterate over each group of categories and save them separately grouped in one segmentation image
            for k, v in anns_ordered.items():
                if 0 not in n:
                    n[0] = v["category_id"]
                    anns_img = np.zeros((img["height"],img["width"]))
                    anns_sub = [v for v in anns_ordered.values() if n[0] in v.values()]
                    for ann in anns_sub:
                        anns_img = np.maximum(anns_img,self.coco.annToMask(ann)*ann["category_id"])	
                    binmask = np.asfortranarray(anns_img, dtype=np.uint8)
                    rle = encode(binmask)
                    bitmap_mask = decode(rle)
                    resized_seg_img_bin = cv2.resize(bitmap_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
                    class_id = int(v["category_id"])
                    mask_tens[class_id,:,:] = torch.Tensor(resized_seg_img_bin)
                elif v["category_id"] > n[0]:
                    n[0] = v["category_id"]
                    anns_img = np.zeros((img["height"],img["width"]))
                    anns_sub = [v for v in anns_ordered.values() if n[0] in v.values()]
                    for ann in anns_sub:
                        anns_img = np.maximum(anns_img,self.coco.annToMask(ann)*ann["category_id"])	
                    binmask = np.asfortranarray(anns_img, dtype=np.uint8)
                    rle = encode(binmask)
                    bitmap_mask = decode(rle)
                    resized_seg_img_bin = cv2.resize(bitmap_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
                    class_id = int(v["category_id"])
                    mask_tens[class_id,:,:] = torch.Tensor(resized_seg_img_bin)
        return mask_tens

    def _load_image_name(self, id: int):
        image_name = self.coco.imgs[id]["file_name"]
        return image_name

    def _load_image_label(self, id: int):
        anns_ids = self.coco.getAnnIds(id)
        anns = self.coco.loadAnns(anns_ids)
        image_anns = list(set([ann["category_id"] for ann in anns]))
        idx_labels = dict(zip(list(range(91)), [0 for i in range(91)]))
        for i in image_anns:
            if i in idx_labels:
                idx_labels[int(i)] = 1
        image_label = torch.tensor(list(idx_labels.values()), dtype=torch.float)
        return image_label

    def __getitem__(self, index: int):
        id = self.ids[index]
        if len(self.coco.loadAnns(self.coco.getAnnIds(id))) == 0:
            return None
        else:      
            image = self._load_image(id)
            mask = self._load_mask(id)
            image_name = self._load_image_name(id)
            image_label = self._load_image_label(id)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask, image_name, image_label

    def __len__(self) -> int:
        return len(self.ids)


class VocCocoDataset(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform = None,
        target_transform = None,
        transforms = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cat_ids = self.coco.getCatIds()
        self.coco_to_voc_map = {
                0: 0,   # "__background__" -> "__background__"
                5: 1,   # airplane -> aeroplane
                2: 2,   # bicycle -> bicycle
                16: 3,  # bird -> bird
                9: 4,   # boat -> boat
                44: 5,  # bottle -> bottle
                6: 6,   # bus -> bus
                3: 7,   # car -> car
                17: 8,  # cat -> cat
                62: 9,  # chair -> chair
                21: 10, # cow -> cow
                67: 11, # dining table -> diningtable
                18: 12, # dog -> dog
                19: 13, # horse -> horse
                4: 14,  # motorcycle -> motorbike
                1: 15,  # person -> person
                64: 16, # potted plant -> pottedplant
                20: 17, # sheep -> sheep
                63: 18, # couch -> sofa
                7: 19,  # train -> train
                72: 20  # tv -> tvmonitor
            }
        self.size = 520

    def _load_image(self, id: int) -> Image.Image:
        path = os.path.join(self.root,self.coco.loadImgs(id)[0]["file_name"])
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_mask(self, id: int):
        img = self.coco.imgs[id]
        anns_ids = self.coco.getAnnIds(id)
        anns = self.coco.loadAnns(anns_ids)
        mask_tens = torch.zeros([21,self.size,self.size])
        if len(anns) > 0:
            anns_ids = self.coco.getAnnIds(imgIds=img["id"], catIds=self.cat_ids, iscrowd=None)
            anns_dict = dict()
            for i, ann in enumerate(anns):
                anns_dict[i] = ann
            anns_ordered = OrderedDict(sorted(anns_dict.items(), key = lambda x: x[1]["category_id"]))		
            n = dict()
            for k, v in anns_ordered.items():
                if 0 not in n:
                    n[0] = v["category_id"]
                    anns_img = np.zeros((img["height"],img["width"]))
                    anns_sub = [v for v in anns_ordered.values() if n[0] in v.values()]
                    for ann in anns_sub:
                        anns_img = np.maximum(anns_img,self.coco.annToMask(ann)*ann["category_id"])	
                    binmask = np.asfortranarray(anns_img, dtype=np.uint8)
                    rle = encode(binmask)
                    bitmap_mask = decode(rle)
                    resized_seg_img_bin = cv2.resize(bitmap_mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                    class_id = int(v["category_id"])
                    if class_id in list(self.coco_to_voc_map.keys()):
                        class_id = self.coco_to_voc_map[int(v["category_id"])]
                        mask_tens[class_id,:,:] = torch.Tensor(resized_seg_img_bin)
                elif v["category_id"] > n[0]:
                    n[0] = v["category_id"]
                    anns_img = np.zeros((img["height"],img["width"]))
                    anns_sub = [v for v in anns_ordered.values() if n[0] in v.values()]
                    for ann in anns_sub:
                        anns_img = np.maximum(anns_img,self.coco.annToMask(ann)*ann["category_id"])	
                    binmask = np.asfortranarray(anns_img, dtype=np.uint8)
                    rle = encode(binmask)
                    bitmap_mask = decode(rle)
                    resized_seg_img_bin = cv2.resize(bitmap_mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                    class_id = int(v["category_id"])
                    if class_id in list(self.coco_to_voc_map.keys()):
                        class_id = self.coco_to_voc_map[int(v["category_id"])]
                        mask_tens[class_id,:,:] = torch.Tensor(resized_seg_img_bin)
        return mask_tens

    def _load_image_name(self, id: int):
        image_name = self.coco.imgs[id]["file_name"]
        return image_name

    def _load_image_label(self, id: int):
        anns_ids = self.coco.getAnnIds(id)
        anns = self.coco.loadAnns(anns_ids)
        image_anns = list(set([ann["category_id"] for ann in anns]))
        voc_anns = [cat_id for cat_id in image_anns if cat_id in list(self.coco_to_voc_map.keys())]
        idx_labels = dict(zip(list(range(21)), [0 for i in range(21)]))
        for i in voc_anns:
            idx_labels[self.coco_to_voc_map[int(i)]] = 1
        image_label = torch.tensor(list(idx_labels.values()), dtype=torch.float)
        return image_label

    def __getitem__(self, index: int):
        id = self.ids[index]
        anns_ids = self.coco.getAnnIds(id)
        anns = self.coco.loadAnns(anns_ids)
        image_anns = list(set([ann["category_id"] for ann in anns]))
        voc_anns = [cat_id for cat_id in image_anns if cat_id in list(self.coco_to_voc_map.keys())]
        if len(voc_anns) == 0:
            return None
        else:      
            image = self._load_image(id)
            mask = self._load_mask(id)
            image_name = self._load_image_name(id)
            image_label = self._load_image_label(id)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, mask, image_name, image_label

    def __len__(self) -> int:
        return len(self.ids)


def collate_fn(batch):
    # Filter None values in Dataset that don"t have coco labels
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class CustomMaskRCNN(nn.Module):
    def __init__(self):
        super(CustomMaskRCNN, self).__init__()
        self.maskrcnn = maskrcnn_resnet50_fpn(pretrained=True)
        self.to(device)

    def forward(self, input):
        output = self.maskrcnn(input)
        # Transform the output dictionary into an output tensor       
        class_logits = torch.zeros([input.shape[0], 91, 224, 224]).to(device)
       
        for i in range(len(output)):
            for j, label in enumerate(output[i]["labels"]):
                mask = output[i]["masks"][j][0]
                class_logits[i][label] = torch.max(class_logits[i][label], mask)
        
        return class_logits


def get_area(input):
    count_elements_above_0_5 = (input[:, :, :, :] >= 0.5).sum(dim=(2, 3))
    total_count_elements = input.size(-2) * input.size(-1)
    output_area = count_elements_above_0_5 / total_count_elements
    return output_area


def flatten_tens(input):
    flat = torch.flatten(input.permute(1,0,2,3), start_dim=1)
    return flat


def pairwise_sum(x, y):
    x = torch.reshape(x.sum(axis=1), (-1, 1)) # shape (N, 1)
    y = torch.reshape(y.sum(axis=1), (-1, 1)) # shape (N, 1)
    z = x + torch.transpose(y, 0, 1) # shape (N, N)
    return z

empty_coco_classes = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]

voc_idx_class = {
        0: "__background__",
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "diningtable",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "pottedplant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tvmonitor"}

coco_idx_class = {
        0: "__background__",
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        12: "N/A",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        26: "N/A",
        27: "backpack",
        28: "umbrella",
        29: "N/A",
        30: "N/A",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        45: "N/A",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        66: "N/A",
        67: "dining table",
        68: "N/A",
        69: "N/A",
        70: "toilet",
        71: "N/A",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        83: "N/A",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush"}


class QuantileVector:
    """
    Original code from compexp:
    https://github.com/jayelm/compexp/blob/master/vision/util/vecquantile.py

    Streaming randomized quantile computation for numpy.

    Add any amount of data repeatedly via add(data).  At any time,
    quantile estimates (or old-style percentiles) can be read out using
    quantiles(q) or percentiles(p).

    Accuracy scales according to resolution: the default is to
    set resolution to be accurate to better than 0.1%,
    while limiting storage to about 50,000 samples.

    Good for computing quantiles of huge data without using much memory.
    Works well on arbitrary data with probability near 1.

    Based on the optimal KLL quantile algorithm by Karnin, Lang, and Liberty
    from FOCS 2016.  http://ieee-focs.org/FOCS-2016-Papers/3933a071.pdf
    """

    def __init__(
        self, depth=1, resolution=24 * 1024, buffersize=None, dtype=None, seed=None
    ):
        self.resolution = resolution
        self.depth = depth
        # Default buffersize: 128 samples (and smaller than resolution).
        if buffersize is None:
            buffersize = min(128, (resolution + 7) // 8)
        self.buffersize = buffersize
        self.samplerate = 1.0
        self.data = [np.zeros(shape=(depth, resolution), dtype=dtype)]
        self.firstfree = [0]
        self.random = np.random.RandomState(seed)
        self.extremes = np.empty(shape=(depth, 2), dtype=dtype)
        self.extremes.fill(np.NaN)
        self.size = 0

    def add(self, incoming):
        assert len(incoming.shape) == 2
        assert incoming.shape[1] == self.depth
        self.size += incoming.shape[0]
        # Convert to a flat numpy array.
        if self.samplerate >= 1.0:
            self._add_every(incoming)
            return
        # If we are sampling, then subsample a large chunk at a time.
        self._scan_extremes(incoming)
        chunksize = np.ceil[self.buffersize / self.samplerate]
        for index in range(0, len(incoming), chunksize):
            batch = incoming[index : index + chunksize]
            sample = batch[self.random.binomial(1, self.samplerate, len(batch))]
            self._add_every(sample)

    def _add_every(self, incoming):
        supplied = len(incoming)
        index = 0
        while index < supplied:
            ff = self.firstfree[0]
            available = self.data[0].shape[1] - ff
            if available == 0:
                if not self._shift():
                    # If we shifted by subsampling, then subsample.
                    incoming = incoming[index:]
                    if self.samplerate >= 0.5:
                        print("SAMPLING")
                        self._scan_extremes(incoming)
                    incoming = incoming[
                        self.random.binomial(1, 0.5, len(incoming - index))
                    ]
                    index = 0
                    supplied = len(incoming)
                ff = self.firstfree[0]
                available = self.data[0].shape[1] - ff
            copycount = min(available, supplied - index)
            self.data[0][:, ff : ff + copycount] = np.transpose(
                incoming[index : index + copycount, :]
            )
            self.firstfree[0] += copycount
            index += copycount

    def _shift(self):
        index = 0
        # If remaining space at the current layer is less than half prev
        # buffer size (rounding up), then we need to shift it up to ensure
        # enough space for future shifting.
        while self.data[index].shape[1] - self.firstfree[index] < (
            -(-self.data[index - 1].shape[1] // 2) if index else 1
        ):
            if index + 1 >= len(self.data):
                return self._expand()
            data = self.data[index][:, 0 : self.firstfree[index]]
            data.sort()
            if index == 0 and self.samplerate >= 1.0:
                self._update_extremes(data[:, 0], data[:, -1])
            offset = self.random.binomial(1, 0.5)
            position = self.firstfree[index + 1]
            subset = data[:, offset::2]
            self.data[index + 1][:, position : position + subset.shape[1]] = subset
            self.firstfree[index] = 0
            self.firstfree[index + 1] += subset.shape[1]
            index += 1
        return True

    def _scan_extremes(self, incoming):
        # When sampling, we need to scan every item still to get extremes
        self._update_extremes(
            np.nanmin(incoming, axis=0), np.nanmax(incoming, axis=0)
        )

    def _update_extremes(self, minr, maxr):
        self.extremes[:, 0] = np.nanmin([self.extremes[:, 0], minr], axis=0)
        self.extremes[:, -1] = np.nanmax([self.extremes[:, -1], maxr], axis=0)

    def minmax(self):
        if self.firstfree[0]:
            self._scan_extremes(self.data[0][:, : self.firstfree[0]].transpose())
        return self.extremes.copy()

    def _expand(self):
        cap = self._next_capacity()
        if cap > 0:
            # First, make a new layer of the proper capacity.
            self.data.insert(
                0, np.empty(shape=(self.depth, cap), dtype=self.data[-1].dtype)
            )
            self.firstfree.insert(0, 0)
        else:
            # Unless we"re so big we are just subsampling.
            assert self.firstfree[0] == 0
            self.samplerate *= 0.5
        for index in range(1, len(self.data)):
            # Scan for existing data that needs to be moved down a level.
            amount = self.firstfree[index]
            if amount == 0:
                continue
            position = self.firstfree[index - 1]
            # Move data down if it would leave enough empty space there
            # This is the key invariant: enough empty space to fit half
            # of the previous level"s buffer size (rounding up)
            if self.data[index - 1].shape[1] - (amount + position) >= (
                -(-self.data[index - 2].shape[1] // 2) if (index - 1) else 1
            ):
                self.data[index - 1][:, position : position + amount] = self.data[
                    index
                ][:, :amount]
                self.firstfree[index - 1] += amount
                self.firstfree[index] = 0
            else:
                # Scrunch the data if it would not.
                data = self.data[index][:, :amount]
                data.sort()
                if index == 1:
                    self._update_extremes(data[:, 0], data[:, -1])
                offset = self.random.binomial(1, 0.5)
                scrunched = data[:, offset::2]
                self.data[index][:, : scrunched.shape[1]] = scrunched
                self.firstfree[index] = scrunched.shape[1]
        return cap > 0

    def _next_capacity(self):
        cap = np.ceil(self.resolution * np.power(0.67, len(self.data)))
        if cap < 2:
            return 0
        return max(self.buffersize, int(cap))

    def _weighted_summary(self, sort=True):
        if self.firstfree[0]:
            self._scan_extremes(self.data[0][:, : self.firstfree[0]].transpose())
        size = sum(self.firstfree) + 2
        weights = np.empty(shape=(size), dtype="float32")  # floating point
        summary = np.empty(shape=(self.depth, size), dtype=self.data[-1].dtype)
        weights[0:2] = 0
        summary[:, 0:2] = self.extremes
        index = 2
        for level, ff in enumerate(self.firstfree):
            if ff == 0:
                continue
            summary[:, index : index + ff] = self.data[level][:, :ff]
            weights[index : index + ff] = np.power(2.0, level)
            index += ff
        assert index == summary.shape[1]
        if sort:
            order = np.argsort(summary)
            summary = summary[np.arange(self.depth)[:, None], order]
            weights = weights[order]
        return (summary, weights)

    def quantiles(self, quantiles, old_style=False):
        if self.size == 0:
            return np.full((self.depth, len(quantiles)), np.nan)
        summary, weights = self._weighted_summary()
        cumweights = np.cumsum(weights, axis=-1) - weights / 2
        if old_style:
            # To be convenient with np.percentile
            cumweights -= cumweights[:, 0:1]
            cumweights /= cumweights[:, -1:]
        else:
            cumweights /= np.sum(weights, axis=-1, keepdims=True)
        result = np.empty(shape=(self.depth, len(quantiles)))
        for d in range(self.depth):
            result[d] = np.interp(quantiles, cumweights[d], summary[d])
        return result

    def integrate(self, fun):
        result = None
        for level, ff in enumerate(self.firstfree):
            if ff == 0:
                continue
            term = np.sum(
                fun(self.data[level][:, :ff]) * np.power(2.0, level), axis=-1
            )
            if result is None:
                result = term
            else:
                result += term
        if result is not None:
            result /= self.samplerate
        return result

    def percentiles(self, percentiles):
        return self.quantiles(percentiles, old_style=True)

    def readout(self, count, old_style=True):
        return self.quantiles(np.linspace(0.0, 1.0, count), old_style=old_style)


if __name__ == "__main__":
    import time

    # An adverarial case: we keep finding more numbers in the middle
    # as the stream goes on.
    amount = 10000000
    percentiles = 1000
    data = np.arange(float(amount))
    data[1::2] = data[-1::-2] + (len(data) - 1)
    data /= 2
    depth = 50
    alldata = data[:, None] + (np.arange(depth) * amount)[None, :]
    actual_sum = np.sum(alldata * alldata, axis=0)
    amt = amount // depth
    for r in range(depth):
        np.random.shuffle(alldata[r * amt : r * amt + amt, r])
    # data[::2] = data[-2::-2]
    # np.random.shuffle(data)
    starttime = time.time()
    qc = QuantileVector(depth=depth, resolution=8 * 1024)
    qc.add(alldata)
    ro = qc.readout(1001)
    endtime = time.time()
    # print "ro", ro
    # print ro - np.linspace(0, amount, percentiles+1)
    gt = (
        np.linspace(0, amount, percentiles + 1)[None, :]
        + (np.arange(qc.depth) * amount)[:, None]
    )
    print(
        "Maximum relative deviation among %d perentiles:" % percentiles,
        (np.max(abs(ro - gt) / amount) * percentiles),
    )
    print(
        "Minmax eror %f, %f"
        % (
            max(abs(qc.minmax()[:, 0] - np.arange(qc.depth) * amount)),
            max(abs(qc.minmax()[:, -1] - (np.arange(qc.depth) + 1) * amount + 1)),
        )
    )
    print(
        "Integral error:",
        np.max(np.abs(qc.integrate(lambda x: x * x) - actual_sum) / actual_sum),
    )
    print(
        "Count error: ",
        (qc.integrate(lambda x: np.ones(x.shape[-1])) - qc.size) / (0.0 + qc.size),
    )
    print("Time", (endtime - starttime))

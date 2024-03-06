# these imports are necessary for manual collection of activations
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import ImageFolder
from glob import glob
import cv2
import torchvision

IMAGENET_TRANSFORMS = torchvision.transforms.Compose([
                                torchvision.transforms.ToPILImage(),
                            torchvision.transforms.Resize(224),
                            torchvision.transforms.CenterCrop((224,224)),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                            std=[0.26862954, 0.26130258, 0.27577711])
                        ])

class ImageDataset(Dataset):
        def __init__(self,root,transform):
            self.root=root
            self.transform=transform

            self.image_names=glob(self.root + '*')
            self.image_names.sort()
    
        #The __len__ function returns the number of samples in our dataset.
        def __len__(self):
            return len(self.image_names)
    
        def __getitem__(self,index):
            image=cv2.imread(self.image_names[index])
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            image=self.transform(image)

            return image
    
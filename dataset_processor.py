import os
import torch
import cv2 as cv
import numpy as np
from PIL import Image
from torchvision import transforms

class GenericSimpleOneHotDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir : str, augmentation = transforms.Compose([transforms.ToTensor()])) -> None:
        self.dataset = []

        class_one_path = os.path.join(root_dir, "2")
        self.__addtodataset__(class_one_path, [0, 0, 1])

        class_two_path = os.path.join(root_dir, "1")
        self.__addtodataset__(class_two_path, [0, 1, 0])
        
        class_three_path = os.path.join(root_dir, "0")
        self.__addtodataset__(class_three_path, [1, 0, 0])

        self.augmentation = augmentation

    # helper class to load files
    def __addtodataset__(self, full_dir_path : str, label : list) -> None:
        for fname in os.listdir(full_dir_path):
            if fname.endswith(".png"):
                fpath = os.path.join(full_dir_path, fname)
                self.dataset.append( (fpath, label) )

    # return the size of the dataset
    def __len__(self) -> int:
        return len(self.dataset)

    # grab one item form the dataset
    def __getitem__(self, index: int):
        fpath, label = self.dataset[index]

        # load image into numpy RGB numpy array in pytorch format
        image = Image.open(fpath)
        image = transforms.ToTensor()(image)
        image = image.repeat(3, 1, 1)
        image = self.augmentation(image)

        # minmax norm the image
        image = (image - image.min()) / (image.max() - image.min())
        label = torch.Tensor(label)

        return image, label
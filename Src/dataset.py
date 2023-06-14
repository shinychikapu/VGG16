import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from .util import reformat
from pathlib import Path
import PIL.Image
import numpy as np

class catDogDataset(torch.utils.data.Dataset):
    def __init__(self, rootFolder, transform = None):
        '''Data set class constructor

        Attribute
        -------
        rootFolder : absolute or relative path to the folder containing data
        '''
        self.rootFolder = rootFolder
        self.paths = []
        self.labels = []
        self.transform = transform
        self.getImgsAndLabels()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,  index):
        img_pth = self.paths[index]
        img = PIL.Image.open(img_pth)
        img_transformed = self.transform(img)
        return img_transformed, self.labels[index]
    
    def getImgsAndLabels(self):

        current_dir = Path.cwd()
        dir = current_dir / self.rootFolder

        file_extensions = ['.jpg', '.jpeg', '.png']

        for class_dir in dir.iterdir():
            if class_dir.is_dir():
                # Get the folder name as the class label    
                if class_dir.name == "Cat":
                    class_label = 1
                else: 
                    class_label = 0  

                # Iterate over the image files in the class folder
                for ext in file_extensions:
                    for image_path in class_dir.glob('*' + ext):
                        self.paths.append(image_path)
                        self.labels.append(class_label)

    
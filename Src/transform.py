import torch
from torchvision import models, transforms

class ImageTransform():
    
    def __init__(self, resize, mean, std):
        self.resize = resize
        self.mean = mean
        self.std = std
        
        self.data_transform = transforms.Compose([
                                transforms.Resize((resize, resize)),
                                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                self.normalize
                            ])
        
    def __call__(self, img):
        return self.data_transform(img)
    
    def normalize(self, tensor):
        if tensor.shape[0] == 1:
            # Convert single-channel image to three channels
            tensor = torch.cat([tensor] * 3, dim=0)
        elif tensor.shape[0] == 4:
            # Convert RGBA image to RGB
            tensor = tensor[:3, :, :]
        
        return transforms.Normalize(self.mean, self.std)(tensor)
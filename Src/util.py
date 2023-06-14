import cv2 as cv
import torch
from pathlib import Path
import numpy as np
import PIL.Image
from Src.transform import ImageTransform
def reformat(img_path):
        '''This function take in a image path and return it as a (1, 224, 224) tensor to pass into the VGG16 Model
        ,
        Attribute:
        ---------
        img_path : the path to the image on the computer
        '''
        img = read_image(img_path) # H, W, C
        #resize to 224x224
        img_ = cv.resize(img, (224, 224))

        #reshape 
        if img_.shape == (224, 224):
            img_= cv.cvtColor(img_,cv.COLOR_GRAY2RGB)
        # #get the color
        # col = img_.shape[2]
        #convert to tensor
        img_ = torch.from_numpy(img_).reshape(3, 224, 224).float()

        return img_

def read_image(path: Path):
        """This function read an image from a path.
        The read is perform using PIL.Image (cause PyTorch).
        """
        size = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        paths = []
        imgs = []
        file_extensions = ['.jpg', '.jpeg', '.png']

        current_dir = Path.cwd()
        dir = current_dir / path

        for ext in file_extensions:
            for image_path in dir.glob('*' + ext):
                paths.append(image_path)

        transformer = ImageTransform(size, mean, std)

        for path in paths:
            img = PIL.Image.open(path)
            img_ = transformer.__call__(img)
            imgs.append(img_)
        return imgs, paths
                
    

def load_checkpoint(path, device, model, optimizer = None):
        state = torch.load(path)
        epoch = state['epoch']
        train_loss = state['train_loss']
        val_loss = state['val_loss']

        model.load_state_dict(state['model'])
        model = model.to(device)
        if optimizer != None:
            optimizer.load_state_dict(state['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(True, epoch))
        print("Checkpoint's train loss is: {:.4f}".format(train_loss))
        print("Checkpoint's validation loss is: {:.4f}".format(val_loss))
        return epoch, train_loss, val_loss

def save_checkpoint(state, path:Path, filename='lastest.pth.tar'):
  out_path = path / filename
  torch.save(state, out_path)
import argparse
import torch
from Src.train import train_mode, test_mode, queury_mode
from Src.dataset import catDogDataset
from torch.utils.data import DataLoader
import cv2 as cv
from Src.transform import ImageTransform

parser = argparse.ArgumentParser(description='This is a VGG16 model trained to classify dogs and cats. User can either use train mode or test mode.')
subparser = parser.add_subparsers(dest='mode')

train = subparser.add_parser("train")
test = subparser.add_parser("test")
queury = subparser.add_parser("queury")

train.add_argument("--pretrained", help = "If you want to train a pretrained model then input the path to the model. Otherwise, leave blank")
train.add_argument("--num_epoch", type = int, help = "Enter how many epoch do you want to train the model on")

test.add_argument("--mod_path", help = "path to the model which will be test on")

queury.add_argument("--queury_path", help = "path to the queury folder")
queury.add_argument("--mod", help = "input the path to the model you want to use")
if __name__ == "__main__":
    args = parser.parse_args()
    # Config
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if args.mode == "train":
        #Load data to train
        train_set = catDogDataset(r"Data\train", transform=ImageTransform(size, mean, std))
        val_set = catDogDataset(r"Data\val", transform=ImageTransform(size, mean, std))
        train_dataLoader = DataLoader(dataset = train_set,
                            batch_size = 20, 
                            shuffle= True,
                            drop_last= True)

        val_dataLoader = DataLoader(dataset = val_set,
                                batch_size = 20,
                                shuffle= True,
                                drop_last= True)
        dataloader_dict = {'train': train_dataLoader, 'val': val_dataLoader}
        for picture, label in train_dataLoader:
            print(f"Shape of X [N, C, H, W]: {picture.shape}")
            print(f"Shape of y: {label.shape} {label.dtype}")
            break

        print("------Begin Training-------")
        train_mode(args.num_epoch, train_dataLoader, val_dataLoader, args.pretrained)
        #train_model(args.num_epoch, dataloader_dict, args.pretrained)
        print("--------Done--------")
    
    if args.mode == "test":
        data_set = catDogDataset("C:/VGG16/Data/test", transform=ImageTransform(size, mean, std))

        dataLoader = DataLoader(dataset = data_set,
                                batch_size = 20,
                                shuffle= True,
                                drop_last= True)
        for picture, label in dataLoader:
            print(f"Shape of X [N, C, H, W]: {picture.shape}")
            print(f"Shape of y: {label.shape} {label.dtype}")
            break
        test_mode(dataLoader, args.mod_path)
    
    if args.mode == "queury":
        queury_mode(args.queury_path, args.mod)

import torch
from .model import VGG16
from torch import nn
from .util import save_checkpoint, load_checkpoint, read_image
from pathlib import Path
import numpy as np
from torchvision import models
from tqdm import tqdm

def train(dataloader, model, loss_fun, optimizer, device):
    model.train()
    num_batches = len(dataloader)
    epoch_loss = 0
    for batch, (pic, lab) in tqdm(enumerate(dataloader), total = num_batches):
        pic, lab = pic.to(device), lab.to(device)
        
        # Compute prediction error
        pred = model(pic)
        loss = loss_fun(pred, lab)

        # Backpropagation
        optimizer.zero_grad() #clear the model's gradient to avoid gradient accumulation
        loss.backward() #compute the loss function's gradients
        optimizer.step() #update the parameters

         #Calculate loss
        epoch_loss = epoch_loss + loss.item()

    avg_loss = epoch_loss / num_batches
    print("Epoch's avg train loss:", avg_loss)
    return avg_loss

def test_validate(dataloader,model, loss_fun, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for pic, lab in dataloader:
            pic, lab = pic.to(device), lab.to(device)
            pred = model(pic)
            test_loss += loss_fun(pred, lab).item() #get the loss
            correct += (pred.argmax(1) == lab).type(torch.float).sum().item() #get how many times the model guess correctuly
    test_loss /= num_batches #loss per batch
    correct /= size #accuracy
    print(f"Test/Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

def train_mode(epochs, train_dataLoader, val_dataLoader, model_path = None):
    
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Using {device} device")

    #create model if no path is provided
    model = models.vgg16(pretrained =True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    params_to_update = []

    update_params_name = ['classifier.6.weight', 'classifier.6.bias']
    for name, param in model.named_parameters():
        if name in update_params_name:
            param.requires_grad = True
            params_to_update.append(param)
            print(name)
        else:
            param.requires_grad = False

    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss = np.inf
    val_loss = np.inf
    if model_path != None:
        startEpoch, train_loss, val_loss = load_checkpoint(Path(model_path),
                                                        device,
                                                        model, 
                                                        optimizer)

 
    
    model.to(device)
    save_path = "model"

    for epoch in range(epochs):
        print("Epoch:", epoch + 1)
        epoch_train_loss = train(train_dataLoader, model, loss_fun, optimizer, device)
        epoch_val_loss = test_validate(val_dataLoader, model, loss_fun, device)

        if (epoch_train_loss < train_loss): #lowest loss on train set
            print("Best train loss so far, saving this model...")
            train_loss = epoch_train_loss
            save_checkpoint({
                'epoch': epoch,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, Path(save_path), 'best_train.pth.tar')

        if (epoch_val_loss < val_loss): #lowest loss on val set
            print("Best val loss so far, saving this model...")
            val_loss = epoch_val_loss
            save_checkpoint({
                'epoch': epoch,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, Path(save_path), 'best.pth.tar')

def test_mode(test_dataLoader, model_path):
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Using {device} device")
    model = models.vgg16(pretrained =True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    loss_fun = nn.CrossEntropyLoss()
    startEpoch, train_loss, val_loss = load_checkpoint(Path(model_path),
                                                        device,
                                                        model
                                                        )
    print("----------Begin Testing----------")
    test_validate(test_dataLoader, model, loss_fun, device)
    print("-------Done--------")

def queury_mode(path, mod_path):
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    model = models.vgg16(pretrained =True)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    startEpoch, train_loss, val_loss = load_checkpoint(Path(mod_path),
                                                        device,
                                                        model
                                                        )
    classes = ["dog", "cat"]
    
    imgs, paths = read_image(path)
    for i in range(len(imgs)):
        img = imgs[i].unsqueeze(0).to(device)  # Add an extra dimension (batch size)
        with torch.no_grad():
            model.eval()  
            output = model(img)
            pred = classes[output[0].argmax()]
        print("File:", paths[i], ", prediction:", pred)


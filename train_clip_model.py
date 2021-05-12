import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, models
from torchvision.transforms import *

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import wandb
import argparse
from math import sqrt

from utils.util import *
from utils.model_io import *
from models.models import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--load_name", type=str,
                        help="To load and continue training model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_path", type=str,
                        default="data/tiny-imagenet-200/")
    
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # setup wandb
    wandb.init(project="182-final-proj")
    wandb.config.update(args)
    
    # train
    start_training(args)


def start_training(config):
    trfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])

    # Setup data loaders
    train_set = datasets.ImageFolder(config.data_path + '/train', trfms)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    print(f"train set size: {len(train_set)}")

    #val dataloader
    val_set = datasets.ImageFolder(config.data_path + '/val-fixed', trfms)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    print(f"val set size: {len(val_set)}")

    dataloaders = {'train': train_loader, 'val': val_loader}

    # instantiate model + optimizer
    
    model = ClipStudent(512, 200)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss = nn.CrossEntropyLoss()

    # load model + optimizer
    if config.load_name:
        print("Loading model...")
        model_dict, optim_dict = load_model_checkpoint(config.load_name)

        model.load_state_dict(model_dict)
        optimizer.load_state_dict(optim_dict)
    
    train_model(model, loss, optimizer, dataloaders, config.save_name, num_epochs=config.epochs)

def train_model(model, criterion, optimizer, dataloaders, save_name, num_epochs=100):
    since = time.time()

    best_loss = 1000
    best_acc = 0.0
    
    wandb.watch(model, log='gradients')
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, loss = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    assert preds.shape == labels.shape, "Loss on predictions and labels must have same shape\n" + \
                        f"Predictions: {preds.shape}\nLabels: {labels.shape}"

                    class_loss = criterion(outputs, labels)
                    total_loss = class_loss + loss
                    wandb.log({f'{phase}_step_acc': torch.sum(preds == labels) / labels.shape[0], 
                                f'{phase}_step_loss': class_loss,
                                f'{phase}_step_cosine_loss': loss})

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                # statistics
                running_loss += class_loss.item()
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase])

            if phase == 'train':
                wandb.log({'train_loss': epoch_loss, 'train_acc': epoch_acc, 'epoch': epoch})
            else:
                wandb.log({'val_loss': epoch_loss, 'val_acc': epoch_acc})
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    
            # save the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                save_model_checkpoint(save_name, model, optimizer=optimizer)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model

if __name__ == "__main__":
    main()
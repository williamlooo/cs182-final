"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""

import pathlib
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from model import Net
import matplotlib.pyplot as plt
from torch import nn
import os
os.environ['CUDA_VISIBLE_DEVICES']='3,4'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Create a pytorch dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} images'.format(image_count))

    # Create the training data generator
    batch_size = 32
    im_height = 64
    im_width = 64
    num_epochs = 10
    
    #transforms.RandomResizedCrop(60),
    data_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])
    train_set = torchvision.datasets.ImageFolder(data_dir / 'train', data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)

    #validation is formatted differently, we need to write code for this
    train_set, val_set = torch.utils.data.random_split(train_set,[8*(len(train_set)//10), 2*(len(train_set)//10)])
    
    #val_set = torchvision.datasets.ImageFolder(data_dir / 'val', data_transforms)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)
    
    test_dataset = torchvision.datasets.ImageFolder(data_dir / 'test', data_transforms)                                               
    #metrics
    #best_loss = 1e10
    train_losses = []
    val_losses = []

    def validate(val_loader, model, criterion, device):
        model.eval() #switch to eval mode
        running_val_loss = 0
        val_correct,val_total = 0,0
        
        for X, y_true in val_loader:
        
            X = X.to(device)
            y_true = y_true.to(device)

            # Forward pass and record loss
            y_hat = model(X)
            
            loss = criterion(y_hat, y_true) 
            
            running_val_loss += loss.item() * X.size(0)
            _, predicted = y_hat.max(1)
            val_correct += predicted.eq(y_true).sum().item()
            val_total += y_true.size(0)
        val_acc = val_correct / val_total
        val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        model.train() #switch back to train mode
        return model, val_loss, val_acc

    def evaluate(model):
        ROW_IMG = 10
        N_ROWS = 5

        fig = plt.figure()
        for index in range(1, ROW_IMG * N_ROWS + 1):
            plt.subplot(N_ROWS, ROW_IMG, index)
            plt.axis('off')
            plt.imshow(test_dataset.data[index], cmap='gray_r')
            
            with torch.no_grad():
                model.eval()
                output = model(test_dataset[index][0].unsqueeze(0))
                probs = F.softmax(output, dim=1)
                model.train()
                
            title = f'{torch.argmax(probs)} ({torch.max(probs * 100):.0f}%)'
            
            plt.title(title, fontsize=7)
        fig.suptitle('Predictions')
        plt.savefig("test.png")

    # Create a simple model
    model = Net(len(CLASS_NAMES), im_height, im_width).to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(),lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    for e in range(num_epochs):
        running_train_loss = 0
        train_total, train_correct = 0,0
        #every 4 epochs reduce by 10 (e=4, e=8)
        if e % 4 == 0:
            curr_lr = optim.param_groups[0]['lr']
            optim.param_groups[0]['lr']  = curr_lr / 10
        
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optim.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            loss.backward()
            optim.step()

            running_train_loss += loss.item() * inputs.size(0)
            train_losses.append(loss.item()*inputs.size(0))

            _, predicted = outputs.max(1)
            
            train_total += targets.size(0)
            
            train_correct += predicted.eq(targets).sum().item()
            
            print("\r", end='')
            print(f'training {100 * idx / len(train_loader):.2f}%: {train_correct / train_total:.3f}', end='')

        #calculate averaged training loss
        train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        #calculate averaged validation loss, and accuracy
        with torch.no_grad():
            model, val_loss, val_acc = validate(val_loader, model, criterion, device)
        train_acc = train_correct / train_total
        print("\r", end='')
        print(f'Epoch: {e+1}\t'
                f'Train loss: {train_loss:.4f}\t'
                f'Valid loss: {val_loss:.4f}\t'
                f'Train accuracy: {train_acc:.2f}\t'
                f'Valid accuracy: {val_acc:.2f}')
        torch.save({
            'net': model.state_dict(),
        }, f'weights/latest_{e}.pt')
    
    #plot training results
    plt.plot(num_epochs, train_losses, 'g', label='Training loss')
    plt.plot(num_epochs, val_losses, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plot.png')

    evaluate(model)

if __name__ == '__main__':
    main()

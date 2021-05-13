"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""
#typical stuff
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pickle

#nlp stuff
import spacy 
#!pip install -U spacy[cuda102]

#torch stuff
import torch
import torch.nn.functional as F
from torch import nn
import torchvision
import torchvision.transforms as transforms

#our stuff
import evaluation_tools
from model import ResNetUNet, Net
from custom_augmentations import RandomGaussianBlur, RandomSaltAndPepperNoise

#GPU setup
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    #regular train: train loss: 0.9244, val loss: 1.7809, train acc: 69.1650, val acc: 51.2300 on 20 epochs
    #w/ces: train loss: 0.6905, val loss: 1.9986, train acc: 77.2250, val acc: 50.9800 on 12 epochs
    #set params
    batch_size = 128
    im_height = 64
    im_width = 64
    num_epochs = 20
    INIT_LR = 1e-4
    USE_CES_LOSS = True
    CHECKPOINTS_DIR = "./weights/"
    if not os.path.isdir(CHECKPOINTS_DIR):
        os.mkdir(CHECKPOINTS_DIR)
    PLOT_PATH = "ces_plot.png"

    # Create a pytorch dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    print('Discovered {} train images'.format(image_count))
    print('Training on {} classes'.format(len(CLASS_NAMES)))

    # Create the training data generator
    train_data_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        RandomSaltAndPepperNoise(rate=0.1,noiseType="RGB"),
        RandomGaussianBlur(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])

    val_data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])

    #train dataloader
    train_set = torchvision.datasets.ImageFolder(data_dir / 'train', train_data_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=4, pin_memory=True)
    print(f"train set size: {len(train_set)}")

    #index to class id map
    index_to_class_dict = {v: k for k, v in train_loader.dataset.class_to_idx.items()}
    with open('index_to_class_dict.p', 'wb') as f:
        pickle.dump(index_to_class_dict, f)

    #val dataloader
    val_set = torchvision.datasets.ImageFolder(data_dir / 'val-fixed', val_data_transforms)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=200,
                                               shuffle=True, num_workers=4, pin_memory=True)
    print(f"val set size: {len(val_set)}")
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    #load word dict
    class_id_to_word_dict = {}
    file = open('./data/tiny-imagenet-200/words.txt', 'r')
    for line in file.readlines():
        parts = line.strip().split('\t')
        assert len(parts) == 2
        class_id_to_word_dict[parts[0]] = parts[1]
    print("word dict loaded.")

    #load nlp component
    if USE_CES_LOSS:
        print("using CES loss.")
        spacy.prefer_gpu() #replace with prefer_gpu() if u don't have a gpu
        nlp = spacy.load('en_core_web_md')
        #!python -m spacy download en_core_web_md
        print("nlp loaded.")

    def validate(val_loader, model, criterion, device):
        model.eval() #switch to eval mode
        running_val_loss = 0
        val_correct,val_total = 0,0
        
        for X, y_true in val_loader:
        
            X = X.to(device)
            y_true = y_true.to(device)

            # Forward pass and record loss
            y_hat = model(X)
            loss = criterion(y_hat, y_true, use_nlp=USE_CES_LOSS)

            running_val_loss += loss.item() * X.size(0)
            _, predicted = y_hat.max(1)
            #print("\n")
            #print(predicted)
            val_correct += predicted.eq(y_true).sum().item()
            val_total += y_true.size(0)
        val_acc = val_correct / val_total
        val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        model.train() #switch back to train mode
        return model, val_loss, val_acc

    def condense_label(input_string):
        synonyms = input_string.split(',')
        #grabs default word (assuming everything in front are adjectives), very hacky, not ideal
        picked = synonyms[0].split(' ')[-1].strip() 
        for synonym in synonyms:
            #chooses first word that contains no spaces and is within the nlp vocabulary
            synonym = synonym.strip() #get rid of extra spacing around the word
            if not ' ' in synonym and not nlp(synonym)[0].is_oov:
                picked = synonym
                break
        return picked

    def condense_label_groups(input_strings):
        condensed = [condense_label(label) for label in input_strings]
        return condensed

    def CES_loss_function(outputs, targets, use_nlp=True, CE_weight=0.8, similarity_weight=0.2):
        """
        Cross-Entropy-Similarity loss function incorporating cross entropy loss with label similarity
        """
        CE_loss = nn.CrossEntropyLoss()(outputs, targets)

        label_similarity_loss = 0
        if use_nlp:
            #grab predictions
            probs = F.softmax(outputs, dim=1)
            _, predicted = probs.max(1)
            predicted = predicted.cpu().numpy()
            ground_truth = targets.cpu().numpy()
            #grab labels
            predicted_labels = [class_id_to_word_dict[index_to_class_dict[index]] for index in predicted]
            ground_truth_labels = [class_id_to_word_dict[index_to_class_dict[index]] for index in ground_truth]
            #condense labels for nlp
            condensed_predicted_labels = condense_label_groups(predicted_labels)
            condensed_ground_truth_labels = condense_label_groups(ground_truth_labels)

            #take averaged similarity score among all words, np.clip is used because of floating point imprecisions
            similarities = [np.clip(nlp(condensed_predicted_labels[i])[0].similarity(nlp(condensed_ground_truth_labels[i])[0]),0,1) for i in range(len(condensed_ground_truth_labels))]
            mean_similarity_score = np.mean(np.array(similarities).astype(np.float32))

            label_similarity_loss = 1-mean_similarity_score
        
        loss = (CE_weight*CE_loss) + (similarity_weight*label_similarity_loss)
        return loss

    #plot training results
    def plot_training(train_losses, val_losses, train_accuracies, val_accuracies):
        #plot training and validation losses
        plt.figure()
        plt.plot(range(len(train_losses)), train_losses, 'g', label='Training loss')
        plt.plot(range(len(val_losses)), val_losses, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("loss_"+PLOT_PATH)
        plt.close()
        
        #plot training and validation accuracies
        plt.figure()
        plt.plot(range(len(train_accuracies)), train_accuracies, 'g', label='Training acc')
        plt.plot(range(len(val_accuracies)), val_accuracies, 'b', label='validation acc')
        plt.title('Training and Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.savefig("acc_"+PLOT_PATH)
        plt.close()

    # Create the model
    #model = Net(len(CLASS_NAMES), im_height, im_width).to(device)
    model = ResNetUNet(len(CLASS_NAMES)).to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(),lr=INIT_LR, weight_decay=1e-5)
    criterion = CES_loss_function

    #PRELOAD WEIGHTS
    path = "./weights/best/ces_weights.pt"
    model.load_state_dict(torch.load(path), strict=True)

    #MAIN TRAINING LOOP
    for e in range(num_epochs):
        running_train_loss = 0
        train_total, train_correct = 0,0
        #every 4 epochs reduce by 10 (e=4, e=8)
        if e > 5 and e % 4 == 0:
            curr_lr = optim.param_groups[0]['lr']
            optim.param_groups[0]['lr']  = curr_lr / 10
        
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optim.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,targets, use_nlp=USE_CES_LOSS)
            
            loss.backward()
            optim.step()

            running_train_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            if e == 9 or e == 19:
                print(f"\npredicted label: {predicted}")
                print(f"\ntarget labels: {targets}")
            
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
                f'Train accuracy: {100*train_acc:.4f}\t'
                f'Valid accuracy: {100*val_acc:.4f}')
        train_accuracies.append(100*train_acc)
        val_accuracies.append(100*val_acc)
        torch.save(model.state_dict(), f'{CHECKPOINTS_DIR}/latest_{e}.pt')
        plot_training(train_losses, val_losses, train_accuracies, val_accuracies)

    #evaluation on the test set
    evaluation_tools.evaluate_model(f'{CHECKPOINTS_DIR}/latest_{e}.pt', CLASS_NAMES, index_to_class_dict, im_height, im_width)
    exit(0)
if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
 
import torch
import torch.nn.functional as F
import torchvision
import skimage
from skimage import io
import os
from model import ResNetUNet, Net
import pathlib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pickle
import matplotlib.pyplot as plt

class EvalDataset(Dataset):
    def __init__(self, transform=None,data_dir="./data/tiny-imagenet-200/test/images/"):
        imgs = []
        for filename in os.listdir(data_dir):
            imgs.append({"image":filename})
        self.data_dir = data_dir
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]["image"]
        print(img_path)
        img = io.imread(self.data_dir + img_path)
        img = skimage.color.gray2rgb(img)
        if self.transform is not None:
            img = self.transform(img)
        return {"img_path":self.data_dir + img_path, "img":img}

    def __len__(self):
        return len(self.imgs)

def evaluate_model(path, class_names, index_to_class_dict , im_height, im_width):
    """
    evaluates the model given by path, and plots out results
    """

    test_data_path = "/home/wloo/cs182-final/data/tiny-imagenet-200/test/images/"

    #load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model = Net(len(class_names), im_height, im_width)#.to(device)
    model = ResNetUNet(len(class_names))
    model.load_state_dict(torch.load(path), strict=True)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])
    test_dataset = EvalDataset(transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1,shuffle=False, num_workers=4)

    #load word dict
    class_id_to_word_dict = {}
    file = open('./data/tiny-imagenet-200/words.txt', 'r')
    for line in file.readlines():
        parts = line.strip().split('\t')
        assert len(parts) == 2
        class_id_to_word_dict[parts[0]] = parts[1]
    #plots the number of the ten worst classes along with # of misclassifications
    plot_ten_worst_classes(model, index_to_class_dict, class_id_to_word_dict)

    #run data
    fig = plt.figure(figsize=(16, 12), dpi=80)
    
    for index, sample_batched in enumerate(test_dataloader):
        input = sample_batched["img"]
        ROW_IMG = 10
        N_ROWS = 5
        if index+1 < ROW_IMG * N_ROWS + 1:
            plt.subplot(N_ROWS, ROW_IMG, index+1)
            plt.axis('off')

            img_path = sample_batched["img_path"][0]
            img = io.imread(img_path)
            img = skimage.color.gray2rgb(img)
            plt.imshow(img, cmap='gray_r')

            with torch.no_grad():
                output = model(input)
                probs = F.softmax(output, dim=1)
                #print(probs)
                _, predicted = probs.max(1)
                #print(predicted)
                idx = predicted.numpy()[0]
                label = index_to_class_dict[idx]
                translated_label = class_id_to_word_dict[label] if label in class_id_to_word_dict else "UNKNOWN"
                print(translated_label)

                title = f'{translated_label[:10]} ({torch.max(probs * 100):.0f}%)'

            plt.title(title, fontsize=7)
        else:
            break
    fig.suptitle('Predictions')
    plt.savefig("results.png") 

def plot_ten_worst_classes(model, index_to_class_dict, class_id_to_word_dict):
    from collections import defaultdict
    correctness_map = defaultdict(lambda: 0)
    val_data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), tuple(np.sqrt((255, 255, 255)))),
    ])
    val_set = torchvision.datasets.ImageFolder(data_dir / 'val-fixed', val_data_transforms)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=200,
                                            shuffle=True, num_workers=4, pin_memory=True)
    model.eval() #switch to eval mode
    
    for X, y_true in val_loader:
        # Forward pass and record loss
        y_hat = model(X)
        probs = F.softmax(y_hat, dim=1)
        _, predicted = probs.max(1)
        correctness = predicted.eq(y_true)
        for i in range(len(predicted)):
            if not correctness.numpy()[i]:
                label = index_to_class_dict[y_true.numpy()[i]]
                translated_label = class_id_to_word_dict[label] if label in class_id_to_word_dict else "UNKNOWN"
                correctness_map[translated_label[:10]]+=1 

    worst_ten_classes = sorted(correctness_map.items(), key=lambda t: t[1], reverse=True)[:10]
    print(worst_ten_classes)
    worst_ten_class_names = [a[0] for a in worst_ten_classes]
    worst_ten_class_values = [a[1] for a in worst_ten_classes]
    plt.figure(figsize=(16, 12), dpi=80)
    plt.bar(worst_ten_class_names, worst_ten_class_values)
    plt.title('10 Most Difficult Classes')
    plt.xlabel('Class names')
    plt.ylabel('Misclassifications')
    plt.savefig("hist.png")
    plt.close()

if __name__ == "__main__":
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])

    #this is generated by training dataloder..
    with open('index_to_class_dict.p', 'rb') as f:
        index_to_class_dict = pickle.load(f)

    evaluate_model("./weights/ces_loss/latest_11.pt", CLASS_NAMES, index_to_class_dict, 64,64)
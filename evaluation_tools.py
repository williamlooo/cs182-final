import torch
import torch.nn.functional as F
import torchvision
import skimage
from skimage import io
import os
from model import Net
import pathlib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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

def evaluate_model(path, class_names, im_height, im_width):
    """
    evaluates the model given by path, and plots out results
    """

    test_data_path = "/home/wloo/cs182-final/data/tiny-imagenet-200/test/images/"

    #load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net(len(class_names), im_height, im_width)#.to(device)
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = EvalDataset(transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1,shuffle=True, num_workers=4)

    #load word dict
    id_to_word_dict = {}
    file = open('./data/tiny-imagenet-200/words.txt', 'r')
    for line in file.readlines():
        parts = line.strip().split('\t')
        assert len(parts) == 2
        id_to_word_dict[parts[0]] = parts[1]

    #run data
    fig = plt.figure(figsize=(16, 12), dpi=80)
    
    for index, sample_batched in enumerate(test_dataloader):
        print(sample_batched["img"].shape)
        input = sample_batched["img"]
        ROW_IMG = 10
        N_ROWS = 5
        if index+1 < ROW_IMG * N_ROWS + 1:
            plt.subplot(N_ROWS, ROW_IMG, index+1)
            plt.axis('off')
            
            img_path = sample_batched["img_path"][0]
            print(img_path)
            img = io.imread(img_path)
            img = skimage.color.gray2rgb(img)
            plt.imshow(img, cmap='gray_r')

            with torch.no_grad():
                output = model(input)
                probs = F.softmax(output, dim=1)
                _, predicted = probs.max(1)
                print(torch.argmax(probs))

                print(predicted)
                idx = predicted.numpy()
                label = CLASS_NAMES[idx][0]
                translated_label = id_to_word_dict[label] if label in id_to_word_dict else "UNKNOWN"
                print(translated_label)

                title = f'{translated_label[:10]} ({torch.max(probs * 100):.0f}%)'

            plt.title(title, fontsize=7)
        else:
            break
    fig.suptitle('Predictions')
    plt.savefig("results.png") 
        


if __name__ == "__main__":
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    evaluate_model("weights/latest_20.pt", CLASS_NAMES, 64,64)
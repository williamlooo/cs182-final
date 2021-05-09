import torch
import torchvision
import skimage
from skimage import io
import os
from model import Net
import pathlib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
        if img.shape != (torch.Size([1,3,64,64])):
            print("convert to rgb")
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
    test_dataloader = DataLoader(test_dataset, batch_size=1,shuffle=True, num_workers=0)

    #load word dict
    id_to_word_dict = {}
    file = open('./data/tiny-imagenet-200/words.txt', 'r')
    for line in file.readlines():
        parts = line.strip().split('\t')
        assert len(parts) == 2
        id_to_word_dict[parts[0]] = parts[1]

    #run data
    for i_batch, sample_batched in enumerate(test_dataloader):
        print(sample_batched["img"].shape)
        input = sample_batched["img"]
        output = model(input)
        _, predicted = output.max(1)
        print(predicted)
        idx = predicted.numpy()
        label = CLASS_NAMES[idx][0]
        translated = id_to_word_dict[label] if label in id_to_word_dict else "UNKNOWN"
        print(translated)
        print("done")

        img_path = sample_batched["img_path"][0]
        print(img_path)
        img = io.imread(img_path)
        if img.shape != (torch.Size([1,3,64,64])):
            img = skimage.color.gray2rgb(img)

        io.imsave(f"./test_results/{translated}.JPEG",img)


if __name__ == "__main__":
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    evaluate_model("weights/latest_20.pt", CLASS_NAMES, 64,64)
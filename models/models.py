import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as f
import torchvision.transforms as transforms
import clip
from PIL import Image

from utils.model_io import *

def get_clip_features(img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    return image_features

clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        
        self.resnet18 = models.resnet18(pretrained=True)
        modules = list(self.resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*modules).to(DEVICE)
        self.reshape_layer = nn.Linear(512, latent_dim)

        for p in self.resnet18.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.resnet18(x)
        x = torch.squeeze(x)
        x = f.relu(x)
        x = self.reshape_layer(x)
        return x

class Classifier(nn.Module):
    def __init__(self, in_dim, latent_dim, num_classes):
        super(Classifier, self).__init__()

        self.l1 = nn.Linear(in_dim, latent_dim)
        self.l2 = nn.Linear(latent_dim, latent_dim)
        self.l3 = nn.Linear(latent_dim, latent_dim)
        self.l4 = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = f.relu(self.l1(x))
        x = f.relu(self.l2(x))
        x = f.relu(self.l3(x))
        x = self.l4(x)
        return x, 0

class ClipStudent(nn.Module):
    def __init__(self, latent_dim, num_classes, device=None):
        super(ClipStudent, self).__init__()

        if not device:
            device = DEVICE

        self.encoder = Encoder(latent_dim)
        self.classifier = Classifier(latent_dim, 256, num_classes)

        self.feature_loss = nn.CosineSimilarity(dim=1, eps=1e-08)

    def clip_features(self, x):
        transform = transforms.ToPILImage()
        image_features = []

        with torch.no_grad():
            for i in range(x.shape[0]):
                t_img = preprocess(transform(x[i, ...].squeeze())).unsqueeze(0)
                image_features.append(clip_model.encode_image(t_img))

        image_features = torch.cat(image_features)
        return image_features

    def forward(self, x):
        features = self.encoder(x)
        clip_features = self.clip_features(x)

        loss = self.feature_loss(features, clip_features)
        loss = torch.sum(torch.abs(loss))
        
        preds, _ = self.classifier(features)
        return preds, loss

class ClipOnly(nn.Module):
    def __init__(self, num_classes):
        super(ClipOnly, self).__init__()
        self.classifier = Classifier(512, 256, num_classes)
    
    def forward(self, x):
        transform = transforms.ToPILImage()
        image_features = []

        with torch.no_grad():
            for i in range(x.shape[0]):
                t_img = preprocess(transform(x[i, ...].squeeze())).unsqueeze(0)
                image_features.append(clip_model.encode_image(t_img))

        image_features = torch.cat(image_features)
        image_features = image_features.detach().float()
        return self.classifier(image_features)

class ClipFeatureStudent(nn.Module):
    def __init__(self, latent_dim, device=None):
        super(ClipFeatureStudent, self).__init__()

        if not device:
            device = DEVICE

        self.encoder = Encoder(latent_dim)
        self.feature_loss = nn.CosineSimilarity(dim=1, eps=1e-08)

    def clip_features(self, x):
        transform = transforms.ToPILImage()
        image_features = []

        with torch.no_grad():
            for i in range(x.shape[0]):
                t_img = preprocess(transform(x[i, ...].squeeze())).unsqueeze(0)
                image_features.append(clip_model.encode_image(t_img))

        image_features = torch.cat(image_features)
        return image_features

    def forward(self, x):
        features = self.encoder(x)
        clip_features = self.clip_features(x)

        loss = self.feature_loss(features, clip_features)
        loss = torch.sum(torch.abs(loss))
        
        return None, loss
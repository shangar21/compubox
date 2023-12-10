import sys
import path
import os
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from .models.hitnet import HitNet
import json
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, transform):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.transform(self.imgs[idx]), 0

def predict(X, model_path="./hitnet_model.pth"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((647, 916), antialias=True),
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB if your image is grayscale
    ])
    net = HitNet()
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    net.to(device)

    dataset = ImageDataset(X, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)

    hits = []

    with torch.no_grad():
        for x, _ in tqdm(data_loader):
            output = net(x.to(device))
            output = torch.sigmoid(output)
            output = torch.round(output)
            for i in output:
                hits.append(i)

    return hits


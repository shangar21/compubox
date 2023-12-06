import sys
import path
import os
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from .models import actionnet
import json
import torch

PUNCHES = ['1', '2', '3']
DEFENSE = ['slip', 'roll', 'block']

def pad_and_reformat(X):
    max_len = max([len(i) for i in X])
    for i in range(len(X)):
        X[i] = torch.cat((X[i], torch.zeros(max_len - len(X[i])).to(X[i].device)))
    return X

def pad_sequence(X, max_clip_len, input_size):
    for i in range(len(X)):
        if len(X[i]) < max_clip_len:
            X[i] = torch.cat((X[i], torch.zeros(max_clip_len - len(X[i]), input_size).to(X[i].device)))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels, data):
        self.labels = labels
        self.list_IDs = list_IDs
        self.data = data

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = pad_and_reformat(self.data[ID])
        return X, ID

def predict(X, info_path="./clip_len.json", model_path="./actionnet_model.pth"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    clip_info = json.load(open(info_path, 'r'))
    dimension = clip_info['max_clip_len']
    input_size = len(X[0])
    X = [X]
    pad_sequence(X, dimension, input_size)
    dataset = Dataset(list(range(len(X))), [[0, 0, 0]], X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    net = actionnet.ActionNet(hidden_size=dimension, input_size=input_size, num_actions=3)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    net.to(device)
    punches = []
    with torch.no_grad():
        for x, _ in loader:
            output = net(x.to(device))
            print(output)
            punches.append(PUNCHES[torch.argmax(output)])
    return punches

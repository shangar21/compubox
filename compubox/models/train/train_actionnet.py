import sys
import path
import os
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent.parent.parent)
from actionnet import ActionNet
from pose_estimation import get_model, predict, flatten_results
import argparse
from PIL import Image
import torchvision.transforms as transforms
import torch.utils
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import utils


#PUNCHES = ['1', '1b', '2', '2b', '3', '3b', '4', '4b', '5', '5b', '6', '6b', '7']
PUNCHES = ['1', '2', '3']
DEFENSE = ['slip', 'roll', 'block']
POSE_ESTIMATOR = get_model('yolov8m-pose.pt')

def gen_dataset(path, info_path='./clip_len.json'):
    X = []
    y = []
    max_clip_len = 0
    min_clip_len = float('inf')
    for file in tqdm(os.listdir(path)):
        opp1 = file.split('_')[0]
        opp1_punch = PUNCHES.index(opp1)
        result = predict(POSE_ESTIMATOR, f"{path}/{file}")
        flat_results = flatten_results(result)[1]
        max_clip_len = max(max_clip_len, len(flat_results))
        min_clip_len = min(min_clip_len, len(flat_results))
        X.append(flat_results)
        t = [0]*len(PUNCHES)
        t[opp1_punch] = 1
        y.append(t)
    json.dump(
        {"max_clip_len": max_clip_len, 'min_clip_len': min_clip_len},
        open(info_path, 'w+')
    )
    return X, y, info_path

def gen_dataset_from_json(path, info_path='./clip_len.json'):
    X = []
    y = []
    max_clip_len = 0
    min_clip_len = float('inf')
    data = json.load(open(path, 'r'))
    for i in data:
        X.append(torch.tensor(data[i]['poses']))
        t = [0]*len(PUNCHES)
        t[PUNCHES.index(data[i]['punch'])] = 1
        y.append(t)
        max_clip_len = max(max_clip_len, len(data[i]['poses']))
        min_clip_len = min(min_clip_len, len(data[i]['poses']))
    json.dump(
        {"max_clip_len": max_clip_len, 'min_clip_len': min_clip_len},
        open(info_path, 'w+')
    )
    return X, y, info_path

def pad_and_reformat(X):
    max_len = max([len(i) for i in X])
    for i in range(len(X)):
        X[i] = torch.cat((X[i], torch.zeros(max_len - len(X[i]))))
    return X

def pad_sequence(X, max_clip_len, input_size):
    for i in range(len(X)):
        if len(X[i]) < max_clip_len:
            X[i] = torch.cat((torch.zeros(max_clip_len - len(X[i]), input_size), X[i]))

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="hitnet train cli", description="CLI for training hitnet")
    parser.add_argument('--dataset-path', '-d', nargs="?", default="./punch_imgs", type=str)
    parser.add_argument('--hit-keyword', nargs="?", default="land", type=str)
    parser.add_argument('--miss-keyword', nargs="?", default="miss", type=str)
    parser.add_argument('--train-split', nargs="?", default=0.8, type=float)
    parser.add_argument('--learning_rate', nargs="?", default=0.0001, type=float)
    parser.add_argument('--momentum', nargs="?", default=0.9, type=float)
    parser.add_argument('--epochs', nargs="?", default=100, type=int)
    parser.add_argument('--batch_size', nargs="?", default=5, type=int)
    parser.add_argument('--loss-every', nargs="?", default=1, type=int)
    parser.add_argument('--output', '-o', nargs="?", default="./actionnet_model.pth", type=str)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Generating dataset from path...")
    X, y, info_path = gen_dataset_from_json(args.dataset_path)
    clip_info = json.load(open(info_path, 'r'))
    dimension = clip_info['max_clip_len']
    input_size = len(X[0][0])

    pad_sequence(X, dimension, input_size)
    dataset = Dataset(list(range(len(X))), y, X)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, drop_last=True)

    net = ActionNet(hidden_size=dimension, input_size=input_size, num_actions=len(PUNCHES))
    net.to(device)
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0

    torch.cuda.empty_cache()

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        print(f"Running epoch {epoch + 1}/{args.epochs}")
        running_loss = 0
        for x, label in tqdm(train_loader):
            optimizer.zero_grad()
            output = net(x.to(device))
            t = torch.tensor([y[i] for i in label]).to(torch.float).to(device)
            loss = criterion(output, t)
            loss.backward()
            optimizer.step()
            running_loss += loss
        correct = 0
        total = 0
        for x, label in train_loader:
            output = net(x.to(device))
            t = torch.tensor([y[i] for i in label]).to(torch.float).to(device)
            output = torch.argmax(output)
            correct += 1 if torch.argmax(t) == output else 0
            total += 1
        print(f"Running Loss: {running_loss} \t\t\t Accuracy: {correct/total}")

    torch.cuda.empty_cache()
    torch.save(net.state_dict(), args.output)
#
#    print("Testing model...")
#    accuracy = utils.accuracy(X_test, y_test, net, expected_size, device, verbose=False)
#    print(f"Test accuracy: {accuracy}")


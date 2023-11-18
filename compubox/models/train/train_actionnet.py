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

PUNCHES = ['1', '1b', '2', '2b', '3', '3b', '4', '4b', '5', '5b', '6', '6b', 'overhand']
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

def pad_and_reformat(X):
    max_len = max([len(i) for i in X])
    for i in range(len(X)):
        X[i] = torch.cat((X[i], torch.zeros(max_len - len(X[i]))))
    return torch.cat(X).reshape(-1, max_len)

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
    parser.add_argument('--output', '-o', nargs="?", default="./model.pth", type=str)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Generating dataset from path...")
    X, y, info_path = gen_dataset(args.dataset_path)
#    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=args.train_split, random_state=42)
    clip_info = json.load(open(info_path, 'r'))
    dimension = clip_info['max_clip_len']
    input_size = len(X[0][0])

    X_train = X
    y_train = y

    net = ActionNet(dimension=dimension, input_size=input_size, num_actions=len(PUNCHES), num_layers=13)
    net.to(device)
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0

    torch.cuda.empty_cache()

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)#, momentum=args.momentum)

    for epoch in range(args.epochs):
        print(f"Running epoch {epoch + 1}/{args.epochs}")
        for i in tqdm(range(len(X_train))):
            x = pad_and_reformat(X_train[i]) #torch.cat(X_train[i]).reshape(-1, max(len(i) for i in X_train))
            optimizer.zero_grad()
            output = net(x.to(device))
            t = torch.tensor(y_train[i]).reshape(output.shape).to(torch.float).to(device)
            loss = criterion(output, t)
            loss.backward()
            optimizer.step()
            running_loss += loss
        correct = 0
        total = 0
        for i in tqdm(range(len(X_train))):
            x = pad_and_reformat(X_train[i]) #torch.cat(X_train[i]).reshape(-1, max(len(i) for i in X_train))
            output = net(x.to(device))
            t = torch.tensor(y_train[i]).reshape(output.shape).to(torch.float).to(device)
            output = torch.argmax(output)
            correct += 1 if torch.argmax(t) == output else 0
            total += 1
        print("Accuracy: ", correct/total)
#
#    torch.cuda.empty_cache()
#    torch.save(net.state_dict(), args.output)
#
#    print("Testing model...")
#    accuracy = utils.accuracy(X_test, y_test, net, expected_size, device, verbose=False)
#    print(f"Test accuracy: {accuracy}")


import sys
import path
import os

directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

from hitnet import HitNet

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

def gen_dataset(path, hit, miss):
    X = []
    y = []
    transformer = transforms.ToTensor()
    for file in tqdm(os.listdir(path)):
        image = Image.open(f"{path}/{file}")
        image = image.convert('L')
        t = 1 if hit in file else 0
        tensor = transformer(image)
        X.append(tensor)
        y.append(t)
    return X, y

def get_max_img_size(imgs):
    max_size  = [0, 0]
    for i in imgs:
        max_size[0] = max(i.shape[1], max_size[0])
        max_size[1] = max(i.shape[2], max_size[1])
    json.dump({'img_dim': max_size}, open("img_dim.json", "w+"))
    return max_size

def resize_with_padding(img, expected_size):
    d_w = expected_size[0] - img.shape[1]
    d_h = expected_size[1] - img.shape[2]
    pad_w = d_w // 2
    pad_h = d_h // 2
    padding = (pad_h, d_h - pad_h, pad_w, d_w - pad_w)
    return torch.nn.functional.pad(img, padding)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="hitnet train cli", description="CLI for training hitnet")
    parser.add_argument('--dataset-path', '-d', nargs="?", default="./punch_imgs", type=str)
    parser.add_argument('--hit-keyword', nargs="?", default="landed", type=str)
    parser.add_argument('--miss-keyword', nargs="?", default="miss", type=str)
    parser.add_argument('--train-split', nargs="?", default=0.8, type=float)
    parser.add_argument('--learning_rate', nargs="?", default=0.001, type=float)
    parser.add_argument('--momentum', nargs="?", default=0.9, type=float)
    parser.add_argument('--epochs', nargs="?", default=100, type=int)
    parser.add_argument('--batch_size', nargs="?", default=5, type=int)
    args = parser.parse_args()

    print("Generating dataset from path...")
    X, y = gen_dataset(args.dataset_path, args.hit_keyword, args.miss_keyword)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=args.train_split)
    expected_size = get_max_img_size(X)

    net = HitNet(img_size=expected_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)

    for epoch in range(args.epochs):
        running_loss = 0.0
        print(f"Running epoch {epoch + 1}/{args.epochs}")
        for i in tqdm(range(len(X_train))):
            img = X_train[i]
            t = torch.tensor([y_train[i]])
            img = resize_with_padding(img, expected_size)
            output = net(img)
            loss = criterion(output, t)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if epoch % 20 == 19:
                print(running_loss)
                running_loss = 0.0




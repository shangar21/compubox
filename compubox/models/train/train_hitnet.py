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

import utils

def gen_dataset(path, hit, miss):
    X = []
    y = []
    transformer = transforms.ToTensor()
    for file in tqdm(os.listdir(path)):
        image = Image.open(f"{path}/{file}")
        image = image.convert('L')
        #t = [0, 0]
        #t[0 if hit in file else 1] = 1
        t = 1 if hit in file else 0
        tensor = transformer(image)
        X.append(tensor)
        y.append(t)
    return X, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="hitnet train cli", description="CLI for training hitnet")
    parser.add_argument('--dataset-path', '-d', nargs="?", default="./punch_imgs", type=str)
    parser.add_argument('--hit-keyword', nargs="?", default="land", type=str)
    parser.add_argument('--miss-keyword', nargs="?", default="miss", type=str)
    parser.add_argument('--train-split', nargs="?", default=0.8, type=float)
    parser.add_argument('--learning_rate', nargs="?", default=0.001, type=float)
    parser.add_argument('--momentum', nargs="?", default=0.9, type=float)
    parser.add_argument('--epochs', nargs="?", default=100, type=int)
    parser.add_argument('--batch_size', nargs="?", default=5, type=int)
    parser.add_argument('--loss-every', nargs="?", default=1, type=int)
    parser.add_argument('--output', '-o', nargs="?", default="Downloads/model.pth", type=str)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Generating dataset from path...")
    X, y = gen_dataset(args.dataset_path, args.hit_keyword, args.miss_keyword)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=args.train_split, random_state=42)
    expected_size = utils.get_max_img_size(X)
    print(expected_size)

    net = HitNet()
    net.to(device)
    criterion = nn.BCEWithLogitsLoss()
    # Define the transformation
    transform = transforms.Compose([
    transforms.Resize((647, 916)),
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB if your image is grayscale
    transforms.ToTensor(),
    ])


    running_loss = 0.0

    torch.cuda.empty_cache()

    for i in tqdm(range(len(X_train))):
        img = X_train[i]
        img = utils.resize_with_padding(img, expected_size)
        img = transforms.ToPILImage()(img)
        img = transform(img)
        img = img.unsqueeze(0)
        X_train[i] = img

    for epoch in range(args.epochs):
        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)#, momentum=args.momentum)
        print(f"Running epoch {epoch + 1}/{args.epochs}")
        for i in tqdm(range(len(X_train))):
            img = X_train[i]
            optimizer.zero_grad()
            output = net(img.to(device))
            t = torch.tensor(y_train[i]).reshape(output.shape).to(torch.float).to(device)
            loss = criterion(output, t)
            loss.backward()
            optimizer.step()
            running_loss += loss
        if epoch % args.loss_every == args.loss_every - 1:
            print(f"Loss: {running_loss} \t\t Training accuracy: {utils.accuracy(X_train, y_train, net, expected_size, device, verbose=False)}")
            running_loss = 0

    torch.cuda.empty_cache()
    torch.save(net.state_dict(), args.output)

    print("Testing model...")
    accuracy = utils.accuracy(X_test, y_test, net, expected_size, device, verbose=False)
    print(f"Test accuracy: {accuracy}")


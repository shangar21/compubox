import json
import torch.nn.functional
from tqdm import tqdm

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


def accuracy(X_test, y_test, net, expected_size, device, threshold = lambda x : 1 if x > 0.5 else 0, verbose=True):
    correct = 0
    for i in tqdm(range(len(X_test)), disable=not verbose):
        img = X_test[i]
        img = resize_with_padding(img, expected_size)
        output = net(img.to(device))
        output = threshold(output)
        correct += 1 if output == y_test[i] else 0
    if verbose:
        print(correct/len(X_test))
    return correct / len(X_test)


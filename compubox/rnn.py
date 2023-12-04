from models import actionnet
from models.train.train_actionnet import pad_sequence, Dataset, PUNCHES, pad_and_reformat
import json
import torch

def predict(X, info_path="./clip_len.json", model_path="./actionnet_model.pth"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    clip_info = json.load(open(info_path, 'r'))
    dimension = clip_info['max_clip_len']
    input_size = len(X[0][0])
    pad_sequence(X, dimension, input_size)
    net = actionnet.ActionNet(hidden_size=dimension, input_size=input_size, num_actions=3)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    net.to(device)
    punches = []
    for x in X:
        x = pad_and_reformat(x)
        output = net(x.to(device))
        punches.append(PUNCHES[torch.argmax(output)])
    return punches

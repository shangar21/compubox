from models.hitnet import Hitnet
import json
import torch
import torchvision.transforms as transforms

def predict(X, model_path="./hitnet_model.pth"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    transform = transforms.Compose([
        transforms.Resize((647, 916)),
        transforms.Grayscale(num_output_channels=3),  # Convert to RGB if your image is grayscale
        transforms.ToTensor(),
    ])
    net = HitNet()
    net.load_state_dict(torch.load(model_path))
    net.eval()
    net.to(device)
    hits = []

    for i in tqdm(range(len(X))):
            img = X[i]
            img = utils.resize_with_padding(img, expected_size)
            img = transforms.ToPILImage()(img)
            img = transform(img)
            img = img.unsqueeze(0)
            X[i] = img

    for i in tqdm(range(len(X))):
        img = X[i]
        output = net(img.to(device))
        hits.append(0 if output < 0.5 else 1)

    return hits


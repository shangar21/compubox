from ultralytics import YOLO
from models.train import utils
import torch

def get_model(path):
    return YOLO(path)

def predict(model, img):
    results = model.track(img, show=True)
    return results

def flatten_results(results):
    poses = {}
    for result in results:
        ids = result.boxes.id
        for i in range(len(ids)):
            print(i)
            poses[ids[i].item()] = poses.get(ids[i].item(), []) + [torch.flatten(result.keypoints.xyn[i])]
    return poses

if __name__ == '__main__':
    model = get_model('yolov8m-pose.pt')
    results = predict(model, '/home/shangar21/Downloads/roll_vid_1.mp4')
    flat_res = flatten_results(results)
    print(flat_res)


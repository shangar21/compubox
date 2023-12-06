import sys
import path
import os
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
from ultralytics import YOLO
try:
    from .models.train import utils
except:
    from models.train import utils
import torch
import matplotlib.pyplot as plt

def get_model(path):
    return YOLO(path)

def predict(model, img, show=False):
    results = model.track(img, show=show)
    return results

def flatten_results(results):
    poses = {}
    for result in results:
        ids = result.boxes.id
        for i in range(len(ids)):
            poses[ids[i].item()] = poses.get(ids[i].item(), []) + [torch.flatten(result.keypoints.xyn[i])]
    return poses

def get_fist_frames(results):
    frames = {}

    for result in results:
        ids = result.boxes.id
        for i in range(len(ids)):
            fist_keypts = [result.keypoints.xy[i][10], result.keypoints.xy[i][9]]
            for j in fist_keypts:
                row_lims = [
                    max(int(j[1].item() - 100), 0),
                    min(int(j[1].item() + 100), result.orig_img.shape[0])
                ]
                col_lims = [
                    max(int(j[0].item() - 100), 0),
                    min(int(j[0].item() + 100), result.orig_img.shape[1])
                ]
                cropped = result.orig_img[row_lims[0]: row_lims[1], col_lims[0]: col_lims[1]]
                frames[ids[i].item()] = frames.get(ids[i].item(), []) + [cropped]

    return frames

if __name__ == '__main__':
    model = get_model('yolov8m-pose.pt')
    results = predict(model, '/home/shangar21/Downloads/roll_vid_1.mp4')
    flat_res = flatten_results(results)
    print(flat_res)


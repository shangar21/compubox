import os
import argparse
from compubox import rnn, punch_pred, pose_estimation
import torch
import image_slicer
from compubox.models.train import utils
from functools import partial
import concurrent.futures
import json

def get_nxm_slices(n, m, sliced, arr):
    """
    Takes a NumPy array as input and outputs a list of slices of size nxm.

    Parameters:
    - arr: NumPy array
    - n: Number of rows in slices
    - m: Number of columns in slices

    Returns:
    - List of slices
    """
    slices = []
    rows, cols, _ = arr.shape

    if n > rows or m > cols:
        raise ValueError("Slice dimensions exceed array dimensions")

    for i in range(0, rows - n + 1, n):
        for j in range(0, cols - m + 1, m):
            slice_ = arr[i:i+n, j:j+m]
            slices.append(slice_)

    sliced += slices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="compubox-cli", description="CLI for live compubox punch counter")
    parser.add_argument('--path', '-p', nargs="?", default=".", type=str)
    parser.add_argument('--output', '-o', nargs="?", default="result.json", type=str)
    args = parser.parse_args()

    pose_model = pose_estimation.get_model('yolov8m-pose.pt')

    results = {}

    for i in os.listdir(args.path):
        poses = pose_estimation.predict(pose_model, f"{args.path}/{i}")
        flat_poses = pose_estimation.flatten_results(poses)
        fist_frames = pose_estimation.get_fist_frames(poses)
        results[i] = {}
        for k in flat_poses:
            results[i][k] = {}
            X = flat_poses[k]
            X = torch.stack(X)
            punches = rnn.predict(X)
            results[i][k]['punches'] = punches
            if len(flat_poses) > 1:
                hits = punch_pred.predict(fist_frames[k])
                hits = [i.item() for i in hits]
                hit = 1 in hits
                results[i][k]['landed'] = hit

    json.dump(results, open(args.output, 'w+'))



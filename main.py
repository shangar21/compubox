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
        for k in flat_poses:
            X = flat_poses[k]
            X = torch.stack(X)
            punches = rnn.predict(X)

        frames = utils.extract_frames(f"{args.path}/{i}")

        sliced_frames = []
        target = partial(get_nxm_slices, frames[0].shape[0]//5, frames[0].shape[1]//5, sliced_frames)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(target, frames)

        hit = 1 in punch_pred.predict(sliced_frames)

        results[i] = {"punches": punches, "hit": hit}

    json.dump(results, open(args.output, 'w+'))



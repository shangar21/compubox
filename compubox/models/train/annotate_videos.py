import sys
import path
import os
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent.parent.parent)
from pose_estimation import get_model, predict, flatten_results
import json
import argparse
import torch

POSE_ESTIMATOR = get_model('yolov8m-pose.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="video annotate cli", description="CLI for annotating training video data")
    parser.add_argument('--dataset-path', '-d', nargs="?", default="./punch_videos", type=str)
    parser.add_argument('--output-path', '-o', nargs="?", default="./dataset.json", type=str)
    args = parser.parse_args()

    dataset = {}

    for i in os.listdir(args.dataset_path):
        print(f"Entering directory {i}")
        if os.path.isdir(f"{args.dataset_path}/{i}"):
            for j in os.listdir(f"{args.dataset_path}/{i}"):
                print(f"Processing file {j}")
                result = predict(POSE_ESTIMATOR, f"{args.dataset_path}/{i}/{j}")
                flat_results = flatten_results(result)[1]
                dataset[f"{i}_{j}"] = {'punch': i, 'poses': [k.tolist() for k in flat_results]}

    json.dump(dataset, open(args.output_path, 'w+'), indent=4)

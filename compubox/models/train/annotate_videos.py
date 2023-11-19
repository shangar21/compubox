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
        dataset[i] = {}
        result = predict(POSE_ESTIMATOR, f"{args.dataset_path}/{i}", show=True)
        flat_results = flatten_results(result)

        for k in flat_results:
            punch = str(input(f"Enter punch/defensive action for fighter with id {k}: "))
            entry = {'poses': [i.tolist() for i in flat_results[k]], 'punch': punch}
            dataset[i][k] = entry

    json.dump(dataset, open(args.output_path, 'w+'))

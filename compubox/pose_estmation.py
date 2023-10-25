from ultralytics import YOLO
from ultralytics.utils import plotting

model = YOLO('~/Downloads/yolov8n-pose.pt')

def predict(img_path):
    results = model.predict(img_path, show=True)
    return results

def show_result(result):
    kpts = result.keypoints
    annotator = plotting.Annotator(result.orig_img)
    annotator.kpts(kpts.data[0])

if __name__ == '__main__':
    result = predict('/home/shangar21/Downloads/guy_standing.jpg')

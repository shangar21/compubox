import cv2
import os
import shutil

def create_tmp(tmp_path):
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)

def close(tmp_path):
    shutil.rmtree(tmp_path)

def get_photo_num(file_name):
    return int(file_name.split('.')[0])

def get_photo_nums(path):
    return [get_photo_num(i) for i in os.listdir(path)]

def store_photo(frame, path):
    photo_nums = get_photo_nums(path)
    new_num = max(photo_nums) + 1 if photo_nums else 1
    cv2.imwrite(f"{path}/{new_num}.png", frame)

def clean_photo_dir(tmp_path, max_frames):
    photo_nums = get_photo_nums(tmp_path)
    if len(photo_nums) > max_frames:
        delta = max(photo_nums) - max_frames
        for i in range(1, delta + 1):
            os.remove(f"{tmp_path}/{i}.png")
        photo_nums = sorted(get_photo_nums(tmp_path))
        for i in photo_nums:
            n = i - delta
            os.rename(f"{tmp_path}/{i}.png", f"{tmp_path}/{n}.png")

def start_stream(cam_id = 0, tmp_path="./tmp", max_frames=20):
    create_tmp(tmp_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('Input', frame)
        store_photo(frame, tmp_path)
        clean_photo_dir(tmp_path, max_frames)

        c = cv2.waitKey(1)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_stream()

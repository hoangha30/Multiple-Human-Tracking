import os
import numpy as np
import configparser
import shutil
import argparse

mot17_root = '/mnt/HDD/Project/ML2/MOT17'
output_dir = '/mnt/HDD/Project/ML2/yolo'
def convert(img_width, img_height, left, top, width, height):
    """Chuyển đổi tọa độ bounding box sang định dạng YOLO, đảm bảo tọa độ hợp lệ."""

    # Kiểm tra và điều chỉnh tọa độ nếu cần
    left = max(0, min(left, img_width))  # Giới hạn left trong khoảng [0, img_width]
    top = max(0, min(top, img_height))   # Giới hạn top trong khoảng [0, img_height]
    right = max(0, min(left + width, img_width))  # Giới hạn right trong khoảng [0, img_width]
    bottom = max(0, min(top + height, img_height)) # Giới hạn bottom trong khoảng [0, img_height]

    # Tính toán lại chiều rộng và chiều cao
    width = right - left
    height = bottom - top

    # Tính toán tọa độ trung tâm và kích thước tương đối
    x_center = (left + width / 2.0) / img_width
    y_center = (top + height / 2.0) / img_height
    w_norm = width / img_width
    h_norm = height / img_height

    # Kiểm tra xem bounding box có diện tích dương không
    if width <= 0 or height <= 0:
        print(f"WARNING: Bỏ qua bounding box có diện tích không dương: {left}, {top}, {width}, {height}")
        return None  # Trả về None để bỏ qua bounding box không hợp lệ

    return f"{x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

for split in ['train', 'test']:
    split_dir = os.path.join(output_dir, split)
    os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)

    for mot_dir in os.listdir(os.path.join(mot17_root, split)):
        det_path = os.path.join(mot17_root, split, mot_dir, 'det/det.txt')
        dets = np.loadtxt(det_path, delimiter=',')
        ini_path = os.path.join(mot17_root, split, mot_dir, 'seqinfo.ini')

        conf = configparser.ConfigParser()
        conf.read(ini_path)

        try:
            img_width = int(conf['Sequence']['imWidth'])
            img_height = int(conf['Sequence']['imHeight'])
        except KeyError:
            print(f"Lỗi đọc file {ini_path}. Bỏ qua sequence {mot_dir}.")
            continue

        for det in dets:
            frame_id = int(det[0])
            left, top, width, height = det[2], det[3], det[4], det[5]  # Chỉ lấy các giá trị bounding box
            yolo_bbox = convert(img_width, img_height, left, top, width, height)

            image_name = f"{mot_dir}+{frame_id:06d}.jpg"
            label_name = f"{mot_dir}+{frame_id:06d}.txt"

            old_img_path = os.path.join(mot17_root, split, mot_dir, 'img1', f"{frame_id:06d}.jpg")
            new_img_path = os.path.join(split_dir, 'images', image_name)
            label_path = os.path.join(split_dir, 'labels', label_name)

            if not os.path.exists(new_img_path):
                shutil.copyfile(old_img_path, new_img_path)

            if yolo_bbox:  
                with open(label_path, 'a') as f:
                    f.write(f"0 {yolo_bbox}\n")

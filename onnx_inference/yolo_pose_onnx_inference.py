import os
import numpy as np
import cv2
import argparse
import onnxruntime
import yaml
from tqdm import tqdm

from pathlib import Path
from typing import List, Union, Tuple

from utils.plots import plot_skeleton_kpts

# Define a list of supported image file formats
IMAGE_FORMATS: List[str] = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg', '.jpe', '.jp2', '.tiff', '.tif', '.png', '.exr']

_CLASS_COLOR_MAP = [
    (0, 0, 255) , # Person (blue).
    (255, 0, 0) ,  # Bear (red).
    (0, 255, 0) ,  # Tree (lime).
    (255, 0, 255) ,  # Bird (fuchsia).
    (0, 255, 255) ,  # Sky (aqua).
    (255, 255, 0) ,  # Cat (yellow).
]

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
radius = 5

def get_image_files(directory_path: Union[Path, str]) -> List[str]:
    """
    Get a list of image file paths from the specified directory.

    Parameters
    ----------
    directory_path : Union[Path, str]
        The directory path to search for image files.

    Returns
    -------
    List[str]
        A list of image file paths that meet the required criteria.
    """
    image_files: List[str] = []

    for file_path in Path(directory_path).iterdir():
        # Check if the file is a regular file (i.e. not a directory) and has a supported file extension
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_FORMATS:
            # Append the file path to the list of image filenames
            image_files.append(str(file_path))

    return image_files

def read_img(img_file, img_shape):
    h, w, ch = img_shape
    img = cv2.imread(img_file, cv2.IMREAD_COLOR if ch > 1 else cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    return img

def prepare_input(img, img_shape, img_mean=127.5, img_scale=1/127.5):
    h, w, ch = img_shape
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    if ch == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if ch == 1:
        img = np.reshape(img, img_shape)

    img = (img - img_mean) * img_scale
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img, 0)
    img = img.transpose(0, 3, 1, 2)
    return img


def model_inference(model_path=None, input=None):
    #onnx_model = onnx.load(args.model_path)
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: input})
    return output


def model_inference_image_list(model_path: str, data_dict, img_path: str, img_shape: Tuple[int, int, int], mean=None, scale=None, dst_path=None):
    os.makedirs(args.dst_path, exist_ok=True)
    img_file_list = get_image_files(img_path)

    pbar = tqdm(enumerate(img_file_list), total=min(len(img_file_list), 10))
    for img_index, img_file in pbar:
        pbar.set_description("{}/{}".format(img_index, len(img_file_list)))
        img = read_img(img_file, img_shape)
        _input = prepare_input(img, img_shape, mean, scale)
        output = model_inference(model_path, _input)
        dst_file = os.path.join(dst_path, os.path.basename(img_file))
        if output[0].size:
            post_process(img_file, dst_file, output[0][0], data_dict["skeleton"])


def post_process(img_file, dst_file, output, skeleton, score_threshold=0.45):
    """
    Draw bounding boxes on the input image. Dump boxes  in a txt file.
    """
    det_bboxes, det_scores, det_labels, kpts = output[:, 0:4], output[:, 4], output[:, 5], output[:, 6:]
    img = cv2.imread(img_file)
    #To generate color based on det_label, to look into the codebase of Tensorflow object detection api.
    dst_txt_file = dst_file.replace('png', 'txt')
    f = open(dst_txt_file, 'wt')
    for idx in range(len(det_bboxes)):
        det_bbox = det_bboxes[idx]
        kpt = kpts[idx]
        if det_scores[idx]>0:
            f.write("{:8.0f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f}\n".format(det_labels[idx], det_scores[idx], det_bbox[1], det_bbox[0], det_bbox[3], det_bbox[2]))
        if det_scores[idx]>score_threshold:
            color_map = _CLASS_COLOR_MAP[int(det_labels[idx])]

            x, y = (int(det_bbox[0]), int(det_bbox[1]))
            w, h = (int(det_bbox[2]), int(det_bbox[3]))
            img = cv2.rectangle(img, (x, y), (w, h), color_map[::-1], 2)
            cv2.putText(img, "id:{}".format(int(det_labels[idx])), (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[::-1], 2)
            cv2.putText(img, "score:{:2.1f}".format(det_scores[idx]), (x + 5, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[::-1], 2)
            plot_skeleton_kpts(img, img.shape, kpt, 3, skeleton)
    cv2.imwrite(dst_file, img)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str)
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument("-i", "--img-path", type=str, default="./sample_ips.txt")
    parser.add_argument("-d", "--dst-path", type=str, default="./sample_ops_onnxrt")
    parser.add_argument('-s', '--img-shape', nargs=3, type=int, default=[640, 640, 3], help='image shape (h, w, ch)')
    args = parser.parse_args()

    with open(args.data) as f:
        data_dict = yaml.safe_load(f)  # data dict

    model_inference_image_list(
        args.model_path,
        data_dict,
        args.img_path,
        args.img_shape,
        mean=0.0,
        scale=0.00392156862745098,
        dst_path=args.dst_path
    )

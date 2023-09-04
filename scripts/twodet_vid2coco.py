## 2det_vid2coco.py
# Author: Felipe Parodi
# Date: 2023-03-15
# Description: Generate 2-class pseudo-labels from dir of videos

# script to take in dir of images and output COCO json with bbox

'''
Usage: python twodet_vid2coco.py \
--input-dir "...\test_vids" \
--output-dir "...\output" \
--count 2 \
--device "cuda:0"
'''

import argparse
import glob
import json
import os
import time

import cv2
import mmcv
import numpy as np
from mmdet.apis import inference_detector, init_detector

# from mmpose_old.apis import process_mmdet_results

# time script:
start_time = time.time()
def process_mmdet_results(mmdet_results, cat_id=1):
    """Process mmdet results, and return a list of bboxes.

    Args:
        mmdet_results (list|tuple): mmdet results.
        cat_id (int): category id (default: 1 for human)

    Returns:
        person_results (list): a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id - 1]

    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results

def main():
    # arguments for the function
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--input-dir", type=str, help="path to videos")
    parser.add_argument("--output-dir", type=str, help="path to the output directory")
    parser.add_argument("--count", type=int, default=3, help="max number of monkeys in image")
    args = parser.parse_args()
    
    # Load all videos in input directory:
    vid_dir = args.input_dir
    out_dir = args.output_dir
    count = args.count
    device = "cuda:0"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    os.makedirs(out_dir + "/viz", exist_ok=True)
    os.makedirs(out_dir + "/imgs", exist_ok=True)
    os.makedirs(out_dir + "/annotations", exist_ok=True)

    # Set file for output COCO json to annotations dir:
    out_json_file = os.path.join(
        out_dir, "annotations", os.path.basename(out_dir) + "_pseudo.json"
    )
    print(out_json_file)

    # Initialize detector:
    det_config = r"Y:\MacTrack\scripts\fasterrcnn_2classdet_mt.py"
    det_checkpoint = r"Y:\MacTrack\results\mactrack_detection\fasterrcnn2class_best_bbox_mAP_epoch_50.pth"
    det_model = init_detector(det_config, det_checkpoint, device=device)

    # Initialize dict for COCO file:
    categories = [
        {
            "id": 1,
            "name": "monkey",
            "supercategory": "monkey",
        },
        {
            "id": 2,
            "name": "logger",
            "supercategory": "logger",
        },
    ]
    img_anno_dict = {
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    monkey_thr: float = 0.99
    logger_thr: float = 0.7
    ann_uniq_id: int = int(0)

    # is_upside_down = False
# put all videos in a list
    vids = [vid for vid in os.listdir(vid_dir) if vid.endswith(".mp4") or vid.endswith(".avi")]
    print(f'Found {len(vids)} videos in {vid_dir}')
    for vid in vids:
        # if not vid.endswith(".mp4") or not vid.endswith(".avi"):
        #     continue
        # Load video:
        print(vid) 
        video = mmcv.VideoReader(vid_dir + vid)
        for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
            frame_id: str = int(frame_id) + np.random.randint(0, 20000000)
            cur_frame = cv2.flip(cur_frame, -1) # flip horizontally and vertically

            result = inference_detector(det_model, cur_frame)

            # if frame_id % 2 != 0:
            #     is_upside_down = True
            #     cur_frame = cv2.flip(cur_frame, -1) # flip back horizontally and vertically

            if len(result[0]) == 0:
                continue
            if len(result[0]) > count:
                result[0] = result[0][:count]
            # if len(result[1]) > 1:
            #     result[1] = result[1][:1]
            for i in range(len(result[0])):
                if result[0][i][4] < monkey_thr:
                    result[0][i] = [0, 0, 0, 0, 0]

            monkey_result = process_mmdet_results(result, 1)
            logger_result = process_mmdet_results(result, 2)

            logger_present = True

            if len(monkey_result) > count:
                monkey_result = monkey_result[:count]
            elif len(monkey_result) == 0:
                continue
            if not logger_result:
                logger_present = False
            elif len(logger_result) > 1:
                logger_result = logger_result[:1]
            elif logger_result[0]["bbox"][4] < logger_thr:
                continue
            for i, monkey in enumerate(monkey_result):
                if monkey["bbox"][4] < monkey_thr:
                    continue

            annotations_added = False
            frame_id_uniq: int = np.random.randint(0, 10000000)
            # print(result)
            for indx in range(len(result)):
                if indx == 0:
                    x: int = int(monkey_result[i]["bbox"][0])
                    y: int = int(monkey_result[i]["bbox"][1])
                    w: int = int(monkey_result[i]["bbox"][2] - monkey_result[i]["bbox"][0])
                    h: int = int(monkey_result[i]["bbox"][3] - monkey_result[i]["bbox"][1])
                    bbox: list[int] = [x, y, w, h]
                    # if bbox is all 0, skip:

                    # if is_upside_down is True:
                    #     # flip bbox horizontally and vertically:
                    #     bbox = [cur_frame.shape[1] - x - w, cur_frame.shape[0] - y - h, w, h]  
                    # elif is_upside_down is False:
                    #     bbox = [x, y, w, h]  
                    if bbox == [0, 0, 0, 0]:
                        continue
                    area: int = round(w * h, 0)
                    center: list[float] = [x + w / 2, y + h / 2]
                    scale: list[float] = [w / 200, h / 200]
                    category: Literal[1] = 1
                    annotations = {
                        "area": area,
                        "iscrowd": 0,
                        "image_id": frame_id_uniq,
                        "bbox": bbox,
                        "center": center,
                        "scale": scale,
                        "category_id": category,
                        "id": ann_uniq_id,
                    }
                    img_anno_dict["annotations"].append(annotations)
                    ann_uniq_id += 1
                    annotations_added = True
                if indx == 1:
                    if logger_present is False:
                        continue

                    x = int(logger_result[0]["bbox"][0])
                    y = int(logger_result[0]["bbox"][1])
                    w = int(logger_result[0]["bbox"][2] - logger_result[0]["bbox"][0])
                    h = int(logger_result[0]["bbox"][3] - logger_result[0]["bbox"][1])
                    bbox = [x, y, w, h]
                    # if is_upside_down is True:
                    #     # flip bbox horizontally and vertically:
                    #     bbox = [cur_frame.shape[1] - x - w, cur_frame.shape[0] - y - h, w, h]
                    # elif is_upside_down is False:
                        # bbox = [x, y, w, h]   
                    area = round(w * h, 0)
                    center = [x + w / 2, y + h / 2]
                    scale = [w / 200, h / 200]
                    category = 2

                    annotations = {
                        "area": area,
                        "iscrowd": 0,
                        "image_id": frame_id_uniq,
                        "bbox": bbox,
                        "center": center,
                        "scale": scale,
                        "category_id": category,
                        "id": ann_uniq_id,
                    }
                    img_anno_dict["annotations"].append(annotations)
                    ann_uniq_id += 1
                    annotations_added = True
                images = {
                    "file_name": os.path.basename(vid)[:-4]
                    + "_"
                    + str(frame_id_uniq)
                    + ".jpg",
                    "height": video.height,
                    "width": video.width,
                    "id": frame_id_uniq,
                }

            raw_frame = (
                out_dir
                + "/imgs/"
                + os.path.basename(vid)[:-4]
                + "_"
                + str(frame_id_uniq)
                + ".jpg"
            )
            cv2.imwrite(raw_frame, cur_frame)

            det_frame = det_model.show_result(
                cur_frame, result, score_thr=logger_thr, show=False
            )
            viz_frame = (
                out_dir
                + "/viz/"
                + os.path.basename(vid)[:-4]
                + "_"
                + str(frame_id_uniq)
                + "_vis.jpg"
            )
            cv2.imwrite(viz_frame, det_frame)

            if annotations_added:
                img_anno_dict["images"].append(images)

    print("Number of images added to COCO json: ", len(img_anno_dict["images"]))
    print("Number of annotations added to COCO json: ", len(img_anno_dict["annotations"]))
    with open(out_json_file, "w") as outfile:
        json.dump(img_anno_dict, outfile, indent=2)
    print("Time elapsed: ", time.time() - start_time)

if __name__ == "__main__":
    main()

import argparse
import json
import os
import time

import numpy as np
from mmdet.apis import init_detector
from mmpose.apis import init_model
from mmpose.registry import VISUALIZERS
from mmpose.utils import adapt_mmdet_pipeline

from pseudolabel_utils import (process_images_in_directory,
                            process_videos_in_directory)

start_time = time.time()


def main():
    """
    Main function to process images and videos for pose estimation and detection.
    Utilizes command-line arguments to specify paths for input images, videos,
    and the output directory. Initializes the detection and pose estimation models,
    and processes the input data accordingly.

    Command-line Arguments:
        --imgdir: Path to the directory containing images
        --viddir: Path to the directory containing videos
        --output-dir: Path to the output directory
        --ckpt-intvl: Checkpoint interval (default: 5)
        --device: Device to use (default: "cuda:0")
        --draw-bbox: Flag to draw bounding boxes of instances
    """
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--imgdir", type=str, help="path to images")
    parser.add_argument("--viddir", type=str, help="path to videos")
    parser.add_argument("--output-dir", type=str, help="path to the output directory")
    parser.add_argument("--ckpt-intvl", default=5, type=int, help="checkpoint interval")
    parser.add_argument("--device", default="cuda:0", type=str, help="device to use")
    parser.add_argument(
        "--draw-bbox", action="store_true", help="Draw bboxes of instances"
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    bbox_thr = 0.5
    checkpoint_interval = args.ckpt_intvl
    device = args.device
    keypoint_thr = 0.5
    nms_thr = 0.1
    min_num_keypoints_desired = 15
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + "/imgs", exist_ok=True)
    os.makedirs(out_dir + "/viz", exist_ok=True)
    os.makedirs(out_dir + "/annotations", exist_ok=True)

    out_json_file = out_dir + "/annotations/enclosure_pose_labels.json"

    det_config = "C:\\Users\\Felipe Parodi\\Documents\\felipe_code\\MacTrack\\scripts\\pose_pseudolabel_230612\\fasterrcnn_2classdet_mt_3x.py"
    det_checkpoint = "Y:\\MacTrack\\results\\mactrack_detection\\fasterrcnn2class_best_bbox_mAP_epoch_50.pth"
    # pose_config = "C:\\Users\\Felipe Parodi\\Documents\\felipe_code\\MacTrack\\scripts\\pose_pseudolabel_230612\\hrnet_w48_macaque_256x192_3x.py"
    pose_config = r"y:\MacTrack\scripts\hrnet_w48_macaque_256x192_3x_230822.py"
    pose_checkpoint = (
        r"Y:\MacTrack\results\freepick_model3_230824_2\best_coco_AP_epoch_290.pth"
    )
    # pose_checkpoint = "https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_macaque_256x192-9b34b02a_20210407.pth"

    print(
        f"Output json file: {out_json_file}. \nInitializing NHP detection and pose estimation models ..."
    )
    det_model = init_detector(det_config, det_checkpoint, device=device)
    det_model.cfg = adapt_mmdet_pipeline(det_model.cfg)
    pose_estimator = init_model(pose_config, pose_checkpoint, device=device)
    pose_estimator.cfg.visualizer.radius = 2
    pose_estimator.cfg.visualizer.alpha = 0.8
    pose_estimator.cfg.visualizer.line_width = 2
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=pose_estimator.cfg.skeleton_style
    )
    categories = [
        {
            "id": 1,
            "name": "monkey",
            "supercategory": "monkey",
            "keypoints": [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ],
            "skeleton": [
                [16, 14],
                [14, 12],
                [17, 15],
                [15, 13],
                [12, 13],
                [6, 12],
                [7, 13],
                [6, 7],
                [6, 8],
                [7, 9],
                [8, 10],
                [9, 11],
                [2, 3],
                [1, 2],
                [1, 3],
                [2, 4],
                [3, 5],
                [4, 6],
                [5, 7],
            ],
        }
    ]

    img_anno_dict = {
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    frame_id_uniq_counter = 0
    ann_uniq_id, checkpoint_count = int(0), int(0)
    id_pool = np.arange(0, 10_000_000)
    np.random.shuffle(id_pool)

    if args.imgdir:
        frame_id_uniq_counter, ann_uniq_id = process_images_in_directory(
            args.imgdir,
            id_pool,
            det_model,
            pose_estimator,
            visualizer,
            bbox_thr,
            keypoint_thr,
            nms_thr,
            min_num_keypoints_desired,
            frame_id_uniq_counter,
            ann_uniq_id,
            img_anno_dict,
            out_dir,
        )

    if args.viddir:
        frame_id_uniq_counter, ann_uniq_id = process_videos_in_directory(
            args.viddir,
            id_pool,
            det_model,
            pose_estimator,
            visualizer,
            bbox_thr,
            keypoint_thr,
            nms_thr,
            min_num_keypoints_desired,
            frame_id_uniq_counter,
            ann_uniq_id,
            img_anno_dict,
            out_dir,
        )

    # Saving the JSON file
    print(f"Number of images added to COCO json: {len(img_anno_dict['images'])}")
    with open(out_json_file, "w") as outfile:
        json.dump(img_anno_dict, outfile, indent=2)

    print("Time elapsed: ", time.time() - start_time)


if __name__ == "__main__":
    main()

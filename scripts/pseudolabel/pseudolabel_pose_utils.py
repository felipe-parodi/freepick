# utils.py
import gc
import os
import random
import json
import cv2
import glob
import mmcv
# import matplotlib.pyplot as plt
# from segment_anything import SamPredictor, sam_model_registry
import numpy as np
# import torch
from mmdet.apis import inference_detector
from mmpose.apis import inference_topdown
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from tqdm import tqdm

def checkpoint(annotations, checkpoint_file, checkpoint_counter):
    with open(f"{checkpoint_file}_{checkpoint_counter}.json", 'w') as f:
        json.dump(annotations, f, indent=4)

def overlay_mask_on_frame(frame, mask):
    # Ensure mask is a binary mask
    mask = mask.astype(bool)
    # Create a color mask
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_mask[mask] = [255, 200, 200]  # Dodger Blue color for the mask

    # Blend the color mask and the frame
    alpha = 0.4
    frame_with_mask = cv2.addWeighted(color_mask, alpha, frame, 1 - alpha, 0)

    return frame_with_mask

def all_valid_dimensions(bbox, frame_shape):
    """
    Validates the dimensions of a bounding box against a frame shape.

    Args:
        bbox (list): List of bounding box coordinates [x1, y1, x2, y2].
        frame_shape (tuple): Shape of the frame as (height, width, _).

    Returns:
        bool: True if the bounding box dimensions are valid, False otherwise.
    """
    bbox_top_left_x, bbox_top_left_y, bbox_bottom_right_x, bbox_bottom_right_y = bbox
    height, width, _ = frame_shape

    bbox_width = bbox_bottom_right_x - bbox_top_left_x
    bbox_height = bbox_bottom_right_y - bbox_top_left_y

    # Check width and height
    if bbox_width <= 0 or bbox_height <= 0:
        return False

    # Check top left corner positions
    if bbox_top_left_x < 0 or bbox_top_left_y < 0:
        return False

    # Check bottom right corner positions
    if bbox_bottom_right_x > width or bbox_bottom_right_y > height:
        return False

    return True

def process_frame(
    frame,
    det_model,
    sam_predictor,
    pose_estimator,
    visualizer,
    bbox_thr,
    keypoint_thr,
    nms_thr,
    min_num_keypoints_desired,
    frame_id_uniq,
    ann_uniq_id,
    img_anno_dict,
    out_dir,
    file_name,
):
    """
    Processes a single frame for detection and pose estimation.

    Args:
        frame (numpy.ndarray): Input frame.
        det_model (object): Detection model.
        sam_predictor (object): SAM model.
        pose_estimator (object): Pose estimation model.
        visualizer (object): Visualizer object.
        bbox_thr (float): Threshold for bounding box score.
        keypoint_thr (float): Threshold for keypoint score.
        nms_thr (float): Threshold for Non-Maximum Suppression (NMS).
        min_num_keypoints_desired (int): Minimum number of keypoints desired.
        frame_id_uniq (int): Unique frame ID.
        ann_uniq_id (int): Unique annotation ID.
        img_anno_dict (dict): Image annotation dictionary.
        out_dir (str): Output directory.
        file_name (str): Name of the file.

    Returns:
        frame_id_uniq (int): Updated unique frame ID.
        ann_uniq_id (int): Updated unique annotation ID.
    """
    num_monkeys = 1
    
    detection_results = inference_detector(det_model, frame)
    if len(detection_results.pred_instances) == 0:
        return frame_id_uniq, ann_uniq_id
    if len(detection_results.pred_instances) > num_monkeys:  ## changed to 1
        detection_results.pred_instances = detection_results.pred_instances[
            :num_monkeys
        ]
    pred_instance = detection_results.pred_instances.cpu().numpy()
    bboxes = pred_instance.bboxes
    scores = pred_instance.scores
    labels = pred_instance.labels

    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
    )
    valid_indices = np.where((labels == 0) & (scores > bbox_thr))[0]
    bboxes = bboxes[valid_indices]
    bboxes = bboxes[nms(bboxes, nms_thr), :4]

    if sam_predictor and len(bboxes) > 0:
        # Use the first bounding box for SAM prediction
        sam_predictor.set_image(frame)
        input_box = np.array([bboxes[0][0], 
                     bboxes[0][1],
                     bboxes[0][2],
                    bboxes[0][3]
        ]) # [x1, y1, x2, y2]
        input_label = np.array([1])
        center_x = (bboxes[0][0] + bboxes[0][2]) / 2
        center_y = (bboxes[0][1] + bboxes[0][3]) / 2
        input_point = np.array([center_x, center_y])
        masks, scores, logits = sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
            )
    
    pose_results = inference_topdown(pose_estimator, frame, bboxes)

    if len(pose_results) == 0:
        return frame_id_uniq, ann_uniq_id
    data_samples = merge_data_samples(pose_results)

    visualizer.add_datasample(
        "result",
        frame,
        data_sample=data_samples,
        draw_gt=False,
        draw_bbox=True,
        kpt_thr=keypoint_thr,
    )
    vis_frame = visualizer.get_image()
    if sam_predictor:
        vis_frame_with_mask = overlay_mask_on_frame(vis_frame, masks[0])
    
    height, width, _ = frame.shape

    keypoints = data_samples.pred_instances.keypoints
    keypoint_scores = data_samples.pred_instances.keypoint_scores

    annotations_added = False
    visible_keypoints = 0
    annotations_list = []

    for bbox, score, label, kpts, kpts_scores in zip(
        bboxes, scores, labels, keypoints, keypoint_scores
    ):
        visible_keypoints = 0
        if score < bbox_thr:
            continue
        if not all_valid_dimensions(bbox, frame.shape):
            continue

        bbox_top_left_x, bbox_top_left_y = bbox[0], bbox[1]
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        bbox = [bbox_top_left_x, bbox_top_left_y, bbox_width, bbox_height]

        area = round(bbox_width * bbox_height, 2)
        center = [
            bbox_top_left_x + bbox_width / 2,
            bbox_top_left_y + bbox_height / 2,
        ]
        scale = [bbox_width / 200, bbox_height / 200]

        kpts_flat = []
        for pt, pt_score in zip(kpts, kpts_scores):
            if pt_score >= keypoint_thr:
                # if greater than the image size, ignore
                if pt[0] >= width or pt[1] >= height:
                    return frame_id_uniq, ann_uniq_id
                if sam_predictor and masks[0][int(pt[1]), int(pt[0])] == 0:
                    return frame_id_uniq, ann_uniq_id
                visibility = 2  # visible
                visible_keypoints += 1
            else:
                visibility = 0  # ignore this keypoint
                pt = [0, 0]  # setting the coordinates to (0, 0)

            kpts_flat.extend([float(pt[0]), float(pt[1]), visibility])

        images = {
            "file_name": file_name,
            "height": height,
            "width": width,
            "id": frame_id_uniq,
        }
        if visible_keypoints >= min_num_keypoints_desired:
            annotations = {
                "keypoints": kpts_flat,
                "num_keypoints": visible_keypoints,
                "area": float(area),
                "iscrowd": 0,
                "image_id": int(frame_id_uniq),
                "bbox": [float(i) for i in bbox],
                "center": [float(i) for i in center],
                "scale": [float(i) for i in scale],
                "category_id": 1,
                "id": int(ann_uniq_id),
            }
            annotations_list.append(annotations)
            annotations_added = True
            for annotation in annotations_list:
                img_anno_dict["annotations"].append(annotation)
                ann_uniq_id += 1
            raw_frame = out_dir + "/imgs/" + file_name
            cv2.imwrite(raw_frame, frame)
            viz_frame = out_dir + "/viz/" + file_name
            if sam_predictor: 
                cv2.imwrite(viz_frame, vis_frame_with_mask)        
            else:
                cv2.imwrite(viz_frame, vis_frame)

    # if len(annotations_list) == len(bboxes):
    #     for annotation in annotations_list:
    #         img_anno_dict["annotations"].append(annotation)
    #         ann_uniq_id += 1
    #     raw_frame = out_dir + "/imgs/" + file_name
    #     cv2.imwrite(raw_frame, frame)
    #     viz_frame = out_dir + "/viz/" + file_name
    #     if sam_predictor: 
    #         cv2.imwrite(viz_frame, vis_frame_with_mask)        
    #     else:
    #         cv2.imwrite(viz_frame, vis_frame)
    # visible_keypoints = 0
    if annotations_added:
        img_anno_dict["images"].append(images)
    else:
        return frame_id_uniq, ann_uniq_id

    del detection_results
    del pose_results
    del data_samples
    del vis_frame

    return frame_id_uniq, ann_uniq_id

def process_images_in_directory(
    img_dir,
    id_pool,
    det_model,
    sam_predictor,
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
    final_flag
):
    """
    Processes all images in a specified directory and its subdirectories.

    Args:
        img_dir (str): Path to the directory containing images.
        id_pool (numpy.ndarray): Pool of unique IDs.
        det_model, pose_estimator, visualizer, bbox_thr, keypoint_thr, nms_thr,
        min_num_keypoints_desired, frame_id_uniq_counter, ann_uniq_id, img_anno_dict,
        out_dir: Same as process_frame function.

    Returns:
        frame_id_uniq_counter (int): Updated unique frame ID counter.
        ann_uniq_id (int): Updated unique annotation ID.
    """
    for root, _, files in os.walk(img_dir):
        if final_flag and 'final' not in root:
            continue
        for img_file in tqdm(files):
            if not img_file.endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(root, img_file)
            img_basename = os.path.basename(img_path)
            frame = cv2.imread(img_path)
            frame_id_uniq = int(id_pool[frame_id_uniq_counter])
            frame_id_uniq_counter += 1
            file_name = img_file

            frame_id_uniq, ann_uniq_id = process_frame(
                frame,
                det_model,
                sam_predictor,
                pose_estimator,
                visualizer,
                bbox_thr,
                keypoint_thr,
                nms_thr,
                min_num_keypoints_desired,
                frame_id_uniq,
                ann_uniq_id,
                img_anno_dict,
                out_dir,
                file_name,
            )
    return frame_id_uniq_counter, ann_uniq_id

def process_videos_in_directory(
    vid_dir,
    id_pool,
    det_model,
    sam_predictor,
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
    final_flag,
    ckpt_intvl
):
    """
    Processes all videos in a specified directory and its subdirectories.

    Args:
        vid_dir (str): Path to the directory containing videos.
        id_pool (numpy.ndarray): Pool of unique IDs.
        det_model, pose_estimator, visualizer, bbox_thr, keypoint_thr, nms_thr,
        min_num_keypoints_desired, frame_id_uniq_counter, ann_uniq_id, img_anno_dict,
        out_dir: Same as process_frame function.
        final_flag (bool): If true, only process videos in directories named 'final'.

    Returns:
        frame_id_uniq_counter (int): Updated unique frame ID counter.
        ann_uniq_id (int): Updated unique annotation ID.
    """
    frame_counter = 0
    camera_counter = {}
    camera_videos = {}
    for root, _, files in os.walk(vid_dir):
        if final_flag and 'final' not in root:
            continue

        print(f'Analyzing videos in {root}') 
        avi_files = glob.glob(os.path.join(root, '*.avi'))
        mp4_files = glob.glob(os.path.join(root, '*.mp4'))
        video_files = sorted(avi_files + mp4_files)
        if len(video_files) > 10 and video_files[0].endswith(".avi"):
            video_files = random.sample(video_files, 10)
        elif len(video_files) > 8 and video_files[0].endswith(".mp4"):
            video_files = random.sample(video_files, 8)

        checkpoint_interval = ckpt_intvl
        checkpoint_counter = 0
        for i, vid in enumerate(video_files):
            camera_name = vid[:7]
            if vid.endswith(".avi"):
                camera_name += "_avi"
            elif vid.endswith(".mp4"):
                camera_name += "_mp4"
            if camera_name in camera_counter:
                if vid in camera_videos[camera_name]:
                    continue
            else:
                camera_counter[camera_name] = 0
                camera_videos[camera_name] = []
            
            camera_videos[camera_name].append(vid)

            print(f'\n {vid}')
            video = mmcv.VideoReader(os.path.join(root, vid))

            for frame_id, cur_frame in enumerate(tqdm(video)):
                frame_counter += 1
                if frame_counter % 60 != 0:
                    continue
                camera_counter[camera_name] += 1
                frame_id_uniq = int(id_pool[frame_id_uniq_counter])
                frame_id_uniq_counter += 1
                vid_basename = os.path.basename(vid)[:-4]
                file_name = vid_basename + "_" + str(frame_id_uniq) + ".jpg"

                frame_id_uniq, ann_uniq_id = process_frame(
                    cur_frame,
                    det_model,
                    sam_predictor,
                    pose_estimator,
                    visualizer,
                    bbox_thr,
                    keypoint_thr,
                    nms_thr,
                    min_num_keypoints_desired,
                    frame_id_uniq,
                    ann_uniq_id,
                    img_anno_dict,
                    out_dir,
                    file_name,
                )

                if (frame_counter % 10000 == 0):  # Check if frame_counter is a multiple of 10,000
                    gc.collect()  # Run garbage collection
                    frame_counter = 0  # Reset frame counter (optional, but good for avoiding overflow)
            if i % checkpoint_interval == 0:
                checkpoint(img_anno_dict, out_dir + "/annotations\labels", checkpoint_counter)
                checkpoint_counter += 1
    return frame_id_uniq_counter, ann_uniq_id

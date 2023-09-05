# FreePick Pseudo-labeling Scripts

...

### Usage
For pose pseudo-labeling on a directory of videos or images:
```
python pseudolabel_pose_dir2coco3x.py
--viddir path2InputDirectoryofVideos\Hooke # can also use --imgdir for images
--output-dir path2OutputDirectory\freepick_hooke_230824
--draw-bbox
```

For 2-class detection (e.g., monkey and neural logger) pseudo-labeling on a directory of videos:
```
python pseudolabel_2det_vid2coco.py
--input-dir "...\test_vids"
--output-dir "...\output"
--count 2
--device "cuda:0"
```

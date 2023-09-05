# FreePick Inference Scripts

...

### Usage
'''
python topdown_demo.py 
  path2Detectionmodelconfig2\fasterrcnn_2classdet_mt_3x.py"
  path2Detectionmodelcheckpoint\fasterrcnn2class_best_bbox_mAP_epoch_50.pth 
  path2PoseModelConfig\hrnet_w48_macaque_256x192_3x_230822.py 
  path2PoseModelChkpt\best_coco_AP_epoch_410.pth 
  --input pathtoTestVideo\e3v822f.avi 
  --bbox-thr 0.9 
  --output-root path2OutputVideo\freepick_hooke220830_output.mp4 
  --device cuda:0
'''

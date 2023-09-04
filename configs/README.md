# FreePick Model Configs

Pose estimation:
hrnet

### Mix of FreePick and MacTrack dataset (1:1)

Results on 230903_train_data (3,308 images) with Faster-RCNN 2-class detector. This dataset mostly contains single-monkey images.

|      Model      | Input Size |  AP   |  AR   |                Details and Download                 |
| :-------------: | :--------: | :---: | :---: | :-------------------------------------------------: |
|    HRNet-w48_coco_macaquepose |  256x192   | 0.9393 | 0.9459 |      [HRNet Model Weights](https://drive.google.com/file/d/1nxdLVU_O-xmFx1iF-H9KvepxClCncTjt/view?usp=drive_link)      |
|  RTMPose-L_coco  |  256x192   | 0.9380 | 0.9456 |    [RTMPose-L Model Weights](https://drive.google.com/file/d/1nJ3BQ5es7xNaNa0cZx7qYOilcU51guSt/view?usp=drive_link)    |
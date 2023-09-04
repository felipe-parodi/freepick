_base_ = [
    r'C:\Users\Felipe Parodi\Documents\felipe_code\mmpose\configs\_base_\default_runtime.py',
    r'C:\Users\Felipe Parodi\Documents\felipe_code\mmpose\configs\_base_\datasets\macaque.py'
]

# runtime
train_cfg = dict(max_epochs=500, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up for 50 epochs
    dict(
        type='MultiStepLR',
        begin=50,       # start right after warm-up
        end=500,        # total training epochs
        milestones=[300, 400, 450],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict( # input 256,192 with heatmap 64,48
    type='MSRAHeatmap', input_size=(192,256), heatmap_size=(48,64), sigma=2)

skeleton_style='mmpose'

visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
])

# resume=True
load_from = 'https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_macaque_256x192-9b34b02a_20210407.pth'
# load_from = r'y:\MacTrack\results\freepick_model3_230824\best_coco_AP_epoch_50.pth'

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        frozen_stages=3, # freeze the first 3 stages
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384)
                )
                ),
        init_cfg=dict(
            type='Pretrained',
            # checkpoint = r'y:\MacTrack\results\freepick_model3_230824\best_coco_AP_epoch_50.pth',
            checkpoint='https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_macaque_256x192-9b34b02a_20210407.pth',
        )
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=48,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
        ))

# base dataset settings
dataset_type='MacaqueDataset'
data_mode='topdown'
data_root = r"Y:\MacTrack\data\freepick\230903_train_data\images\\"
anno_root = r"Y:\MacTrack\data\freepick\230903_train_data\annotations\\"

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomFlip', direction='vertical'),
    dict(type='RandomFlip', direction='diagonal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs')
]
test_pipeline = val_pipeline

# data loaders
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),  # Default sampler, supports both distributed and non-distributed training
    # batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=anno_root + "train.json",
        data_prefix=dict(img=data_root),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline)
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),  # Default sampler, supports both distributed and non-distributed training
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=anno_root + "val.json",
        data_prefix=dict(img=data_root),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        test_mode=True,
        pipeline=val_pipeline)
)
test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),  # Default sampler, supports both distributed and non-distributed training
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=anno_root + "val.json",
        data_prefix=dict(img=data_root),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        test_mode=True,
        pipeline=val_pipeline)
)

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=anno_root + "val.json")
test_evaluator = dict(
    type='CocoMetric',
    ann_file=anno_root + "val.json")

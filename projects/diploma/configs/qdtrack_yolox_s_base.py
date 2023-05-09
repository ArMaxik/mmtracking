_base_ = [
    './yolox_s_8x8.py',
    'mmtrack::_base_/default_runtime.py'
]

custom_imports = dict(imports=['models'], allow_failed_imports=False)


model = dict(
    type='QDTrackOneStage',
    data_preprocessor=dict(
        _delete_=True,
        type='TrackDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        # TODO: it is different from the master branch
        bgr_to_rgb=True,
        pad_size_divisor=32),
    detector=dict(
        bbox_head=dict(
            type='YOLOX_QDTrackHead',
            num_classes=1,
            train_cfg=dict(
                track_sampler=dict(
                    _scope_='mmdet',
                    type='CombinedSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=3,
                    add_gt_as_proposals=False,
                    pos_sampler=dict(type='InstanceBalancedPosSampler'),
                    neg_sampler=dict(type='RandomSampler')))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            '/raid/veliseev/dev/mmdetection_3.0/work_dirs/yolox_s_8xb8-100e_coco-people/best_coco'
            '/bbox_mAP_epoch_99.pth'# noqa: E501
        )),
    tracker=dict(
        type='QuasiDenseTrackerOneStage',
        init_score_thr=0.9,
        obj_score_thr=0.5,
        match_score_thr=0.5,
        memo_tracklet_frames=30,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax')
    )

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))
# learning policy
param_scheduler = [
    dict(
        type='mmdet.MultiStepLR',
        begin=0,
        end=100,
        by_epoch=True,
        milestones=[25, 50, 75])
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')

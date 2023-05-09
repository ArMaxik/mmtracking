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
    track_head=dict(
        type='QuasiDenseTrackHead',
        roi_extractor=dict(
            _scope_='mmdet',
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=1, sampling_ratio=0),
            out_channels=128,
            featmap_strides=[8, 16, 32]),
        embed_head=dict(
            type='QuasiDenseEmbedHead',
            roi_feat_size=1,
            in_channels=128,
            num_convs=4,
            num_fcs=0,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25),
            loss_track_aux=dict(
                type='L2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.1,
                hard_mining=True,
                loss_weight=1.0)),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        test_cfg=dict(score_thr=-1.0, nms=dict(type='nms', iou_threshold=0.65)),
        train_cfg=dict(
            assigner=dict(
                _scope_='mmdet',
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                _scope_='mmdet',
                type='CombinedSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=3,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(type='RandomSampler')))),
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
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
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

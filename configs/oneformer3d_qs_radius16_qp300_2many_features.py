_base_ = [
    'mmdet3d::_base_/default_runtime.py',
]
custom_imports = dict(imports=['oneformer3d'])

# model settings
num_channels = 32
num_instance_classes = 3
num_semantic_classes = 2  # Only wood (0) and leaf (1), no ground - 0-indexed
radius=16  #modify the radius of input cylinder
score_th = 00.4
chunk = 10_000  # Reduced from 20_000 to avoid OOM during validation
model = dict(
    type='ForAINetV2OneFormer3D_XAwarequery',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    in_channels=6,  # Changed from 3: x, y, z, intensity, return_number, number_of_returns
    num_channels=num_channels,
    voxel_size=0.2,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    stuff_classes=[],  # Changed from [0]: no ground/stuff class
    thing_cls=[0, 1],  # wood (0), leaf (1) - 0-indexed
    prepare_epoch=1000,   # -1, #700,
    #prepare_epoch2=-1,#1000,
    query_point_num=300,   #modify the number of query points
    radius=radius,
    score_th = score_th,
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='ForAINetv2QueryDecoder_XAwarequery',
        num_layers=6,
        num_classes=1, 
        num_instance_queries=0, 
        num_semantic_queries= num_semantic_classes,
        num_instance_classes=num_instance_classes,
        in_channels=32,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True),
    criterion=dict(
        type='ForAINetv2UnifiedCriterion_XAwarequery',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='S3DISSemanticCriterion',
            loss_weight=1.0),  # Increased from 0.2 to 1.0 - focus on semantic segmentation (wood vs leaf)
        inst_criterion=dict(
            type='InstanceCriterionForAI_OneToManyMatch',
            matcher=dict(
                type='One2ManyMatcher'),
            loss_weight=[0.1, 0.1, 0.05],  # Decreased from [1.0, 1.0, 0.5] - reduce instance segmentation focus
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=300,
        inst_score_thr=0.0,
        pan_score_thr=0.0,
        npoint_thr=10,
        obj_normalization=True,
        obj_normalization_thr=0.01,
        sp_score_thr=0.15,
        nms=True,
        matrix_nms_kernel='linear',
        num_sem_cls=num_semantic_classes,
        stuff_cls=[],  # No ground/stuff class
        thing_cls=[0, 1]))  # wood (0), leaf (1) - 0-indexed

# dataset settings
dataset_type = 'ForAINetV2SegDataset_'
data_root_forainetv2 = 'data/ForAINetV2/'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=6,  # Changed from 3: x, y, z, intensity, return_number, number_of_returns
        use_dim=[0, 1, 2, 3, 4, 5]),  # Changed from [0, 1, 2]: use all 6 dimensions
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='CylinderCrop', radius=radius),
    dict(type='GridSample', grid_size=0.2),
    dict(
        type='PointSample_',
        num_points=640000),
    # dict(type='SkipEmptyScene_'),  # TEMPORARILY DISABLED - allowing all scenes
    dict(type='AddSuperPointAnnotations', num_classes=num_semantic_classes, stuff_classes=[]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='PointInstClassMapping_', num_classes=num_instance_classes),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask', 'ratio_inspoint', 'vote_label', 'instance_mask'
        ]),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=False,
        load_dim=6,  # Changed from 3: x, y, z, intensity, return_number, number_of_returns
        use_dim=[0, 1, 2, 3, 4, 5]),  # Changed from [0, 1, 2]: use all 6 dimensions
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='CylinderCrop', radius=radius),
    dict(type='GridSample', grid_size=0.2),
    dict(
        type='PointSample_',
        num_points=640000),
    dict(type='AddSuperPointAnnotations', num_classes=num_semantic_classes, stuff_classes=[]),
    dict(
        type='Pack3DDetInputs_',
        keys=['points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask','instance_mask'])
]

# dataset settings
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_forainetv2,
        ann_file='forainetv2_oneformer3d_infos_train.pkl',
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        filter_empty_gt=False,  # Disabled to allow single-instance scenes
        test_mode=False),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,  # Reduced from 4 to save memory
    persistent_workers=False,  # Disabled to save memory
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_forainetv2,
        ann_file='forainetv2_oneformer3d_infos_val.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        test_mode=True),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_forainetv2,
        ann_file='forainetv2_oneformer3d_infos_test.pkl',
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        test_mode=True),
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

# learning rate
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=500,
        by_epoch=True,
        milestones=[300, 400],
        gamma=0.1)
]

# evaluator settings
# Note: Labels are 0-indexed: 0=wood, 1=leaf
class_names = ['wood', 'leaf']  # Only 2 classes (no ground)
# Map 0-indexed labels to class names: 0->wood, 1->leaf
label2cat = {0: 'wood', 1: 'leaf'}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[],
    classes=class_names,
    dataset_name='ForAINetV2')

sem_mapping = [0, 1]  # 0=wood, 1=leaf (0-indexed, no ground)
inst_mapping = sem_mapping  # Same mapping for instances
val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[],  # No stuff/ground class
    thing_class_inds=[0, 1],  # wood (0), leaf (1) - 0-indexed
    min_num_points=1, 
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=inst_mapping,
    metric_meta=metric_meta)
test_evaluator = val_evaluator

# training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=500, val_interval=50)  # Increased val_interval to reduce validation frequency
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# runtime settings
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=50, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

default_scope = 'mmdet3d'

# Enable find_unused_parameters for distributed training
# Required when some model parameters don't receive gradients
find_unused_parameters = True

# NOTE: This config is for training with scanner features
# Features used: intensity, return_number, number_of_returns
# Semantic classes: 2 (wood=1, leaf=2, no ground)
# Input channels: 6 (x, y, z + 3 scanner features)




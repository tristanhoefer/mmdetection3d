_base_ = [
    '../_base_/models/hv_pointpillars_secfpn_kitti_myconfig.py',
    '../_base_/datasets/kitti-3d-3class_myconfig.py',
    '../_base_/schedules/cyclic_40e.py', '../_base_/default_runtime.py'
]

#point_cloud_range = [53.335594, -156.8649, -0.00033486407, 279.4691, 1.9748741, 19.15723] das sind die genauen
#point_cloud_range = [54.335594, -157.8649, -0.00333486407, 280.4691, 2.9748741, 20.15723]
#point_cloud_range=[-1000, -1000, -1000, 1000, 1000, 1000]
point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]

# dataset settings
data_root = '/home/hoefer/Documents/mmdetection3d/data/Pointpillar_Finetuning/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
#class_names = ['Cyclist', 'Car']
#class_names = ['Car']


# PointPillars adopted a different sampling strategies among classes

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/kitti/':
#         's3://openmmlab/datasets/detection3d/kitti/',
#         'data/kitti/':
#         's3://openmmlab/datasets/detection3d/kitti/'
#     }))

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Cyclist=15)),
    classes=class_names,
    sample_groups=dict(Car=15, Cyclist=5),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    file_client_args=file_client_args)

# PointPillars uses different augmentation hyper parameters
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=file_client_args),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline, classes=class_names)),
    val=dict(pipeline=test_pipeline, classes=class_names),
    test=dict(pipeline=test_pipeline, classes=class_names))

# In practice PointPillars also uses a different schedule
# optimizer
lr = 0.001
optimizer = dict(lr=lr)
# max_norm=35 is slightly better than 10 for PointPillars in the earlier
# development of the codebase thus we keep the setting. But we does not
# specifically tune this parameter.
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# PointPillars usually need longer schedule than second, we simply double
# the training schedule. Do remind that since we use RepeatDataset and
# repeat factor is 2, so we actually train 160 epochs.
runner = dict(max_epochs=80)

# Use evaluation interval=2 reduce the number of evaluation timese
evaluation = dict(interval=1)

#train_cfg = dict(
#    type='EpochBasedTrainLoop',
#    max_epochs=80,
#    val_interval=2)
#val_cfg = dict(type='ValLoop')
#test_cfg = dict(type='TestLoop')
_base_ = ['oneformer3d_qs_radius16_qp300_2many_features.py']

# Override data paths for Heidelberg scenario
data_root_forainetv2 = 'data/ForAINetV2_heidelberg/'

# Update annotation file names and data_root
train_dataloader = dict(
    dataset=dict(
        data_root=data_root_forainetv2,
        ann_file='forainetv2_heidelberg_oneformer3d_infos_train.pkl',
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root_forainetv2,
        ann_file='forainetv2_heidelberg_oneformer3d_infos_val.pkl',
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root_forainetv2,
        ann_file='forainetv2_heidelberg_oneformer3d_infos_val.pkl',  # Use val as test
    )
)


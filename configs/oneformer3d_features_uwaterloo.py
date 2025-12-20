_base_ = ['oneformer3d_qs_radius16_qp300_2many_features.py']

# Override data paths for UWaterloo scenario
data_root_forainetv2 = 'data/ForAINetV2_uwaterloo/'

# Update annotation file names
train_dataloader = dict(
    dataset=dict(
        data_root=data_root_forainetv2,
        ann_file='forainetv2_uwaterloo_oneformer3d_infos_train.pkl',
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root_forainetv2,
        ann_file='forainetv2_uwaterloo_oneformer3d_infos_val.pkl',
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root_forainetv2,
        ann_file='forainetv2_uwaterloo_oneformer3d_infos_val.pkl',  # Use val as test
    )
)


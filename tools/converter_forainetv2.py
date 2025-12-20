# Modified from mmdetection3d/tools/dataset_converters/indoor_converter.py
# We just support ScanNet 200.
import os

import mmengine

from forainetv2_data_utils import ForAINetV2Data


def create_info_file(data_path,
                        pkl_prefix='forainetv2',
                        save_path=None,
                        workers=4):
    """Create forainetv2 dataset information file.

    Get information of the raw data and save it to the pkl file.

    Args:
        data_path (str): Path of the data.
        pkl_prefix (str, optional): Prefix of the pkl to be saved.
            Default: 'sunrgbd'.
        save_path (str, optional): Path of the pkl to be saved. Default: None.
        workers (int, optional): Number of threads to be used. Default: 4.
    """
    assert os.path.exists(data_path)
    # Allow 'forainetv2' or any prefix starting with 'forainetv2_'
    assert pkl_prefix == 'forainetv2' or pkl_prefix.startswith('forainetv2_'), \
        f'unsupported dataset {pkl_prefix} (must be "forainetv2" or start with "forainetv2_")'
    save_path = data_path if save_path is None else save_path
    assert os.path.exists(save_path)

    # generate infos for both detection and segmentation task
    train_filename = os.path.join(
        save_path, f'{pkl_prefix}_oneformer3d_infos_train.pkl')
    val_filename = os.path.join(
        save_path, f'{pkl_prefix}_oneformer3d_infos_val.pkl')
    test_filename = os.path.join(
        save_path, f'{pkl_prefix}_oneformer3d_infos_test.pkl')
    if pkl_prefix == 'forainetv2' or pkl_prefix.startswith('forainetv2_'):
        # ScanNet has a train-val-test split
        # Check if split files exist and have content
        train_list_file = os.path.join(data_path, 'meta_data', 'train_list.txt')
        val_list_file = os.path.join(data_path, 'meta_data', 'val_list.txt')
        test_list_file = os.path.join(data_path, 'meta_data', 'test_list.txt')
        
        # Process train split if it exists and has content
        if os.path.exists(train_list_file):
            train_samples = [s.strip() for s in mmengine.list_from_file(train_list_file) if s.strip()]
            if train_samples:
                train_dataset = ForAINetV2Data(root_path=data_path, split='train')
                infos_train = train_dataset.get_infos(
                    num_workers=workers, has_label=True)
                mmengine.dump(infos_train, train_filename, 'pkl')
                print(f'{pkl_prefix} info train file is saved to {train_filename}')
            else:
                # Create empty train file
                mmengine.dump([], train_filename, 'pkl')
                print(f'{pkl_prefix} info train file is empty, saved to {train_filename}')
        else:
            # Create empty train file
            mmengine.dump([], train_filename, 'pkl')
            print(f'{pkl_prefix} info train file is empty, saved to {train_filename}')
        
        # Process val split if it exists and has content
        if os.path.exists(val_list_file):
            val_samples = [s.strip() for s in mmengine.list_from_file(val_list_file) if s.strip()]
            if val_samples:
                val_dataset = ForAINetV2Data(root_path=data_path, split='val')
                infos_val = val_dataset.get_infos(
                    num_workers=workers, has_label=True)
                mmengine.dump(infos_val, val_filename, 'pkl')
                print(f'{pkl_prefix} info val file is saved to {val_filename}')
            else:
                # Create empty val file
                mmengine.dump([], val_filename, 'pkl')
                print(f'{pkl_prefix} info val file is empty, saved to {val_filename}')
        else:
            # Create empty val file
            mmengine.dump([], val_filename, 'pkl')
            print(f'{pkl_prefix} info val file is empty, saved to {val_filename}')
        
        # Process test split if it exists and has content
        if os.path.exists(test_list_file):
            test_samples = [s.strip() for s in mmengine.list_from_file(test_list_file) if s.strip()]
            if test_samples:
                test_dataset = ForAINetV2Data(root_path=data_path, split='test')
                infos_test = test_dataset.get_infos(
                    num_workers=workers, has_label=True)
                mmengine.dump(infos_test, test_filename, 'pkl')
                print(f'{pkl_prefix} info test file is saved to {test_filename}')
            else:
                # Create empty test file
                mmengine.dump([], test_filename, 'pkl')
                print(f'{pkl_prefix} info test file is empty, saved to {test_filename}')
        else:
            # If no test split, use val split as test
            print(f'No test_list.txt found, using val split as test')
            if os.path.exists(val_filename):
                import shutil
                shutil.copy2(val_filename, test_filename)
                print(f'{pkl_prefix} info test file (copied from val) is saved to {test_filename}')
            else:
                # Create empty test file
                mmengine.dump([], test_filename, 'pkl')
                print(f'{pkl_prefix} info test file is empty, saved to {test_filename}')

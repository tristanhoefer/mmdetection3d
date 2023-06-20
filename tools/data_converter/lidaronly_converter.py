import mmcv
import numpy as np
from pathlib import Path

from .lidaronly_data_utils import get_base_info


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def create_base_info_file(data_path,
                           pkl_prefix='lidaronly',
                           save_path=None,
                           relative_path=True):
    """Create info file of Base dataset.
    Given the raw data, generate its related info file in pkl format.
    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
        relative_path (bool): Whether to use relative path.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    base_infos_train = get_base_info(
        data_path,
        training=True,
        velodyne=True,
        image_ids=train_img_ids,
        relative_path=relative_path)
    # _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Base info train file is saved to {filename}')
    mmcv.dump(base_infos_train, filename)
    base_infos_val = get_base_info(
        data_path,
        training=False,
        velodyne=True,
        image_ids=val_img_ids,
        relative_path=relative_path)

    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Base info val file is saved to {filename}')
    mmcv.dump(base_infos_val, filename)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Base info trainval file is saved to {filename}')
    mmcv.dump(base_infos_train + base_infos_val, filename)

    base_infos_test = get_base_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        image_ids=test_img_ids,
        relative_path=relative_path)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'Base info test file is saved to {filename}')
    mmcv.dump(base_infos_test, filename)
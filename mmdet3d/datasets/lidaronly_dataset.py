# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import math
import tempfile
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log

from ..core import show_multi_modality_result, show_result
from ..core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose


@DATASETS.register_module()
class LidarOnlyDataset(Custom3DDataset):
    r"""KITTI Dataset.

    This class serves as the API for experiments on the `KITTI Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list, optional): The range of point cloud used to
            filter invalid predicted boxes.
            Default: [0, -40, -3, 70.4, 40, 0.0].
    """
    CLASSES = ('car', 'cyclist')
    #CLASSES = ('cyclist','car')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 pcd_limit_range=[0, -40, -3, 70.4, 40, 0.0],
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        self.split = split
        self.root_split = os.path.join(self.data_root, split)
        assert self.modality is not None
        self.pcd_limit_range = pcd_limit_range
        self.pts_prefix = pts_prefix

    def _get_pts_filename(self, idx):
        """Get point cloud filename according to the given index.

        Args:
            index (int): Index of the point cloud file to get.

        Returns:
            str: Name of the point cloud file.
        """
        pts_filename = osp.join(self.root_split, self.pts_prefix,
                                f'{idx:010d}.bin') #this was changed by me
        return pts_filename

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str): Prefix of image files.
                - img_info (dict): Image info.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        #img_filename = os.path.join(self.data_root, info['image']['image_path'])

        # TODO: consider use torch.Tensor only
        #rect = info['calib']['R0_rect'].astype(np.float32)
        #Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        #P2 = info['calib']['P2'].astype(np.float32)
        #lidar2img = P2 @ rect @ Trv2c

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                    0, 1, 2 represent xxxxx respectively.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        #rect = info['calib']['R0_rect'].astype(np.float32)
        #Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        """
        if 'plane' in info:
            # convert ground plane to velodyne coordinates
            reverse = np.linalg.inv(rect @ Trv2c)

            (plane_norm_cam,
             plane_off_cam) = (info['plane'][:3],
                               -info['plane'][:3] * info['plane'][3])
            plane_norm_lidar = \
                (reverse[:3, :3] @ plane_norm_cam[:, None])[:, 0]
            plane_off_lidar = (
                reverse[:3, :3] @ plane_off_cam[:, None][:, 0] +
                reverse[:3, 3])
            plane_lidar = np.zeros_like(plane_norm_lidar, shape=(4, ))
            plane_lidar[:3] = plane_norm_lidar
            plane_lidar[3] = -plane_norm_lidar.T @ plane_off_lidar
        else:
        """
        plane_lidar = None

        #difficulty = info['annos']['difficulty']
        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)
        loc = annos['location']
        #print(loc)
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        #gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
        #                              axis=1).astype(np.float32)

        #das hier muss angepasst werden um die bboxes_3d von camera zu velodyne coordinates zu konvertieren
        #loc_velodyne = [[line[2], line[-0], line[-1]] for line in loc]
        #loc_velodyne =  np.array([float(x) for x in loc]).reshape(-1, 3)[:, [2, -0, -1]]
        loc_velodyne = np.array([[x[2], -x[0], -x[1]] for x in loc])
        #loc_velodyne = np.array(loc).reshape(-1,3)[:, [2, -0, -1]]
        #print(loc)
        #print(loc_velodyne)

        #print(dims)
        dim_velodyne =  np.array([[x[2], x[0], x[1]] for x in dims])
        #print(dim_velodyne)
        gt_bboxes_3d_velodyne = np.concatenate([loc_velodyne, dim_velodyne, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)
        # convert gt_bboxes_3d to velodyne coordinates
        #gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
        #    self.box_mode_3d, np.linalg.inv(rect @ Trv2c))

        #gt_bboxes_3d = LiDARInstance3DBoxes(
        #    gt_bboxes_3d,
        #    box_dim=gt_bboxes_3d.shape[-1],
        #    #origin=(0.0, 0.0, 0.0)) #0.5, 0.5, 0
        #    origin=(0.5, 0.5, 0.0))

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d_velodyne,
            box_dim=gt_bboxes_3d_velodyne.shape[-1],
            #origin=(0.0, 0.0, 0.0)) #0.5, 0.5, 0
            origin=(0.5, 0.5, 0.0))
        gt_bboxes = annos['bbox']

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names,
            plane=plane_lidar,
            #difficulty=difficulty
            )
        return anns_results

    def drop_arrays_by_name(self, gt_names, used_classes):
        """Drop irrelevant ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be dropped.
        """
        inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def keep_arrays_by_name(self, gt_names, used_classes):
        """Keep useful ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be keeped.
        """
        inds = [i for i, x in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def remove_dontcare(self, ann_info):
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos. The ``'DontCare'``
                annotations will be removed according to ann_file['name'].

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(ann_info['name']) if x != 'DontCare'
        ]
        for key in ann_info.keys():
            img_filtered_annotations[key] = (
                ann_info[key][relevant_annotation_indices])
        return img_filtered_annotations

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not isinstance(outputs[0], dict):
            result_files = self.bbox2result_kitti2d(outputs, self.CLASSES,
                                                    pklfile_prefix,
                                                    submission_prefix)
        elif 'pts_bbox' in outputs[0] or 'img_bbox' in outputs[0]:
            result_files = dict()
            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + name
                if submission_prefix is not None:
                    submission_prefix_ = submission_prefix + name
                else:
                    submission_prefix_ = None
                if 'img' in name:
                    result_files = self.bbox2result_kitti2d(
                        results_, self.CLASSES, pklfile_prefix_,
                        submission_prefix_)
                else:
                    result_files_ = self.bbox2result_kitti(
                        results_, self.CLASSES, pklfile_prefix_,
                        submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES, #das hier wird aufgerufen
                                                  pklfile_prefix,
                                                  submission_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
                Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)
        from mmdet3d.core.evaluation import kitti_eval, kitti_eval_coco_style, get_official_eval_result
        gt_annos = [info['annos'] for info in self.data_infos]

        if isinstance(result_files, dict):
            ap_dict = dict()
            for name, result_files_ in result_files.items():
                eval_types = ['bev', '3d']
                if 'img' in name:
                    eval_types = ['bbox']
                ap_result_str, ap_dict_ = kitti_eval(
                    gt_annos,
                    result_files_,
                    self.CLASSES,
                    eval_types=eval_types)
                
                for ap_type, ap in ap_dict_.items():
                    ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

                print_log(
                    f'Results of {name}:\n' + ap_result_str, logger=logger)

        else:
            if metric == 'img_bbox':
                ap_result_str, ap_dict = kitti_eval(
                    gt_annos, result_files, self.CLASSES, eval_types=['bbox'])
            else:
                #ap_result_str, ap_dict = kitti_eval(gt_annos, result_files,
                #                                    self.CLASSES)
                print("Executed")
                ap_result_str = get_official_eval_result(gt_annos, result_files, self.CLASSES) #hier geändert zu coco_style evaluation
            print_log('\n' + ap_result_str, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        #return ap_dict
        ap_dict = dict()
        ap_dict["ap_scores"] = ap_result_str
        return ap_dict
    
    def do_transform(self, pc, R, T):
        # pc (3, N)
        return R @ pc + T


    def quaternion_matrix(self, quaternion):
        """Return homogeneous rotation matrix from quaternion.
        >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
        >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
        True
        """
        q = np.array(quaternion[:4], dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        _EPS = np.finfo(float).eps * 4.0
        if nq < _EPS:
            return np.identity(4)
        q *= math.sqrt(2.0 / nq)
        q = np.outer(q, q)

        return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1])
        ), dtype=np.float64)
    

    def load_pc(self, path, dtype=np.float32, dim=4):
        pc = np.fromfile(path, dtype=dtype)
        pc = pc.reshape((-1, dim))
        return pc


    def load_tf(self, path):
        with open(path, "r") as file:
            arr =[line.strip() for line in file.readlines()]

        frame_id = arr[0].split(": ")[1:]
        child_frame_id = arr[1].split(": ")[1:]
        sec = arr[2].split(": ")[1:]
        nsec = arr[3].split(": ")[1:]
        translation = arr[4:7]
        rotation = arr[7:]

        return frame_id, child_frame_id, sec, nsec, translation, rotation

    def load_static_tf(self, path, child_frame_id):
        with open(path, "r") as file:
            arr =[line.strip() for line in file.readlines()]

        index = arr.index("Child_Frame_ID: " + child_frame_id)
        translation = arr[index+1:index+4]
        rotation = arr[index+4:index+8]
        return translation, rotation


    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            #image_shape = info['image']['image_shape'][:2]
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            #if len(box_dict['bbox']) > 0: #changed this to bbox3d_lidar because bbox are empty(we have no calib information)
            if len(box_dict['box3d_lidar']) > 0:
    
                box_2d_preds = box_dict['bbox'] 
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                #for box, box_lidar, bbox, score, label in zip(
                #        box_preds, box_preds_lidar, box_2d_preds, scores,
                #        label_preds):

                for box, box_lidar, score, label in zip(
                        box_preds,
                        box_preds_lidar, scores,
                        label_preds):        
                    #bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    #bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(
                        #-np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                        -np.arctan2(-box_lidar[1], box_lidar[0]))
                    #anno['bbox'].append(bbox)
                    anno['bbox'].append(np.array([-1,-1,-1,-1]))
                    #anno['dimensions'].append(box_lidar[3:6])
                    anno['dimensions'].append(box[3:6])
                    
                    """
                    #static_translation, static_rotation = self.load_static_tf("/globalwork/data/6GEM/bagfile_112022/tf_static_msg/tf_static.txt", "vehicles/Player02/lidar_link")
                    static_translation, static_rotation = self.load_static_tf("/globalwork/data/6GEM/2022-05-19_vlp_32_less_pedestrians/Training/scene1/scene1_01/tf_static/tf_static.txt", "vehicles/Ego/velodyne_link")
                    static_translation = np.array(static_translation).astype(float)
                    static_rotation = np.array(static_rotation).astype(float)
                    R_static = self.quaternion_matrix(static_rotation)

                    tf_filename = "%010d" % idx + ".txt"
                    _,_,_,_,translation, rotation = self.load_tf("/globalwork/data/6GEM/2022-05-19_vlp_32_less_pedestrians/Training/tf_msgs_all/" + tf_filename)
                    translation = np.array(translation).astype(float)
                    rotation = np.array(rotation).astype(float)

                    R = self.quaternion_matrix(rotation) 

                    pc_xyz = box_lidar[:3].T
                    pc_xyz_static = self.do_transform(pc_xyz, R_static, static_translation)
                    pc_xyz_map = self.do_transform(pc_xyz_static, R, translation)
                    box_lidar[:3] = pc_xyz_map.T
                    """

                    #anno['location'].append(box_lidar[:3])
                    anno['location'].append(box[:3])

                    #anno['rotation_y'].append(box_lidar[6])
                    anno['rotation_y'].append(box[6])

                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
                annos.append(anno)

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:010d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def bbox2result_kitti2d(self,
                            net_outputs,
                            class_names,
                            pklfile_prefix=None,
                            submission_prefix=None):
        """Convert 2D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries have the kitti format
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        det_annos = []
        print('\nConverting prediction to KITTI format')
        for i, bboxes_per_sample in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            anno = dict(
                name=[],
                truncated=[],
                occluded=[],
                alpha=[],
                bbox=[],
                dimensions=[],
                location=[],
                rotation_y=[],
                score=[])
            sample_idx = self.data_infos[i]['image']['image_idx']

            num_example = 0
            for label in range(len(bboxes_per_sample)):
                bbox = bboxes_per_sample[label]
                for i in range(bbox.shape[0]):
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(0.0)
                    anno['bbox'].append(bbox[i, :4])
                    # set dimensions (height, width, length) to zero
                    anno['dimensions'].append(
                        np.zeros(shape=[3], dtype=np.float32))
                    # set the 3D translation to (-1000, -1000, -1000)
                    anno['location'].append(
                        np.ones(shape=[3], dtype=np.float32) * (-1000.0))
                    anno['rotation_y'].append(0.0)
                    anno['score'].append(bbox[i, 4])
                    num_example += 1

            if num_example == 0:
                annos.append(
                    dict(
                        name=np.array([]),
                        truncated=np.array([]),
                        occluded=np.array([]),
                        alpha=np.array([]),
                        bbox=np.zeros([0, 4]),
                        dimensions=np.zeros([0, 3]),
                        location=np.zeros([0, 3]),
                        rotation_y=np.array([]),
                        score=np.array([]),
                    ))
            else:
                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * num_example, dtype=np.int64)
            det_annos += annos

        if pklfile_prefix is not None:
            # save file in pkl format
            pklfile_path = (
                pklfile_prefix[:-4] if pklfile_prefix.endswith(
                    ('.pkl', '.pickle')) else pklfile_prefix)
            mmcv.dump(det_annos, pklfile_path)

        if submission_prefix is not None:
            # save file in submission format
            mmcv.mkdir_or_exist(submission_prefix)
            print(f'Saving KITTI submission to {submission_prefix}')
            for i, anno in enumerate(det_annos):
                sample_idx = self.data_infos[i]['image']['image_idx']
                cur_det_file = f'{submission_prefix}/{sample_idx:010d}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions'][::-1]  # lhw -> hwl
                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} '
                            '{:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
                                anno['name'][idx],
                                anno['alpha'][idx],
                                *bbox[idx],  # 4 float
                                *dims[idx],  # 3 float
                                *loc[idx],  # 3 float
                                anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f,
                        )
            print(f'Result is saved to {submission_prefix}')

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in
                    camera coordinate.
                - box3d_lidar (np.ndarray): 3D bounding boxes in
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        #print(labels)
        sample_idx = info['image']['image_idx']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        #rect = info['calib']['R0_rect'].astype(np.float32)
        #rect = np.array([[ 0.9999239 ,  0.00983776, -0.00744505,  0.        ],[-0.0098698 ,  0.9999421 , -0.00427846,  0.        ],[ 0.00740253,  0.00435161,  0.9999631 ,  0.        ],[ 0.        ,  0.        ,  0.        ,  1.        ]]).astype(np.float32)
        #Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        #Trv2c = np.array([[ 7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],[ 1.480249e-02,  7.280733e-04, -9.998902e-01, -7.631618e-02],[ 9.998621e-01,  7.523790e-03,  1.480755e-02, -2.717806e-01],[ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]]).astype(np.float32)
        #P2 = info['calib']['P2'].astype(np.float32)
        #P2 = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],[0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],[0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03],[0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]]).astype(np.float32)
        #img_shape = info['image']['image_shape'] #512x1024 for the 5player intersection set
        #img_shape = np.array([512, 1024]).astype(np.int32)
        #P2 = box_preds.tensor.new_tensor(P2)

        #box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)

        #box_corners = box_preds_camera.corners
        #box_corners_in_image = points_cam2img(box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        #minxy = torch.min(box_corners_in_image, dim=1)[0]
        #maxxy = torch.max(box_corners_in_image, dim=1)[0]
        #box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds_camera
        #image_shape = box_preds.tensor.new_tensor(img_shape)
        #valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
        #                  (box_2d_preds[:, 1] < image_shape[0]) &
        #                  (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds


        #if isinstance(box_preds, LiDARInstance3DBoxes):
        #    box_preds_camera = box_preds.convert_to(Box3DMode.CAM)
        #    box_preds_lidar = box_preds
        #elif isinstance(box_preds, CameraInstance3DBoxes):
        #    box_preds_camera = box_preds
        #    box_preds_lidar = box_preds.convert_to(Box3DMode.LIDAR)


        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range) #assumption box_preds is in Lidar Coordinate Frame
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        #valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)
        valid_inds = valid_pcd_inds.all(-1)

        box_preds_camera = box_preds.tensor.numpy()
        #print(box_preds_camera.shape)
        #print(box_preds_camera)
        #print(valid_inds)
        if box_preds_camera.ndim == 1:
            box_preds_camera = np.expand_dims(box_preds_camera, axis=0)
        box_preds_camera[:, :3] = np.array([[-float(x[1]), -float(x[2]),float(x[0])] for x in box_preds_camera])
        #print(box_preds_camera[:, :3])
        box_preds_camera[:, 3:6] = np.array([[float(x[4]), float(x[5]),float(x[3])] for x in box_preds_camera])
        #print(box_preds_camera)
        box_preds_lidar = box_preds

        if valid_inds.all():
            box3d_camera_return = box_preds_camera
        else:
            box3d_camera_return = box_preds_camera[valid_inds]

        if valid_inds.sum() > 0:
            return dict(
                #bbox=box_2d_preds[valid_inds, :].numpy(),
                #box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                bbox=np.zeros([0, 4]),
                box3d_camera=box3d_camera_return,
                box3d_lidar=box_preds_lidar[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        if self.modality['use_camera']:
            pipeline.insert(0, dict(type='LoadImageFromFile'))
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['point_cloud']['velodyne_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, img_metas, img = self._extract_data(
                i, pipeline, ['points', 'img_metas', 'img'])
            points = points.numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)

            # multi-modality visualization
            if self.modality['use_camera'] and 'lidar2img' in img_metas.keys():
                img = img.numpy()
                # need to transpose channel to first dim
                img = img.transpose(1, 2, 0)
                show_pred_bboxes = LiDARInstance3DBoxes(
                    pred_bboxes, origin=(0.5, 0.5, 0))
                show_gt_bboxes = LiDARInstance3DBoxes(
                    gt_bboxes, origin=(0.5, 0.5, 0))
                show_multi_modality_result(
                    img,
                    show_gt_bboxes,
                    show_pred_bboxes,
                    img_metas['lidar2img'],
                    out_dir,
                    file_name,
                    box_mode='lidar',
                    show=show)

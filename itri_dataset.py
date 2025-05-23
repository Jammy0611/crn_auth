import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
import open3d

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate
from pyquaternion import Quaternion
from PIL import Image

from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.data_classes import EvalBoxes

from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import (
            DetectionBox,
            DetectionMetricDataList,
            DetectionMetrics,
        )
from pypcd4 import PointCloud

def quaternion_yaw(rot) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(rot, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def mask_point_by_FOV(points, angle):

    phi = np.arctan2(points[:, 1], points[:, 0])

    mask = (phi <= angle) & (phi >= -angle)
    #print(mask.sum(), points.shape)
    return points[mask]


class ItriDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) #/ dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.info_root = Path("/home/workspace/OpenPCDet-Itri/data/itri_info")
        self.infos = []
        self.camera_config = self.dataset_cfg.get('CAMERA_CONFIG', None)
        if self.camera_config is not None:
            self.use_camera = self.camera_config.get('USE_CAMERA', True)
            self.camera_image_config = self.camera_config.IMAGE
        else:
            self.use_camera = False

        self.include_itri_data(self.mode)
        #self.infos = self.infos[:10]
        self.full_class_name = self.class_names
        # if self.dataset_cfg.get('LABLE_MAPPING', None) is not None:
        #     self.full_class_name = self.dataset_cfg.LABLE_MAPPING.keys()
        
        if not training:
            self.itri_gt = self.generate_itri_groun_truth()

        #self.infos = self.infos[:10]

        self.use_cache = True
        if self.use_cache:
            self.cache_data()

        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)

        self.nusc_class_name_mapping = {
            "car": "car",
            "truck": "car",
            "bus": "car",
            "construction_vehicle": "car",
            "trailer": "car",
            "pedestrian": "pedestrian",
            "motorcycle": "motorcycle",
            "bicycle": "motorcycle",
            "barrier": "barrier",
            "traffic_cone": "traffic_cone",
            "ignore": "ignore"
        }

        

        merge_nusc = False
        if self.training and merge_nusc:
            self.logger.info('Loading Nusce dataset')
            self.include_nusc_data()
            self.infos += self.nusc_infos#[:10]

       


    def cache_data(self, ):
        self.logger.info('Cache dataset')
        for index in tqdm(range(len(self.infos))):
            points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
            self.infos[index]['points'] = points

    
    def include_nusc_data(self, ):
         
        nusc_info_path = "/home/workspace/OpenPCDet-Itri/data/nusc_info/nuscenes_infos_1_yuting_train.pkl"
        with open(nusc_info_path, 'rb') as f:
                self.nusc_infos = pickle.load(f)

        self.logger.info('Loading Nusce dataset')
        self.nusc_root = Path('/home/workspace/OpenPCDet-Itri/data/nuscenes/v1.0-trainval')
        for i in range(len(self.nusc_infos)):

            gt_names = self.nusc_infos[i]['gt_names'] 
            self.nusc_infos[i]['gt_names']  = np.array([self.nusc_class_name_mapping[name] 
                                                        for name in gt_names])
            self.nusc_infos[i]['lidar2ego'] = self.nusc_infos[i]['lidar_ref_from_car'].copy()
            self.nusc_infos[i]['top2ego'] = self.nusc_infos[i]['lidar_ref_from_car'].copy()
            self.nusc_infos[i]['is_nusc'] = True
        
        self.logger.info('Total samples for Itri dataset: %d' % (len(self.nusc_infos)))
        

    def include_itri_data(self, mode):
        self.logger.info('Loading Itri dataset')
        itri_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.info_root / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                itri_infos.extend(infos)

        self.infos.extend(itri_infos)
        self.logger.info('Total samples for Itri dataset: %d' % (len(itri_infos)))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.full_class_name is None:
            return infos

        cls_infos = {name: [] for name in self.full_class_name}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.full_class_name:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.full_class_name)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.full_class_name}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.full_class_name:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]

        
        if not info.get('is_nusc', False):
            #print('use itri')
            path_split = info['lidar_path'].split('/', 1)
            if 'top' not in info['lidar_path']: #not self.dataset_cfg.get('USE_TOP', False): 
                lidar_path = f'{path_split[0]}/Itri_bag_sustech/{path_split[1]}'
            else:
                lidar_path = f'{path_split[0]}/Itri_bag_sustech/aux_lidar/{path_split[1]}'
        
            lidar_path = self.root_path / lidar_path
        
            lidar_path = lidar_path.__str__()
            #points = np.asarray(open3d.io.read_point_cloud(lidar_path).points) 
            pc = PointCloud.from_path(lidar_path)
            points = pc.numpy(("x", "y", "z", "intensity"))
            points[:, -1] = (points[:, -1] - 1) / (86400 - 1) * 255
        else:
            #print('use  nusc')
            lidar_path = self.nusc_root / info['lidar_path']
            points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :3]   
 

        #sig = np.ones((len(points), 1))
        #points = np.concatenate((points, sig), axis=1)

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def crop_image(self, input_dict):
        W, H = input_dict["ori_shape"]
        imgs = input_dict["camera_imgs"]
        img_process_infos = []
        crop_images = []
        for img in imgs:
            if self.training == True:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TRAIN
                resize = np.random.uniform(*resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TEST
                resize = np.mean(resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            
            # reisze and crop image
            img = img.resize(resize_dims)
            img = img.crop(crop)
            crop_images.append(img)
            img_process_infos.append([resize, crop, False, 0])
        
        input_dict['img_process_infos'] = img_process_infos
        input_dict['camera_imgs'] = crop_images
        return input_dict
    
    def load_camera_info(self, input_dict, info):
        input_dict["image_paths"] = []
        input_dict["lidar2camera"] = []
        input_dict["lidar2image"] = []
        input_dict["camera2ego"] = []
        input_dict["camera_intrinsics"] = []
        input_dict["camera2lidar"] = []

        for _, camera_info in info["cams"].items():
            input_dict["image_paths"].append(camera_info["data_path"])

            # lidar to camera transform
            lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            input_dict["lidar2camera"].append(lidar2camera_rt.T)

            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
            input_dict["camera_intrinsics"].append(camera_intrinsics)

            # lidar to image transform
            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            input_dict["lidar2image"].append(lidar2image)

            # camera to ego transform
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                camera_info["sensor2ego_rotation"]
            ).rotation_matrix
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
            input_dict["camera2ego"].append(camera2ego)

            # camera to lidar transform
            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            input_dict["camera2lidar"].append(camera2lidar)
        # read image
        filename = input_dict["image_paths"]
        images = []
        for name in filename:
            images.append(Image.open(str(self.root_path / name)))
        
        input_dict["camera_imgs"] = images
        input_dict["ori_shape"] = images[0].size
        
        # resize and crop image
        input_dict = self.crop_image(input_dict)

        return input_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])

        if 'points' not in info.keys():
            points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        else:
            #print("get points...")
            points = info['points'].copy()
            #print(points.shape, info['lidar_path'])
        
       

        # transform = np.asarray([[ 0.99740223,  0.01398662,  0.07066232,  0.049081],
        #             [-0.0136303,   0.99989185, -0.00552232,  0.02478098],
        #             [-0.07073191,  0.00454483,  0.99748501, -0.12852302],
        #             [ 0,         0,         0,          1,        ]])
        
        transform = self.infos[index]['lidar2ego'].copy()
        if  'top' not in info['lidar_path']: #not self.dataset_cfg.get('USE_TOP', False): 
            points[:, :3] = transform.dot(np.vstack((points[:, :3].transpose(), 
                                                           np.ones(len(points)))))[:3, :].transpose()
        else:
            #print("Use Top")
            transform_top = self.infos[index]['top2ego'].copy()
            #transform_top[2][3] -= 1.87
            points[:, :3] = transform_top.dot(np.vstack((points[:, :3].transpose(), 
                                                           np.ones(len(points)))))[:3, :].transpose() 
            
            # #points[:, 2] -= 1.87
            if not info.get('is_nusc', False): 
                points = mask_point_by_FOV(points, np.pi/3)
        
        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }

        #print(input_dict['frame_id'])

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None
            
            
            gt_boxes = info['gt_boxes'].copy()
            gt_names = info['gt_names'].copy()

            if gt_boxes.shape[0] > 0:
                gt_boxes[:, :3] = transform.dot(np.vstack((gt_boxes[:, :3].transpose(), np.ones(len(gt_boxes)))))[:3, :].transpose()
                gt_boxes[:, 6] += quaternion_yaw(transform[:3, :3])
            else:
                gt_boxes = np.zeros((0,7))

            input_dict.update({
                'gt_names': gt_names if mask is None else gt_names[mask],
                'gt_boxes': gt_boxes if mask is None else gt_boxes[mask]
            })
        if self.use_camera:
            input_dict = self.load_camera_info(input_dict, info)

        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
        return data_dict
    

    def generate_itri_groun_truth(self,):

        ### 
        print("Generate NUSC ground truth for evaluation...")

        all_gt_boxes = EvalBoxes()

        for info in self.infos:
            gt_names = info['gt_names']
            gt_boxes = info['gt_boxes']

            if self.dataset_cfg.get('LABLE_MAPPING', None) is not None:
                gt_names = np.array([self.dataset_cfg.LABLE_MAPPING[name] for name in gt_names])    

            sample_token = info['token']
            num_pts_in = info["num_lidar_pts"]
            sample_boxes = []
            
            for index in range(gt_boxes.shape[0]):
                gt_box = gt_boxes[index].copy()
                #print(gt_names, self.class_names)
                if gt_names[index] not in self.class_names:
                    continue

                
                detection_name = gt_names[index]#
                #self.class_name_mapping[gt_names[index]] #'car'# #
                #print(detection_name)
                quat = Quaternion(axis=[0, 0, 1], radians=gt_box[-1])
                box = DetectionBox(
                        sample_token=sample_token,
                        translation=gt_box[:3].tolist(),
                        size=gt_box[3:6].tolist(),
                        rotation=quat.elements.tolist(),
                        velocity=np.zeros(2).tolist(),
                        num_pts=int(num_pts_in[index]),
                        detection_name=detection_name,
                        detection_score=-1.0
                    )

                sample_boxes.append(box)
            all_gt_boxes.add_boxes(sample_token, sample_boxes)

        return all_gt_boxes

    def evaluation(self, det_annos, class_names, **kwargs):
        import json
        from . import itri_utils
        from . import itri_eval

        itri_pred = itri_utils.transform_det_annos_to_nusc_annos(det_annos, self.infos)
        
        #print(len(det_annos))
        itri_pred['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        all_preds = EvalBoxes.deserialize(itri_pred['results'], DetectionBox)
        
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        all_preds = itri_eval.filter_eval_boxes(all_preds, eval_config.class_range)
        all_gt = itri_eval.filter_eval_boxes(self.itri_gt, eval_config.class_range)

        metric_data_list = DetectionMetricDataList()
        for class_name in self.class_names:
            for dist_th in eval_config.dist_ths:
                md = accumulate(all_gt, all_preds, class_name, eval_config.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)
        
       
        metrics = DetectionMetrics(eval_config)
        for class_name in self.class_names:
            # Compute APs.
            for dist_th in eval_config.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, eval_config.min_recall, eval_config.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, eval_config.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, eval_config.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics_summary = metrics.serialize()
        result_str, result_dict = itri_utils.format_nuscene_results(metrics_summary, self.class_names, version=eval_version)
        
        return result_str, result_dict

    def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
        import torch

        if not self.dataset_cfg.get('USE_TOP', False): 
            database_save_path = self.info_root  / f'gt_database_{max_sweeps}sweeps_withvelov3'
            db_info_save_path = self.info_root  / f'itri_dbinfos_{max_sweeps}sweeps_withvelov3.pkl'
        else:
            database_save_path = self.info_root  / f'gt_database_top_withvelov3'
            db_info_save_path = self.info_root  / f'itri_dbinfos_top_withvelov3.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]

            if 'points' not in info.keys():
                points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
            else:
                points = info['points'].copy()
            gt_boxes = info['gt_boxes'].copy()
            gt_names = info['gt_names']

            transform = info['lidar2ego']

            if  'top' not in info['lidar_path']: 
                points[:, :3] = transform.dot(np.vstack((points[:, :3].transpose(), np.ones(len(points)))))[:3, :].transpose()
            else:
                #print("Use Top")
                transform_top = info['top2ego']
                points[:, :3] = transform_top.dot(np.vstack((points[:, :3].transpose(), 
                                                           np.ones(len(points)))))[:3, :].transpose() 

            gt_boxes[:, :3] = transform.dot(np.vstack((gt_boxes[:, :3].transpose(), np.ones(len(gt_boxes)))))[:3, :].transpose()
            gt_boxes[:, 6] += quaternion_yaw(transform[:3, :3])

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]

                if gt_points.shape[0] < 6:
                    continue

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.info_root))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_itri_info(dataset_cfg, data_path, save_path, max_sweeps=10, use_top=False):
    from . import itri_utils
   
    train_scenes = dataset_cfg.SCENES_SPLIT['train']
    #val_scenes = [['intersection_130250', 70, 390]]
    val_scenes = dataset_cfg.SCENES_SPLIT['val']

    print('train scene(%d), val scene(%d)' % (len(train_scenes), len(val_scenes)))

    # print("Process training data ....")
    # for train_scene in train_scenes:
    #     scene_name = train_scene[0]
    #     startIdx = train_scene[1]
    #     endIdx = train_scene[2]
    #     train_scene_infos = itri_utils.fill_infos(data_path.__str__(), scene_name, startIdx, endIdx, use_top=use_top)

    #     print('%s sample: %d' % (scene_name, len(train_scene_infos)))

    #     scene_info_path = save_path / f'{scene_name}/{scene_name}_infos.pkl'
    #     if use_top:
    #         scene_info_path = save_path / f'{scene_name}/{scene_name}_top_infos.pkl'

    #     with open(scene_info_path, 'wb') as f:
    #         pickle.dump(train_scene_infos, f)

    # print()
    # print("Process validation data ....")
    # for val_scene in val_scenes:
    #     scene_name = val_scene[0]
    #     startIdx = val_scene[1]
    #     endIdx = val_scene[2]
    #     val_scene_infos = itri_utils.fill_infos(data_path.__str__(), scene_name, startIdx, endIdx, use_top=use_top)

    #     print('%s sample: %d' % (scene_name, len(val_scene_infos)))
        
    #     scene_info_path = save_path / f'{scene_name}/{scene_name}_infos.pkl'
    #     if use_top:
    #         scene_info_path = save_path / f'{scene_name}/{scene_name}_top_infos.pkl'

    #     with open(scene_info_path, 'wb') as f:
    #         pickle.dump(val_scene_infos, f)

    # ### Merge
    train_infos = []
    print()
    print("merge all info files for training")
    for train_scene in train_scenes:
        scene_name = train_scene[0]

        scene_info_path = save_path / f'{scene_name}/{scene_name}_infos.pkl'
        if use_top:
            scene_info_path = save_path / f'{scene_name}/{scene_name}_top_infos.pkl'

        with open(scene_info_path, 'rb') as f:
            infos = pickle.load(f)
            print(len(infos))
            train_infos += infos

    train_info_path = save_path / f'train_itri_infos.pkl'
    if use_top:
        train_info_path = save_path / f'train_itri_top_infos.pkl'

    with open(train_info_path, 'wb') as f:
        pickle.dump(train_infos, f)
    print('val sample: %d' % (len(train_infos)))

  
    val_infos = []
    print()
    print("merge all info files for validation")
    for val_scene in val_scenes:
        scene_name = val_scene[0] 
        
        scene_info_path = save_path / f'{scene_name}/{scene_name}_infos.pkl'
        if use_top:
            scene_info_path = save_path / f'{scene_name}/{scene_name}_top_infos.pkl'

        with open(scene_info_path, 'rb') as f:
            infos = pickle.load(f)
            print(len(infos))
            val_infos += infos

    val_info_path = save_path / f'val_itri_infos.pkl'
    if use_top:
        val_info_path = save_path / f'val_itri_top_infos.pkl'
   
    with open(val_info_path, 'wb') as f:
        pickle.dump(val_infos, f)
    print('val sample: %d' % (len(val_infos)))


    



if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    # parser.add_argument('--func', type=str, default='create_itri_infos', help='')
    # parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    parser.add_argument('--with_cam', action='store_true', default=False, help='use camera or not')
    args = parser.parse_args()

    
    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    #dataset_cfg.VERSION = args.version
    # create_itri_info(
    #     dataset_cfg,
    #     data_path=ROOT_DIR / 'data' / 'itri',
    #     save_path=ROOT_DIR / 'data' / 'itri',
    #     max_sweeps=1,
    #     use_top=dataset_cfg.get("USE_TOP", False)
    # )

    itri_dataset = ItriDataset(
        dataset_cfg=dataset_cfg, class_names=None,
        root_path=ROOT_DIR / 'data' / 'itri',
        logger=common_utils.create_logger(), training=True
    )
    itri_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)

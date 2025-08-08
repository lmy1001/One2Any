import os
import cv2
import json
import numpy as np
from scipy.stats import truncnorm
from scipy.spatial.transform import Rotation as R
import trimesh
from  utils.aligning import depth2xyzmap, extract_bboxes, get_pts_mask_in_w, get_inverse_pose, \
    transform_coordinates_3d, save_to_ply, save_array_to_image, estimateSimilarityTransform
from torch.utils.data import Dataset
import tqdm
import copy
from pathlib import Path
from bop_toolkit_lib import pycoco_utils

def group_by_image_level(data, image_key="im_id"):
    data_per_image = {}
    for det in data:
        if isinstance(det, dict):
            dets = [det]
        else:
            dets = det
        for det in dets:
            scene_id, im_id = int(det["scene_id"]), int(det[image_key])
            key = f"{scene_id:06d}_{im_id:06d}"
            if key not in data_per_image:
                data_per_image[key] = []
            data_per_image[key].append(det)
    return data_per_image

def load_test_list_and_cnos_detections(
    root_dir, dataset_name, max_det_per_object_id=None
):
    """
    If test_setting == "localization":
    - We use a sorting techniques which has been done in MegaPose (thanks Mederic Fourmy for sharing!)
    - Idea: when there is no detection at object level, we use use the detections at image level
    else:
    - No sorting techniques since target_objects is not available
    """
    # load cnos detections
    if dataset_name in ["lmo", "tless", "tudl", "icbin", "itodd", "hb", "ycbv"]:
        year = "19"
        det_model = "cnos-fastsam"
    elif dataset_name in ["hope"]:
        year = "24"
        det_model = "cnos-sam"
    else:
        raise NotImplementedError(
            f"Dataset {dataset_name} is not supported with default detections!"
        )
    root_dir = Path(root_dir).parent
    cnos_dets_dir = (
        root_dir / "default_detections" / f"core{year}_model_based_unseen/" / det_model
    )
    # list all detections and take the one matching the dataset_name
    avail_det_files = os.listdir(cnos_dets_dir)
    cnos_dets_path = [file for file in avail_det_files if dataset_name in file][0]
    with open(os.path.join(cnos_dets_dir, cnos_dets_path), 'r') as f:
        all_cnos_dets = json.load(f)

    # sort by image_id
    all_cnos_dets_per_image = group_by_image_level(all_cnos_dets, image_key="image_id")

    target_file_path = root_dir / dataset_name / f"test_targets_bop{year}.json"
    assert target_file_path.exists(), f"Combination (dataset,  year)={dataset_name, year} is not available!"
    with open(target_file_path, 'r') as f: 
        test_list = json.load(f)
    selected_detections = []
    for idx, test in enumerate(test_list):
        test_object_id = test["obj_id"]
        scene_id, im_id = test["scene_id"], test["im_id"]
        image_key = f"{scene_id:06d}_{im_id:06d}"

        # get the detections for the current image
        if image_key in all_cnos_dets_per_image:
            cnos_dets_per_image = all_cnos_dets_per_image[image_key]
            dets = [
                    det
                    for det in cnos_dets_per_image
                    if (det["category_id"] == test_object_id)
                ]
            if len(dets) == 0:  # done in MegaPose
                dets = copy.deepcopy(cnos_dets_per_image)
                for det in dets:
                    det["category_id"] = test_object_id

            assert len(dets) > 0

            # sort the detections by score descending
            dets = sorted(
                dets,
                key=lambda x: x["score"],
                reverse=True,
            )
            # keep only the top detections
            if max_det_per_object_id is not None:
                num_instances = max_det_per_object_id
            else:
                num_instances = test["inst_count"]
            dets = dets[:num_instances]
            selected_detections.append(dets)
        else:
            print(f"No detection for {image_key}")

    print(f"Detections: {len(test_list)} test samples!")
    assert len(selected_detections) == len(test_list)
    selected_detections = group_by_image_level(
            selected_detections, image_key="image_id"
        )
    test_list = group_by_image_level(test_list, image_key="im_id")
    return test_list, selected_detections



class ycbv_dataset(Dataset):
    def __init__(self, data_path, data_name, data_type,
                 is_train=True, scale_size=480, num_view=50, cate=None):
        super().__init__()

        self.scale_size = scale_size

        self.is_train = is_train
        self.data_path = data_path
        self.data_type_path = os.path.join(data_path, data_type)
        self.data_list = []

        self.zfar = np.inf
        self.ref_data = {}
        self.view_data_list = {}
        with open(os.path.join(data_path,  'models', 'models_info.json'), 'r') as f:
            self.models_info = json.load(f)
        for scene_id in os.listdir(self.data_type_path):
            if not os.path.isdir(os.path.join(self.data_type_path, scene_id)): 
                continue
            with open(os.path.join(self.data_type_path, scene_id, 'scene_camera.json'), 'r') as f:
                scene_camera = json.load(f)
            with open(os.path.join(self.data_type_path, scene_id, 'scene_gt.json'), 'r') as f:
                scene_gt = json.load(f)
            with open(os.path.join(self.data_type_path, scene_id, 'scene_gt_info.json'), 'r') as f:
                scene_gt_info = json.load(f)

            curr_num_view = len(scene_camera.keys())
            view_ids = np.array(list(scene_camera.keys()))

            for obj_id in range(1, 22, 1):
                i = 0
                for view_id in view_ids:
                    gt = scene_gt[view_id]
                    num_obj = len(gt)
                    sample_id = -1
                    for n in range(num_obj):
                        if gt[n]['obj_id'] == obj_id:
                            sample_id = n
                            break
                    if sample_id == -1:
                        continue

                    self.data_list.append(
                        {
                            'scene': scene_id,
                            'view': f'{int(view_id):{0}{6}}',
                            'cam': scene_camera[view_id],
                            'gt': scene_gt[view_id],
                            'gt_info': scene_gt_info[view_id],
                            'sample_id': sample_id,        
                            'obj_id': obj_id
                        }
                    )
                    if i == 0:
                        self.prepare_ref_data(len(self.data_list)-1)
                    i += 1
        phase = 'train' if is_train else 'test'
        print("# of %s images: %d" % (phase, len(self.data_list)))
    
    def update_ref_data(self, pick_idx=-1):
        pick_idxs = []
        for scene_id in self.ref_data.keys():
            num_views = len(self.view_data_list[scene_id]['views'])
            if pick_idx >= num_views:
                pick_idx = np.random.choice(num_views)
            self.prepare_ref_data(0, scene_id, pick_idx=pick_idx)
            pick_idxs.append(pick_idx)
        return pick_idxs

    def prepare_ref_data(self, idx, scene_id=-1, pick_idx=-1):
        if pick_idx != -1:
            idx = pick_idx
            ref_idx, view = self.view_data_list[scene_id]['views'][idx]
            ref_data = self.data_list[ref_idx]
        else:
            ref_data = self.data_list[idx]
        
        scene = ref_data['scene']
        view = ref_data['view']
        K = np.asarray(ref_data['cam']['cam_K']).reshape(3, 3)
        obj_id = ref_data['obj_id']

        if scene not in self.ref_data:
            self.ref_data[scene] = {}
        if obj_id not in self.ref_data[scene]:
            self.ref_data[scene][obj_id] = {}

        ref_rgb = self.get_rgb(idx)
        ref_mask = self.get_mask(idx, predicted=False)
        ref_pose = self.get_gt_pose(idx)
        ref_depth = self.get_depth(idx, predicted=False)
        ref_mesh, scale_matrix = self.get_gt_mesh(obj_id)
        ref_nocs, ref_scale_matrix, ref_mask, ref_pcl = self.get_nocs(ref_rgb, ref_depth, ref_mask, K)
        bbox = extract_bboxes(ref_mask)
        c, s = self.xywh2cs(bbox, wh_max=480)
        interpolate = cv2.INTER_NEAREST
        ref_rgb_resized, *_ = self.zoom_in_v2(ref_rgb, c, s, res=self.scale_size, interpolate=interpolate)
        ref_mask_resized, *_ = self.zoom_in_v2(ref_mask, c, s, res=self.scale_size, interpolate=interpolate)
        ref_depth_resized, *_ = self.zoom_in_v2(ref_depth, c, s, res=self.scale_size, interpolate=interpolate)
        ref_nocs_resized, *_ = self.zoom_in_v2(ref_nocs, c, s, res=self.scale_size, interpolate=interpolate)
        ref_pcl_resized, *_ = self.zoom_in_v2(ref_pcl, c, s, res=self.scale_size, interpolate=interpolate)
        ref_nocs_ = ref_nocs_resized.transpose((2, 0, 1)).astype(np.float32)

        ref_rgb_ = ref_rgb_resized.transpose((2, 0, 1)).astype(np.float32) / 255.0 

        self.ref_data[scene][obj_id]['scene'] = scene
        self.ref_data[scene][obj_id]['view'] = view
        self.ref_data[scene][obj_id]['ref_pose'] = ref_pose
        self.ref_data[scene][obj_id]['ref_nocs'] = ref_nocs_
        self.ref_data[scene][obj_id]['ref_K'] = K
        self.ref_data[scene][obj_id]['ref_scale_matrix'] = ref_scale_matrix
        self.ref_data[scene][obj_id]['orig_ref_rgb'] = ref_rgb
        self.ref_data[scene][obj_id]['ref_rgb'] = ref_rgb_
        self.ref_data[scene][obj_id]['ref_mask'] = ref_mask_resized.astype(np.int32)
        self.ref_data[scene][obj_id]['ref_depth'] = ref_depth_resized
        self.ref_data[scene][obj_id]['ref_pcl'] = ref_pcl_resized
        vertices = np.asarray(ref_mesh.vertices).astype(np.float32)
        num_vertices = vertices.shape[0]
        random_indices = np.random.choice(num_vertices, 2048, replace=False)
        sampled_vertices = vertices[random_indices] 
        self.ref_data[scene][obj_id]['ref_mesh'] = sampled_vertices
        self.ref_data[scene][obj_id]['obj_id'] = obj_id

    def get_rgb(self, i):
        data = self.data_list[i]
        scene = data['scene']
        view = data['view']
        rgb_path = os.path.join(self.data_type_path, scene, 'rgb', view+".png")
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(self.data_type_path, scene, 'rgb', view+".jpg")
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb
    
    def get_mask(self,i, predicted=False):
        data = self.data_list[i]
        scene = data['scene']
        view = data['view']
        sample_id = data['sample_id']
        sample_id = f'{int(sample_id):{0}{6}}'
        if predicted:
            mask_path = os.path.join(self.data_type_path, scene, 'pred_mask', view+"_"+sample_id+".png")
        else:
            mask_path = os.path.join(self.data_type_path, scene, 'mask_visib', view+"_"+sample_id+".png")
        mask = cv2.imread(mask_path, -1)
        return mask

    def get_gt_pose(self,i):
        data = self.data_list[i]
        sample_id = data['sample_id']
        gt = data['gt'][sample_id]
        cur = np.eye(4)
        cur[:3,:3] = np.array(gt['cam_R_m2c']).reshape(3,3)
        cur[:3,3] = np.array(gt['cam_t_m2c'])/1e3
        return cur
    
    def get_depth(self,i, predicted=False):
        data = self.data_list[i]
        scene = data['scene']
        view = data['view']
        cam = data['cam']
        if not predicted:
            depth_path = os.path.join(self.data_type_path, scene, 'depth', view+".png")
            if not os.path.exists(depth_path):
                depth_path = os.path.join(self.data_type_path, scene, 'depth', view+".jpg")
            depth = cv2.imread(depth_path,-1)*1e-3 * cam['depth_scale']
            depth[(depth<0.001) | (depth>=self.zfar)] = 0
        else:
            depth_path = os.path.join(self.data_type_path, scene, 'pred_depth', view+".npz")
            data = np.load(depth_path, allow_pickle=True)
            depth = data['arr_0'].item() ['depth']
        return depth

    
    def get_gt_mesh(self, ob_id):
        ob_id_ = f'{int(ob_id):{0}{6}}'
        mesh_file = os.path.join(self.data_path, 'models', 'obj_'+ob_id_+'.ply')
        mesh = trimesh.load(mesh_file)
        mesh.vertices *= 1e-3 
        bounding_box = mesh.bounding_box
        bounding_box_extents = bounding_box.extents
        scale_factor = 2.0 / bounding_box_extents.max()
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] = scale_factor * np.eye(3)
        scale_matrix[:3, 3] = -mesh.bounding_box.centroid * scale_factor
        return mesh, scale_matrix

    def get_nocs(self, rgb, depth, mask, K, gt_pose=None, mesh=None, scale_matrix=None, visualize=None):
        grid = np.zeros_like(rgb).astype(np.float32)
        mask = mask.astype(np.float32) / 255.0
        idx_h, idx_w = np.where(np.logical_and(mask == 1, depth > 0))
        mask = np.zeros_like(mask)
        mask[idx_h, idx_w] = 1
        depth_mean = np.mean(depth[idx_h, idx_w])
        thre = 0.1
        new_idx = np.where(np.logical_and(depth[idx_h, idx_w]<=depth_mean+thre, depth[idx_h, idx_w]>=depth_mean-thre))
        idx_h = idx_h[new_idx[0]]
        idx_w = idx_w[new_idx[0]]
        mask_3d = get_pts_mask_in_w(mask, depth, K)
        mask_3d = mask_3d[new_idx[0]]
        pcl_c = np.zeros_like(rgb).astype(np.float32)
        pcl_c[idx_h, idx_w] = mask_3d
        if gt_pose is not None:
            mask_3d = transform_coordinates_3d(mask_3d.transpose(), get_inverse_pose(gt_pose)).transpose()
            if visualize is not None:
                save_to_ply(mask_3d, "./nocs_vis/"+visualize+"_cano_pts.ply")

        if scale_matrix is not None:
            mask_3d = transform_coordinates_3d(mask_3d.transpose(), scale_matrix).transpose()
        else:
            mask_width = np.max(mask_3d, axis=0) - np.min(mask_3d, axis=0)
            mask_center = (np.max(mask_3d, axis=0) + np.min(mask_3d, axis=0)) / 2
            max_width = np.max(mask_width)
            mask_3d = (mask_3d - mask_center) / (max_width*1.1)

            scale_matrix = np.eye(4)
            scale_matrix[:3, :3] = 1/ (max_width*1.1) * np.eye(3)
            scale_matrix[:3, 3] = -1 * mask_center / (max_width*1.1)

        mask_3d = mask_3d+0.5
        mask_3d = np.clip(mask_3d, 0, 1)
        if visualize is not None:
            save_to_ply(mask_3d, "./nocs_vis/"+visualize+"_nocs_pts.ply")

        grid[idx_h, idx_w] = mask_3d
        if visualize is not None:
            save_array_to_image(grid, "./nocs_vis/"+visualize+"_nocs_gt.png")

        new_mask = np.zeros_like(mask)
        new_mask[idx_h, idx_w] = 1
        return grid, scale_matrix, new_mask, pcl_c

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        scene = data['scene']
        view = data['view']
        sample_id = data['sample_id']
        obj_id = data['obj_id']
        image = self.get_rgb(idx)
        depth = self.get_depth(idx, predicted=False)
        mask = self.get_mask(idx, predicted=False)
        idx_h, idx_w = np.where(np.logical_and(mask == 255, depth > 0))
        if len(idx_h) == 0:
            return self.__getitem__(idx-1)

        gt_pose = self.get_gt_pose(idx)
        if np.array_equal(gt_pose, np.eye(4)):
            return self.__getitem__(idx-1)

        interpolate = cv2.INTER_NEAREST       
        bbox = extract_bboxes(mask)
        c, s = self.xywh2cs(bbox, wh_max=480)
        if s == 0:
            return self.__getitem__(idx-1)

        ref_data = self.ref_data[scene][obj_id]
        ref_pose = ref_data['ref_pose']
        ref_scale_matrix = ref_data['ref_scale_matrix']
        K = ref_data['ref_K']
        
        nocs_pose = np.dot(gt_pose, get_inverse_pose(ref_pose))
        nocs, _, mask, pcl_c = self.get_nocs(image, depth, mask, K, gt_pose=nocs_pose, scale_matrix=ref_scale_matrix)

        rgb_resized, *_ = self.zoom_in_v2(image, c, s, res=self.scale_size, interpolate=interpolate)
        mask_resized, *_ = self.zoom_in_v2(mask, c, s, res=self.scale_size, interpolate=interpolate)


        if len(rgb_resized.shape) == 2:
            return self.__getitem__(idx-1)
        if np.array_equal(rgb_resized, np.zeros((int(s), int(s), image.shape[-1]))) or np.array_equal(mask_resized,  np.zeros((int(s), int(s), image.shape[-1]))):
            return self.__getitem__(idx-1)

        depth_resized, *_ = self.zoom_in_v2(depth, c, s, res=self.scale_size, interpolate=interpolate)

        nocs_resized, *_ = self.zoom_in_v2(nocs, c, s, res=self.scale_size, interpolate=interpolate)
        pcl_resized, *_ = self.zoom_in_v2(pcl_c, c, s, res=self.scale_size, interpolate=interpolate)

        rgb = (rgb_resized.transpose((2, 0, 1)) / 255).astype(np.float32)
        roi_coord = nocs_resized.transpose((2, 0, 1)).astype(np.float32)
        roi_coord = np.clip(roi_coord, 0, 1)
        
        dis_sym = np.zeros((8, 4, 4))
        if 'symmetries_discrete' in self.models_info[f'{obj_id}']:
            mats = np.asarray([np.asarray(mat_list).reshape(4, 4) for mat_list in self.models_info[f'{obj_id}']['symmetries_discrete']])
            dis_sym[:mats.shape[0]] = mats
        con_sym = np.zeros((3, 6))
        if 'symmetries_continuous' in self.models_info[f'{obj_id}']:
            for i, ao in enumerate(self.models_info[f'{obj_id}']['symmetries_continuous']):
                axis = np.asarray(ao['axis'])
                offset = np.asarray(ao['offset'])
                con_sym[i] = np.concatenate([axis, offset])
        instance_id = str(scene) +'_'+ str(view) + '_' + str(obj_id)
        out_dict = {
            'image': rgb,
            'mask':  mask_resized,
            'depth': depth_resized.astype(np.float32),
            'orig_depth': depth.astype(np.float32),
            'nocs': roi_coord, 
            'filename': [scene, view],
            'sample_id': sample_id,
            'ref_data': ref_data, 
            'nocs_pose': nocs_pose,
            'roi_pcl': pcl_resized.astype(np.float32),
            'dis_sym': dis_sym.astype(np.float32),
            'con_sym': con_sym.astype(np.float32),
            'roi_class': obj_id,
            'instance_id': instance_id,
            'orig_image': image.astype(np.float32),
            'roi_mask_orig': mask.astype(np.int32)/255 
        }
        return out_dict


    def load_detections(self):
        max_det_per_object_id = 16
        self.test_list, self.cnos_dets = load_test_list_and_cnos_detections(
            self.data_path,
            'ycbv',
            max_det_per_object_id=max_det_per_object_id,
        )
    @staticmethod
    def get_keypoints(model_info, dt=5):
        mins = [model_info['min_x'], model_info['min_y'], model_info['min_z']]
        sizes = [model_info['size_x'], model_info['size_y'], model_info['size_z']]
        maxs = [mins[i]+sizes[i] for i in range(len(mins))]
        base = [c.reshape(-1) for c in np.meshgrid(*zip(mins, maxs), indexing='ij')]
        base = np.stack(base, axis=-1)
        centroid = np.mean(base, axis=0, keepdims=True)
        base = np.concatenate([centroid, base], axis=0)
        keypoints = [base]
        if 'symmetries_discrete' in model_info:
            mats = [np.asarray(mat_list).reshape(4, 4) for mat_list in model_info['symmetries_discrete']]
            for mat in mats:
                curr = keypoints[0] @ mat[0:3, 0:3].T + mat[0:3, 3:].T
                keypoints.append(curr)
        elif 'symmetries_continuous' in model_info:
            # todo: consider multiple symmetries
            ao = model_info['symmetries_continuous'][0]
            axis = np.asarray(ao['axis'])
            offset = np.asarray(ao['offset'])
            angles = np.deg2rad(np.arange(dt, 180, dt))
            rotvecs = axis.reshape(1, 3) * angles.reshape(-1, 1)
            # https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector
            rots = R.from_rotvec(rotvecs).as_matrix()
            for rot in rots:
                curr = keypoints[0] @ rot.T + offset.reshape(1, 3)
                keypoints.append(curr)
        keypoints = np.stack(keypoints, axis=0)
        return keypoints

    @staticmethod
    def get_intr(h, w):
        fx = fy = 1422.222
        res_raw = 1024
        f_x = f_y = fx * h / res_raw
        K = np.array([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
        return K
    
    @staticmethod
    def read_camera_matrix_single(json_file):
        with open(json_file, 'r', encoding='utf8') as reader:
            json_content = json.load(reader)
        camera_matrix = np.eye(4)
        camera_matrix[:3, 0] = np.array(json_content['x'])
        camera_matrix[:3, 1] = -np.array(json_content['y'])
        camera_matrix[:3, 2] = -np.array(json_content['z'])
        camera_matrix[:3, 3] = np.array(json_content['origin'])

        c2w = camera_matrix
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        c2w = np.matmul(c2w, flip_yz)
        
        T_ = np.eye(4)
        T_[:3, :3] = R.from_euler('x', -90, degrees=True).as_matrix()
        c2w = np.matmul(T_, c2w)

        w2c = np.linalg.inv(c2w)

        return w2c[0:3, 0:3], w2c[0:3, 3].reshape(1, 1, 3) * 1000
    
    @staticmethod
    def K_dpt2cld(dpt, cam_scale, K):
        dpt = dpt.astype(np.float32)
        dpt /= cam_scale

        Kinv = np.linalg.inv(K)

        h, w = dpt.shape[0], dpt.shape[1]

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones((h, w), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(w*h, 3)

        # backproj
        R = np.dot(Kinv, x2d.transpose())

        # compute 3D points
        X = R * np.tile(dpt.reshape(1, w*h), (3, 1))
        X = np.array(X).transpose()

        X = X.reshape(h, w, 3)
        return X
    
    @staticmethod
    def xywh2cs_dzi(xywh, base_ratio=1.5, sigma=1, shift_ratio=0.25, box_ratio=0.25, wh_max=480):
        # copy from
        # https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi/blob/master/lib/utils/img.py
        x, y, w, h = xywh
        shift = truncnorm.rvs(-shift_ratio / sigma, shift_ratio / sigma, scale=sigma, size=2)
        scale = 1+truncnorm.rvs(-box_ratio / sigma, box_ratio / sigma, scale=sigma, size=1)
        assert scale > 0
        center = np.array([x+w*(0.5+shift[1]), y+h*(0.5+shift[0])])
        wh = max(w, h) * base_ratio * scale
        if wh_max != None:
            wh = min(wh, wh_max)
        return center, wh

    @staticmethod
    def xywh2cs(xywh, base_ratio=1.5, wh_max=480):
        x, y, w, h = xywh
        center = np.array((x+0.5*w, y+0.5*h)) # [c_w, c_h]
        wh = max(w, h) * base_ratio
        if wh_max != None:
            wh = min(wh, wh_max)
        return center, wh
    
    @staticmethod
    def zoom_in_v2(im, c, s, res=480, interpolate=cv2.INTER_LINEAR):
        """
        copy from
        https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi/blob/master/lib/utils/img.py
        zoom in on the object with center c and size s, and resize to resolution res.
        :param im: nd.array, single-channel or 3-channel image
        :param c: (w, h), object center
        :param s: scalar, object size
        :param res: target resolution
        :param channel:
        :param interpolate:
        :return: zoomed object patch
        """
        c_w, c_h = c
        c_w, c_h, s, res = int(c_w), int(c_h), int(s), int(res)
        ndim = im.ndim
        if ndim == 2:
            im = im[..., np.newaxis]
        try:
            im_crop = np.zeros((s, s, im.shape[-1]))
        except:
            print(s)
            s = 480
            im_crop = np.zeros((s, s, im.shape[-1]))
        max_h, max_w = im.shape[0:2]
        crop_min_h, crop_min_w = max(0, c_h - s // 2), max(0, c_w - s // 2)
        crop_max_h, crop_max_w = min(max_h, c_h + s // 2), min(max_w, c_w + s // 2)
        up = s // 2 - (c_h - crop_min_h)
        down = s // 2 + (crop_max_h-c_h)
        left = s // 2 - (c_w - crop_min_w)
        right = s // 2 + (crop_max_w - c_w)
        im_crop[up:down, left:right] = im[crop_min_h:crop_max_h, crop_min_w:crop_max_w]
        im_crop = im_crop.squeeze()
        im_resize = cv2.resize(im_crop, (res, res), interpolation=interpolate)
        s = s
        if ndim == 2:
            im_resize = np.squeeze(im_resize)
        return im_resize, c_h, c_w, s
    
    def zoom_out(self, im, img_orig, c, s, res=480, interpolate=cv2.INTER_LINEAR):
        c_w, c_h = c
        c_w, c_h, s, res = int(c_w), int(c_h), int(s), int(res)
        max_h, max_w = img_orig.shape[0:2]
        crop_min_h, crop_min_w = max(0, c_h - s // 2), max(0, c_w - s // 2)
        crop_max_h, crop_max_w = min(max_h, c_h + s // 2), min(max_w, c_w + s // 2)
        up = s // 2 - (c_h - crop_min_h)
        down = s // 2 + (crop_max_h-c_h)
        left = s // 2 - (c_w - crop_min_w)
        right = s // 2 + (crop_max_w - c_w)

        im_deresize = cv2.resize(im, (s, s), interpolation=interpolate)
        if len(im_deresize) == 2:
            im_deresize = np.expand_dims(im_deresize, axis=-1)
        img_orig[crop_min_h:crop_max_h, crop_min_w:crop_max_w] = im_deresize[up:down, left:right]
        return img_orig
        
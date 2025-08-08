import os
import sys
import datetime
import torch
from skimage import exposure
import time
import glob
import numpy as np
import cv2
import glob
from torch.utils.data import Dataset, DataLoader

import pdb
from torchvision import transforms
from PIL import Image
import pdb
from utils.aligning import save_array_to_image, get_inverse_pose, save_to_ply, transform_coordinates_3d, extract_bboxes, get_pts_mask_in_w, estimateSimilarityTransform
import json
from scipy.stats import truncnorm
from scipy.spatial.transform import Rotation as R
import trimesh

glcam_in_cvcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]]).astype(float)

def normalizeRotation(pose):
  new_pose = pose.copy()
  scales = np.linalg.norm(pose[:3,:3],axis=0)
  new_pose[:3,:3] /= scales.reshape(1,3)
  return new_pose

class combined_dataset(Dataset):
      def __init__(self,
            data_path, data_name, data_type,
            resize_to_hw=None,
            move_invalid_to_far_plane: bool = True,
            rgb_transform=lambda x: x / 255.0 * 2 - 1,  #  [0, 255] -> [-1, 1],
            scale_size=160,
            cate="all", 
            **kwargs,
      ) -> None:
        super().__init__()
        dataset_dir = data_path
        self.data_type = data_type #gso / objaverse/ ov9d
        self.resize_to_hw = resize_to_hw
        self.rgb_transform = rgb_transform
        self.move_invalid_to_far_plane = move_invalid_to_far_plane
        self.cate = cate
        self.scale_size = scale_size

        self.data_list = []
        self.idx = 0
        self.data_path = dataset_dir
        if data_type == 'train':
          self.load_ov9d_dataset(dataset_dir, 'ov9d')
          self.load_dataset(dataset_dir, 'gso')
          self.load_dataset(dataset_dir, 'objaverse')
        else:
          self.load_ov9d_dataset(dataset_dir, 'ov9d')
      
      def load_scene_data(self, scene_dir, data_name):
        if not os.path.exists(os.path.join(scene_dir, 'rgb')) or \
          not os.path.exists(os.path.join(scene_dir, 'distance_to_image_plane')) or \
          not os.path.exists(os.path.join(scene_dir, 'instance_segmentation')):
          return None, None, None, None
        if os.path.exists(os.path.join(scene_dir, 'camera_params', 'camera_params_000000.json')):
          camera_param_path = os.path.join(scene_dir, 'camera_params', 'camera_params_000000.json')
          with open(camera_param_path, 'r') as f:
            if os.stat(camera_param_path).st_size == 0:
                return None, None, None, None
            camera_params = json.load(f)
            world_in_glcam = np.array(camera_params['cameraViewTransform']).reshape(4,4).T
            cam_in_world = np.linalg.inv(world_in_glcam)@glcam_in_cvcam
            world_in_cam = np.linalg.inv(cam_in_world)
            cam_K = self.get_cam_K(camera_params)
            mask_path = os.path.join(scene_dir, 'instance_segmentation')
            occlusion_path = os.path.join(scene_dir, 'bounding_box_2d_loose', 'bounding_box_2d_loose_000000.npy')
            occlusion_instance_path = os.path.join(scene_dir, 'bounding_box_2d_loose', 'bounding_box_2d_loose_prim_paths_000000.json')
            if os.path.exists(occlusion_path):
              occlusion_list = np.load(occlusion_path)
              if os.stat(occlusion_instance_path).st_size == 0:
                return None, None, None, None
              with open(occlusion_instance_path, 'r') as f:
                 instance_id_class = json.load(f)

            ob_list_in_scene = {}
            occlusion_scene = {}
            with open(os.path.join(mask_path, 'instance_segmentation_mapping_000000.json'), 'r') as f:
              instance_mask = json.load(f)
              for instance_id, mask_idx in enumerate(instance_mask.keys()):
                if mask_idx in ['0', '1']:
                  continue
                if instance_mask[mask_idx].split('/')[2] != 'objects':
                  continue
                ob_name = instance_mask[mask_idx].split('/')[3].replace(data_name+'_', '', 1)
                
                if os.path.exists(occlusion_path):
                  occ_idx = [idx for idx in range(len(instance_id_class)) if ob_name in instance_id_class[idx]]
                  if len(occ_idx) == 1:
                    occ_idx = occ_idx[0]
                    occlusion = occlusion_list[int(occ_idx)][-1]
                    occlusion_scene[ob_name] = occlusion
                mask_id = int(mask_idx)
                ob_list_in_scene[ob_name] = mask_id
        
          return cam_K, world_in_cam, ob_list_in_scene, occlusion_scene
        else:
          return None, None, None, None
        
      def load_dataset(self, dataset_dir, data_name):
        data_start = len(self.data_list)
        data_path = os.path.join(dataset_dir, data_name)
        if not os.path.exists(data_path):
           return
        for scene_id in os.listdir(data_path):
            scene_path = os.path.join(data_path, scene_id)
            for sub_scene_id in os.listdir(scene_path):
                  sub_scene_id_dir = os.path.join(scene_path, sub_scene_id)
                  if os.path.isdir(sub_scene_id_dir):
                        if os.path.exists(os.path.join(sub_scene_id_dir, 'states.json')):
                          with open(os.path.join(sub_scene_id_dir, 'states.json'), 'r') as f:
                              states = json.load(f)
                        else:
                          continue
                        objects = states['objects']

                        for frame_id in os.listdir(sub_scene_id_dir):
                             frame_dir = os.path.join(sub_scene_id_dir, frame_id)
                             if os.path.isdir(frame_dir):
                              data_dir = os.path.join(frame_dir, 'RenderProduct_Replicator')
                              ref_data_dir = os.path.join(frame_dir, 'RenderProduct_Replicator_01')
                              cam_K, world_in_cam, ob_list_in_scene, occlusion_scene = self.load_scene_data(data_dir, data_name)
                              ref_cam_K, ref_world_in_cam, ref_ob_list_in_scene, ref_occlusion_scene = self.load_scene_data(ref_data_dir, data_name)
                              if cam_K is None or ref_cam_K is None:
                                continue
                              if not ob_list_in_scene or not ref_ob_list_in_scene:
                                 continue
                              for ob_name in ob_list_in_scene.keys():
                                ob_in_world = np.array(objects[ob_name]['transform_matrix_world']).reshape(4,4).T
                                scale = np.array(objects[ob_name]['scale'])

                                if ob_name in ob_list_in_scene and ob_name in ref_ob_list_in_scene:
                                  mask_id = ob_list_in_scene[ob_name]
                                  ob_in_cam = world_in_cam@ob_in_world            

                                  ref_mask_id = ref_ob_list_in_scene[ob_name]
                                  ref_ob_in_cam = ref_world_in_cam@ob_in_world 

                                  if occlusion_scene:
                                    if ob_name in occlusion_scene.keys() and ob_name in ref_occlusion_scene.keys():
                                      if occlusion_scene[ob_name] > 0.5 or ref_occlusion_scene[ob_name] > 0.5:
                                        continue
                                   
                                  self.data_list.append(
                                    {
                                      'data_name': data_name,
                                      'scene': scene_id,
                                      'sub_scene_id': sub_scene_id,
                                      'frame_id': frame_id,
                                      'cam_K': cam_K,
                                      'gt': ob_in_cam,
                                      'ob_name': ob_name,
                                      'ob_in_world': ob_in_world,
                                      'mask_id': mask_id,
                                      'ref_mask_id': ref_mask_id,
                                      'ref_gt': ref_ob_in_cam,
                                      'ref_cam_K': ref_cam_K
                                    }
                                    )
                                  self.idx += 1   
        print("# of %s images: %d" % (data_name, len(self.data_list)-data_start))

      def load_ov9d_dataset(self, dataset_dir, data_name, num_view=50):
        """
        Load a subset of the CAMERA dataset.
        dataset_dir: The root directory of the CAMERA dataset.
        subset: What to load (train, val)
        if_calculate_mean: if calculate the mean color of the images in this dataset
        """
        data_start = len(self.data_list)
        data_path = os.path.join(dataset_dir, data_name)
        if self.data_type == 'train':
            ov9d_data_path = os.path.join(data_path, self.data_type)
        else:
            ov9d_data_path = os.path.join(data_path, self.data_type, self.cate)

        with open(os.path.join(data_path, 'models_info_with_symmetry.json'), 'r') as f:
            self.models_info = json.load(f)

        for scene in os.listdir(ov9d_data_path):
            scene_separate = scene.split('_')
            frame_id = scene_separate[-1]
            sub_scene_id = scene_separate[-2]
            scene_id = '_'.join(scene_separate[:-2])
            if os.path.isdir(os.path.join(ov9d_data_path, scene)):
                with open(os.path.join(ov9d_data_path, scene, 'scene_camera.json'), 'r') as f:
                    scene_camera = json.load(f)
                with open(os.path.join(ov9d_data_path, scene, 'scene_gt.json'), 'r') as f:
                    scene_gt = json.load(f)
                with open(os.path.join(ov9d_data_path, scene, 'scene_gt_info.json'), 'r') as f:
                    scene_gt_info = json.load(f)
                view_ids = np.array(list(scene_camera.keys()))

                view_ids.sort()
                for view_id in view_ids:
                  if scene_gt_info[view_id][0]['bbox_visib'][2] < 50 or scene_gt_info[view_id][0]['bbox_visib'][3] < 50:
                      continue
                  if len(scene_gt[view_id]) > 1:   ####Only consider one object per scene
                      continue
                  ref_view_id = np.random.choice(view_ids)
                  self.data_list.append(
                      {
                        'data_name': data_name,
                        'scene': scene_id,
                        'sub_scene_id': sub_scene_id,
                        'frame_id': frame_id,
                        'cam_K': scene_camera[view_id],
                        'gt': scene_gt[view_id],
                        'ob_name': scene_id,
                        'ob_in_world': scene_gt[view_id],
                        'mask_id': int(view_id),
                        'ref_mask_id': int(ref_view_id),
                        'ref_gt': scene_gt[ref_view_id],
                        'ref_cam_K': scene_camera[ref_view_id]
                        }
                      )
                  self.idx += 1
        print("# of %s images: %d" % (data_name, len(self.data_list)-data_start))     
      
      def get_cam_K(self, camera_params):
        W, H = camera_params['renderProductResolution']

        focal_length = camera_params["cameraFocalLength"]
        horiz_aperture = camera_params["cameraAperture"][0]
        vert_aperture = H / W * horiz_aperture
        focal_y = H * focal_length / vert_aperture
        focal_x = W * focal_length / horiz_aperture
        center_y = H * 0.5
        center_x = W * 0.5

        fx, fy, cx, cy = focal_x, focal_y, center_x, center_y
        K = np.eye(3)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cx
        K[1,2] = cy

        return K


      def get_mesh(self, ob_name):
          mesh_path = os.path.join(self.model_path, ob_name, 'meshes', 'model.obj')
          if os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path)
          else:
            mesh = None
          return mesh

      def get_nocs(self, rgb, depth, mask, K, gt_pose=None, mesh=None, scale_matrix=None, visualize=None, return_pcl=False):
        idx_h, idx_w = np.where(np.logical_and(mask == 1, depth > 0))
        mask = np.zeros_like(mask)
        mask[idx_h, idx_w] = 1
        mask_3d = get_pts_mask_in_w(mask, depth, K)
        mask_3d_orig = mask_3d.copy()
        center = np.median(mask_3d, axis=0)
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
        if visualize is not None:
            save_to_ply(mask_3d, "./nocs_vis/"+visualize+"_nocs_pts.ply")

        valid_idx =  np.where(np.all((mask_3d >= 0) & (mask_3d <= 1), axis=1))[0]
        if len(valid_idx) == 0:
           valid_idx = np.arange(len(idx_h))
        grid = np.zeros_like(rgb).astype(np.float32)
        grid[idx_h[valid_idx], idx_w[valid_idx]] = mask_3d[valid_idx]
        pcl_c = np.zeros_like(rgb).astype(np.float32)
        pcl_c[idx_h[valid_idx], idx_w[valid_idx]] = mask_3d_orig[valid_idx]
        new_mask = np.zeros_like(mask)
        new_mask[idx_h[valid_idx], idx_w[valid_idx]] = 1

        if visualize is not None:
            save_array_to_image(grid, "./nocs_vis/"+visualize+"_nocs_gt.png")
        
        return grid, scale_matrix, new_mask, pcl_c, center
    
      
      def get_ref_nocs(self, index, visualize=None):
        info = self.data_list[index]
        data_name = info['data_name']
        data_path = os.path.join(self.data_path, data_name)
        scene_id, sub_scene_id, frame_id  = info['scene'], info['sub_scene_id'], info['frame_id'], 
        cam_K, gt_pose, ob_name, mask_id = info['ref_cam_K'], info['ref_gt'], info['ob_name'], info['ref_mask_id']
        data_root_path = os.path.join(data_path, scene_id, sub_scene_id, frame_id, 'RenderProduct_Replicator_01')
        image, depth, mask = self.load_data(data_root_path,  mask_id)

        mask_orig = mask.copy()
        ref_nocs, ref_scale_matrix, mask, pcl_c, center = self.get_nocs(image, depth, mask, cam_K, visualize=visualize, return_pcl=True)
        if np.sum(mask) == 0:
          return None
        bbox = extract_bboxes(mask)
        c, s = self.xywh2cs(bbox, wh_max=480)
        interpolate = cv2.INTER_NEAREST
        ref_rgb_resized, *_ = self.zoom_in_v2(image, c, s, res=self.scale_size, interpolate=interpolate)
        ref_mask_resized, *_ = self.zoom_in_v2(mask, c, s, res=self.scale_size, interpolate=interpolate)
        ref_nocs_resized, *_ = self.zoom_in_v2(ref_nocs, c, s, res=self.scale_size, interpolate=interpolate)
        ref_pcl_resized, *_ = self.zoom_in_v2(pcl_c, c, s, res=self.scale_size, interpolate=interpolate)

        ref_pose = gt_pose

        rgb = (ref_rgb_resized.transpose((2, 0, 1)) / 255).astype(np.float32)
        roi_coord = ref_nocs_resized.transpose((2, 0, 1)).astype(np.float32)

        cur_ref_data = {}
        filename = [data_name, scene_id, sub_scene_id, frame_id]
        cur_ref_data['filename'] = filename
        cur_ref_data['ref_nocs'] = roi_coord
        cur_ref_data['ref_rgb'] = rgb
        cur_ref_data['ref_mask'] = ref_mask_resized
        cur_ref_data['ref_pose'] = ref_pose
        cur_ref_data['ref_scale_matrix'] = ref_scale_matrix

        cur_ref_data['ref_obj_id'] = ob_name
        cur_ref_data['ref_K'] = cam_K
        
        idx_h, idx_w = np.where(ref_mask_resized==1)
        mesh_pts = (ref_pcl_resized[idx_h, idx_w] - ref_pose[:3, 3]).dot(ref_pose[:3, :3])
        if len(mesh_pts) == 0:
          return None
        idx = np.random.choice(len(mesh_pts), 2048)
        cur_ref_data['ref_mesh'] = mesh_pts[idx]
        return cur_ref_data
      
      def get_ref_nocs_ov9d(self, index, visualize=None):
        info = self.data_list[index]
        data_name = info['data_name']
        data_path = os.path.join(self.data_path, data_name)
        scene_id, sub_scene_id, frame_id  = info['scene'], info['sub_scene_id'], info['frame_id'], 
        cam, ref_gt, ob_name, mask_id = info['ref_cam_K'], info['ref_gt'], info['ob_name'], info['ref_mask_id']
        if self.data_type == 'train':
          root_path = os.path.join(data_path, self.data_type)
        else:
          root_path = os.path.join(data_path, self.data_type, self.cate)
        scene_name = scene_id + '_' + sub_scene_id + '_' + frame_id
        root_data_path = os.path.join(root_path, scene_name)

        sample_id = 0
        depth_scale = cam['depth_scale']
        image, depth, mask = self.load_data_ov9d(root_data_path, mask_id, depth_scale, sample_id=sample_id)

        cam_K = np.asarray(cam['cam_K']).reshape(3, 3)
        cam_R_m2c, cam_t_m2c, obj_id = ref_gt[sample_id]['cam_R_m2c'], ref_gt[sample_id]['cam_t_m2c'], ref_gt[sample_id]['obj_id']
        cam_R_m2c = np.asarray(cam_R_m2c).reshape(3, 3)
        cam_t_m2c = np.asarray(cam_t_m2c).reshape(1, 1, 3)

        ref_nocs, ref_scale_matrix, mask, pcl_c, center = self.get_nocs(image, depth, mask, cam_K, visualize=visualize, return_pcl=True)
        if np.sum(mask) == 0:
          return None
        bbox = extract_bboxes(mask)
        c, s = self.xywh2cs(bbox, wh_max=480)
        interpolate = cv2.INTER_NEAREST
        ref_rgb_resized, *_ = self.zoom_in_v2(image, c, s, res=self.scale_size, interpolate=interpolate)
        ref_mask_resized, *_ = self.zoom_in_v2(mask, c, s, res=self.scale_size, interpolate=interpolate)
        ref_nocs_resized, *_ = self.zoom_in_v2(ref_nocs, c, s, res=self.scale_size, interpolate=interpolate)
        ref_pcl_resized, *_ = self.zoom_in_v2(pcl_c, c, s, res=self.scale_size, interpolate=interpolate)
        
        ref_pose = np.eye(4)
        ref_pose[:3, :3]= cam_R_m2c
        ref_pose[:3, 3] = cam_t_m2c[0] /1e3
        

        rgb = (ref_rgb_resized.transpose((2, 0, 1)) / 255).astype(np.float32)
        roi_coord = ref_nocs_resized.transpose((2, 0, 1)).astype(np.float32)
        
        cur_ref_data = {}
        filename = [data_name, scene_id, sub_scene_id, frame_id]
        cur_ref_data['filename'] = filename
        cur_ref_data['ref_nocs'] = roi_coord
        cur_ref_data['ref_rgb'] = rgb
        cur_ref_data['ref_mask'] = ref_mask_resized
        cur_ref_data['ref_pose'] = ref_pose
        cur_ref_data['ref_scale_matrix'] = ref_scale_matrix
 
        cur_ref_data['ref_obj_id'] = ob_name
        cur_ref_data['ref_K'] = cam_K

        
        idx_h, idx_w = np.where(ref_mask_resized==1)
        mesh_pts = (ref_pcl_resized[idx_h, idx_w] - ref_pose[:3, 3]).dot(ref_pose[:3, :3])
        if len(mesh_pts) == 0:
          return None
        idx = np.random.choice(len(mesh_pts), 2048)
        cur_ref_data['ref_mesh'] = mesh_pts[idx]
        return cur_ref_data
      
      def __len__(self):
        return len(self.data_list)

      def __getitem__(self, index):
        info = self.data_list[index]
        data_name = info['data_name']
        scene_id, sub_scene_id, frame_id  = info['scene'], info['sub_scene_id'], info['frame_id'], 
        cam, gt, ob_name, mask_id = info['cam_K'], info['gt'],  info['ob_name'], info['mask_id']

        if data_name == 'ov9d':
          cur_ref_data = self.get_ref_nocs_ov9d(index)
        else:
          cur_ref_data = self.get_ref_nocs(index)
        if cur_ref_data is None:
          print("hard to see the object")
          return self.__getitem__(index+1) if index+1 < len(self.data_list) else self.__getitem__(index-1)
        ref_pose = cur_ref_data['ref_pose']
        ref_scale_matrix = cur_ref_data['ref_scale_matrix']


        root_path = os.path.join(self.data_path, data_name)
        if data_name == 'ov9d':
          if self.data_type == 'train':
            root_path = os.path.join(root_path, self.data_type)
          else:
            root_path = os.path.join(root_path, self.data_type, self.cate)
          scene_name = scene_id + '_' + sub_scene_id + '_' + frame_id
          data_root_path = os.path.join(root_path, scene_name)
          depth_scale = cam['depth_scale']
          sample_id = 0
          image, depth, mask = self.load_data_ov9d(data_root_path, mask_id, depth_scale, sample_id)
          cam_K = np.asarray(cam['cam_K']).reshape(3, 3)
          cam_R_m2c, cam_t_m2c, obj_id = gt[sample_id]['cam_R_m2c'], gt[sample_id]['cam_t_m2c'], gt[sample_id]['obj_id']
          cam_R_m2c = np.asarray(cam_R_m2c).reshape(3, 3)
          cam_t_m2c = np.asarray(cam_t_m2c).reshape(1, 1, 3)
          gt_pose = np.eye(4)
          gt_pose[:3, :3] =  cam_R_m2c
          gt_pose[:3, 3] = cam_t_m2c[0]/1e3
          gt_trans = gt_pose[:3, 3]
        else:
          data_root_path = os.path.join(self.data_path, data_name, scene_id, sub_scene_id, frame_id, 'RenderProduct_Replicator')
          image, depth, mask = self.load_data(data_root_path, mask_id)
          cam_K = cam
          gt_pose = gt
          gt_trans = info['ob_in_world'][:3, 3]

        nocs_pose = np.dot(gt_pose, get_inverse_pose(ref_pose))
        nocs, _, mask, pcl_c, center = self.get_nocs(image, depth, mask, cam_K, scale_matrix=ref_scale_matrix, gt_pose=nocs_pose)
 
        idx_h, idx_w = np.where(mask==1)
        if len(idx_h) < 50:
          print("hard to see the object")
          return self.__getitem__(index+1) if index+1 < len(self.data_list) else self.__getitem__(index-1)

        bbox = extract_bboxes(mask)
        c, s = self.xywh2cs(bbox, wh_max=480)
        interpolate = cv2.INTER_NEAREST
        rgb_resized, *_ = self.zoom_in_v2(image, c, s, res=self.scale_size, interpolate=interpolate)
        mask_resized, *_ = self.zoom_in_v2(mask, c, s, res=self.scale_size, interpolate=interpolate)
        depth_resized, *_ = self.zoom_in_v2(depth, c, s, res=self.scale_size, interpolate=interpolate)
        nocs_resized, *_ = self.zoom_in_v2(nocs, c, s, res=self.scale_size, interpolate=interpolate)
        pcl_resized, *_ = self.zoom_in_v2(pcl_c, c, s, res=self.scale_size, interpolate=interpolate)
      
        
        dis_sym = np.zeros((3, 4, 4))
        con_sym = np.zeros((3, 6))
        
        if data_name == 'ov9d':
          if 'symmetries_discrete' in self.models_info[f'{obj_id}']:
            mats = np.asarray([np.asarray(mat_list).reshape(4, 4) for mat_list in self.models_info[f'{obj_id}']['symmetries_discrete']])
            dis_sym[:mats.shape[0]] = mats
          if 'symmetries_continuous' in self.models_info[f'{obj_id}']:
            for i, ao in enumerate(self.models_info[f'{obj_id}']['symmetries_continuous']):
                axis = np.asarray(ao['axis'])
                offset = np.asarray(ao['offset'])
                con_sym[i] = np.concatenate([axis, offset])



        rgb = (rgb_resized.transpose((2, 0, 1)) / 255).astype(np.float32)
        roi_coord = nocs_resized.transpose((2, 0, 1)).astype(np.float32)

        out_dict = {
            'image': rgb,
            'mask': mask_resized,
            'depth': depth_resized,
            'nocs': roi_coord,
            "ref_data": cur_ref_data,
            'filename': [data_name, scene_id, sub_scene_id, frame_id],
            "roi_pcl": pcl_resized.astype(np.float32),
            'roi_center': center.astype(np.float32),
            'dis_sym': dis_sym.astype(np.float32),
            'con_sym': con_sym.astype(np.float32),
            'roi_class': ob_name,
            'nocs_pose': nocs_pose,
            'gt_trans': gt_trans
        }
        return out_dict


      def load_data(self, data_path, mask_id):
        rgb_path = os.path.join(data_path, 'rgb', 'rgb_000000.png')
        depth_path = os.path.join(data_path,  'distance_to_image_plane', 'distance_to_image_plane_000000.npy')
        mask_path = os.path.join(data_path, 'instance_segmentation')

        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)
        depth[depth==np.inf] = 0
        depth[depth==-np.inf] = 0
        
        mask = self.load_mask(mask_path, mask_id)
        return image, depth, mask

      def load_data_ov9d(self, data_path, mask_id, depth_scale=1.0, sample_id=0):
        rgb_path = os.path.join(data_path, 'rgb', f'{mask_id:06}.png')
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(data_path, 'rgb', f'{mask_id:06}.jpg')
        depth_path = os.path.join(data_path, 'depth', f'{mask_id:06}.png')
        mask_path =  os.path.join(data_path, 'mask_visib', '_'.join([f'{mask_id:06}', f'{sample_id:06}'])+'.png')

        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_path, -1) * 1e-3 * depth_scale
        depth[depth==np.inf] = 0
        depth[depth==-np.inf] = 0

        mask = cv2.imread(mask_path, -1)
        mask = mask.astype(np.float32) / 255.0

        return image, depth, mask
      
      def load_mask(self, mask_path, mask_id):
        mask_image_path = os.path.join(mask_path, 'instance_segmentation_000000.png')
        mask_image = cv2.imread(mask_image_path, -1)
        h, w = mask_image.shape
        mask = np.zeros((h,w))
        mask[mask_image == mask_id] = 1
        return mask
      
      @staticmethod
      def xywh2cs(xywh, base_ratio=1.2, wh_max=480):
        x, y, w, h = xywh
        center = np.array((x+0.5*w, y+0.5*h))
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
        im_crop = im[crop_min_h:crop_max_h, crop_min_w:crop_max_w]
        im_resize = cv2.resize(im_crop, (res, res), interpolation=interpolate)
        if ndim == 2:
            im_resize = np.squeeze(im_resize)
        return im_resize, c_h, c_w, s

if __name__=='__main__':
    tmpdir = os.getenv('TMPDIR')
    dataset = combined_dataset(os.path.join(tmpdir,'./foundationpose_dataset/'), 'foundation_data', 'train')
    val_loader = DataLoader(
            dataset=dataset,
            batch_size=4,
            num_workers=0,
            shuffle=True,
        )

    for i, data in enumerate(val_loader):
        ref_data = data['ref_data']
        ob_name = ref_data['ref_obj_id'][0]

        rgb = data["image"].detach().cpu().numpy()
        roi_mask = data['mask'].detach().cpu().numpy()
        roi_coord = data['nocs'].detach().cpu().numpy()
        ref_rgb = ref_data['ref_rgb'].detach().cpu().numpy()
        ref_mask = ref_data['ref_mask'].detach().cpu().numpy()
        ref_coord = ref_data['ref_nocs'].detach().cpu().numpy()

        gt_pose = data['nocs_pose'].detach().cpu().numpy()
        ref_pose = ref_data['ref_pose'].detach().cpu().numpy()
        pose_image = gt_pose[0] @ ref_pose[0]
        rot = normalizeRotation(pose_image[:3, :3])
        save_array_to_image(rgb[0], "./output/image_%d.png"%(i))
        save_array_to_image(roi_mask[0], "./output/roi_mask%d.png"%(i))
        save_array_to_image(roi_coord[0], "./output/roi_coord%d.png"%(i))
        save_array_to_image(ref_rgb[0], "./output/ref_image%d.png"%(i))
        save_array_to_image(ref_mask[0], "./output/ref_mask%d.png"%(i))
        save_array_to_image(ref_coord[0], "./output/ref_coord%d.png"%(i))

        idx_h, idx_w = np.where(roi_mask[0]==1)
        nocs_pts = roi_coord[0, :, idx_h, idx_w]
        save_to_ply(nocs_pts,  "./output/roi_nocs_%d.ply"%i)


        idx_h, idx_w = np.where(ref_mask[0]==1)
        nocs_pts_ref = ref_coord[0, :, idx_h, idx_w]
        save_to_ply(nocs_pts_ref,  "./output/ref_nocs_%d.ply"%i)
# Last modified: 2024-03-11
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------
from tqdm import tqdm

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch.nn import SmoothL1Loss, L1Loss
from scipy.spatial.transform import Rotation as R
from models.model import One2Any
from utils.aligning import estimateSimilarityTransform

from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions
from utils.aligning import pose_estimation_from_nocs, save_to_ply, save_array_to_image, get_symmetry_transformations
import os
import cv2
import time
from utils.evaluator import Evaluator

def get_scale_from_rt(rt_matrix):
    """
    Given a 4x4 numpy array representing [R|t], this function returns the scale of the rotation matrix R.
    
    :param rt_matrix: 4x4 numpy array representing [R|t]
    :return: scale factor of the rotation matrix
    """
    # Ensure the input is a 4x4 matrix
    if rt_matrix.shape != (4, 4):
        raise ValueError("The input matrix must be a 4x4 transformation matrix")
    
    # Extract the 3x3 rotation matrix R from the 4x4 matrix
    R = rt_matrix[:3, :3]
    
    # Compute the norm of each column of the rotation matrix R
    col_norms = [np.linalg.norm(R[:, i]) for i in range(3)]
    
    # Compute the average scale
    scale = np.mean(col_norms)
    
    return scale

def normalize_rotation_matrix(R_matrix):
    """
    Normalize a 3x3 rotation matrix by removing any scale using SVD.
    
    :param R_matrix: 3x3 numpy array representing the rotation matrix (possibly scaled)
    :return: normalized 3x3 rotation matrix
    """
    U, _, Vt = np.linalg.svd(R_matrix)
    return np.dot(U, Vt)

def transform_coordinates_3d(coordinates, RT):
    """
    Input: 
        coordinates: [3, N]
        RT: [4, 4]
    Return 
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def draw(img, axes, color, imgpts=None, axis_i=None):
    img = cv2.line(img, tuple(axes[0]), tuple(axes[1]), color, 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[3]), color, 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[2]), color, 3)
    if axis_i is not None:
        img = cv2.line(img, tuple(axis_i[0]), tuple(axis_i[1]), (125, 0, 0), 3)
        img = cv2.line(img, tuple(axis_i[0]), tuple(axis_i[3]), (125, 125, 0), 3)
        img = cv2.line(img, tuple(axis_i[0]), tuple(axis_i[2]), (40, 60, 80), 3)

    return img

def draw_pose(image, axis_pts, pts_w_cur2next, camera_K, output_path):
    projected_axes = calculate_2d_projections(axis_pts.transpose(), camera_K)
    bbox = pts_w_cur2next.reshape(-1, 3)
    bbox_proj_2d = calculate_2d_projections(bbox.transpose(), camera_K)
    draw_image = draw(image, projected_axes, (255, 255, 0))
    cv2.imwrite(output_path, draw_image[:, :, ::-1])
    return draw_image

def draw_pose_from_axis(image, camera_K, camera_ex, camera_ex_gt, output_path):
    xyz_axis = 0.1 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
    transformed_axes = transform_coordinates_3d(xyz_axis, camera_ex)
    projected_axes = calculate_2d_projections(transformed_axes, camera_K)
    
    transformed_axes_gt = transform_coordinates_3d(xyz_axis, camera_ex_gt)
    projected_axes_gt = calculate_2d_projections(transformed_axes_gt, camera_K)
    draw_image = draw(image, projected_axes_gt, (193, 182, 255))
    draw_image = draw(draw_image, projected_axes, (144, 238, 144))
    cv2.imwrite(output_path, draw_image[:, :, ::-1])
    return draw_image

def draw_pts(image, pts, color):
    for i in range(len(pts)):
        pt = pts[i]
        cv2.circle(image, pt, 1, color, -1)
    return image

def draw_pts_projection(image, camera_K, camera_ex, camera_ex_gt, mesh, output_path):   
    transformed_axes = transform_coordinates_3d(mesh.transpose(), camera_ex)
    projected_axes = calculate_2d_projections(transformed_axes, camera_K)
    
    transformed_axes_gt = transform_coordinates_3d(mesh.transpose(), camera_ex_gt)
    projected_axes_gt = calculate_2d_projections(transformed_axes_gt, camera_K)

    draw_image = draw_pts(image, projected_axes, (144, 238, 144))
    draw_image = draw_pts(draw_image, projected_axes_gt, (193, 182, 255))
    cv2.imwrite(output_path, draw_image[:, :, ::-1])
    return draw_image

def main():
    opt = TestOptions()
    args = opt.initialize().parse_args()

    args.gpu = 'cuda:0'
    args.rank = 0
    device = torch.device(args.gpu)

    model = One2Any(args=args)
    cudnn.benchmark = True
    model.to(device)
    model_weight = torch.load(args.ckpt_dir)['model']
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight, strict=False)
    model.eval()

    # Dataset setting
    dataset_kwargs = {
        'dataset_name': args.dataset,
        'data_path': args.data_path,
        'data_name': args.data_name,
        'data_type': args.data_val,
        'num_view': 50, 
    }
    dataset_kwargs['scale_size'] = args.scale_size

    dataset = get_dataset(**dataset_kwargs, is_train=False)
    loader = torch.utils.data.DataLoader(dataset, 
                                        batch_size=1,
                                        shuffle=False, 
                                        num_workers=0, 
                                        pin_memory=True, 
                                        drop_last=False)

    evaluator = Evaluator(args.data_path, compute_vsd=True)
    all_poses = {}
    all_poses['gt_pose'] = []
    all_poses['pred_pose'] = []

    criterion_o = SmoothL1Loss(beta=0.1)
    for batch_idx, batch in enumerate(tqdm(loader)): 
        input_RGB = batch['image'].to(device)
        input_MASK = batch['mask'].to(device).to(bool)
        input_depth = batch['orig_depth'].to(device)
        pcl_c = batch['roi_pcl'].to(device)
        roi_class = batch['roi_class']
        instance_id = batch['instance_id']

        ref_data = batch['ref_data']
        ref_image = ref_data['ref_rgb'].to(device)
        ref_mask = ref_data['ref_mask'].to(device).unsqueeze(1).int().float()
        ref_scale_matrix = ref_data['ref_scale_matrix'].to(device)
        ref_nocs = ref_data['ref_nocs'].to(device).float()
        ref_cond = torch.cat([ref_image, ref_nocs, ref_mask], dim=1).to(device)
        ref_K = ref_data['ref_K'].to(device).float()

        nocs = batch['nocs'].to(device).permute(0, 2, 3, 1)
        gt_pose = batch['nocs_pose'].to(device)
        ref_pose = ref_data['ref_pose'].to(device)
        
        if args.visualize:
            orig_RGB = batch['orig_image'].to(device)
            orig_ref_rgb = ref_data['orig_ref_rgb'].to(device)
        with torch.no_grad():
            preds = model(input_RGB, input_MASK, ref_cond=ref_cond)

        pred_nocs = preds['pred_nocs'].permute(0, 2, 3, 1)
        
        dis_sym = batch['dis_sym'].to(device)
        con_sym = batch['con_sym'].to(device)

        gt_min_pose_list = []
        pred_nocs_list = []
        gt_nocs_list = []
        for b in range(batch['image'].shape[0]):
            curr_pred_nocs = pred_nocs[b]
            curr_gt_nocs = nocs[b]
            curr_mask = input_MASK[b]
            curr_pred_nocs = curr_pred_nocs[curr_mask]
            curr_gt_nocs = curr_gt_nocs[curr_mask]
            curr_pcl_c = pcl_c[b, curr_mask]
            curr_pcl_m = curr_gt_nocs - 0.5
            # discrete symmetry
            curr_dis_sym = dis_sym[b]
            dis_sym_flag = torch.sum(torch.abs(curr_dis_sym), dim=(1, 2)) != 0
            curr_dis_sym = curr_dis_sym[dis_sym_flag]
            curr_con_sym = con_sym[b]
            con_sym_flag = torch.sum(torch.abs(curr_con_sym), dim=(-1)) != 0
            curr_con_sym = curr_con_sym[con_sym_flag]
            aug_pcl_m = torch.stack([curr_pcl_c], dim=0)
            
            curr_gt_pose = gt_pose[b].float()
            cur_ref_pose = ref_pose[b].float()
            cur_ref_scale_matrix = ref_scale_matrix[b].float()

            sym_pose, cur_gt_pose_list, cur_gt_pose_for_nocs = get_symmetry_transformations(curr_dis_sym, curr_con_sym, gt_pose=curr_gt_pose, ref_pose=cur_ref_pose, ref_scale_matrix=cur_ref_scale_matrix)
            cur_gt_pose_for_nocs = torch.from_numpy(cur_gt_pose_for_nocs).to(device).float()
            aug_pcl_m = curr_pcl_c.unsqueeze(0).repeat(len(cur_gt_pose_for_nocs), 1, 1)
            cur_gt_pose_list = torch.from_numpy(cur_gt_pose_list).to(device).float()
            aug_pcl_m = torch.bmm(cur_gt_pose_for_nocs[:, :3, :3], aug_pcl_m.permute((0, 2, 1))).permute((0, 2, 1)) + cur_gt_pose_for_nocs[:, :3, 3].unsqueeze(1)
            aug_pcl_m[0] = curr_pcl_m

            curr_gt_nocs_set = aug_pcl_m + 0.5
            with torch.no_grad():
                curr_gt_nocs_set = torch.unbind(curr_gt_nocs_set, dim=0)
                loss_tmp = list(map(lambda gt_nocs: criterion_o(curr_pred_nocs, gt_nocs), curr_gt_nocs_set))
                min_idx = torch.argmin(torch.tensor(loss_tmp))
            curr_gt_nocs = curr_gt_nocs_set[min_idx]
            curr_gt_min_pose = cur_gt_pose_list[min_idx]
            pred_nocs_list.append(curr_pred_nocs)
            gt_nocs_list.append(curr_gt_nocs)
            gt_min_pose_list.append(curr_gt_min_pose)
        
        gt_min_pose_list = torch.stack(gt_min_pose_list, dim=0)
        
        pred_pose_2 = pose_estimation_from_nocs(pred_nocs, pcl_c,
                                               input_MASK, ref_K, ref_scale_matrix)   
        pred_pose = torch.from_numpy(pred_pose_2).to(ref_pose.device)
        
        ref_nocs = ref_nocs.permute(0, 2, 3, 1)
        for k in range(len(pred_nocs_list)):
            idx_h, idx_w = np.where(input_MASK[k].detach().cpu().numpy()==1)
            cur_pred_nocs_pts = pred_nocs_list[k].detach().cpu().numpy()
            cur_gt_nocs_pts = gt_nocs_list[k].detach().cpu().numpy()
            cur_gt_nocs_orig = nocs[k].detach().cpu().numpy()
            gt_nocs_min = np.zeros_like(pred_nocs[k].detach().cpu().numpy())
            gt_nocs_min[idx_h, idx_w] = gt_nocs_list[k].detach().cpu().numpy()
            if args.visualize:
                cur_ref_image = orig_ref_rgb[k].detach().cpu().numpy()
                cur_ref_nocs = ref_nocs[k].detach().cpu().numpy()
                idx_h_n, idx_w_n = np.where(ref_mask[k, 0].detach().cpu().numpy()==1)
                save_to_ply(cur_ref_nocs[idx_h_n, idx_w_n], "./nocs_vis/%s_ref.ply"%(instance_id[k]))
                save_to_ply(cur_pred_nocs_pts, "./nocs_vis/%s_pred.ply"%(instance_id[k]))
                save_to_ply(cur_gt_nocs_pts, "./nocs_vis/%s_gt.ply"%(instance_id[k]))
                save_to_ply(cur_gt_nocs_orig[idx_h, idx_w], "./nocs_vis/%s_gt_min.ply"%(instance_id[k]))
                pred_nocs_vis = np.zeros_like(cur_gt_nocs_orig)
                pred_nocs_vis[idx_h, idx_w] = cur_pred_nocs_pts
                save_array_to_image(np.clip(pred_nocs_vis, 0, 1),  "./nocs_vis/%s_pred.png"%(instance_id[k]))
                save_array_to_image(np.clip(gt_nocs_min, 0, 1),  "./nocs_vis/%s_gt_min.png"%(instance_id[k]))
                save_array_to_image(np.clip(cur_ref_nocs, 0, 1),  "./nocs_vis/%s_ref_nocs.png"%(instance_id[k]))
                save_array_to_image(cur_ref_image,  "./nocs_vis/%s_ref_rgb.png"%(instance_id[k]))
                save_array_to_image(orig_RGB[k].detach().cpu().numpy(),  "./nocs_vis/%s_rgb.png"%(instance_id[k]))
                save_array_to_image(input_RGB[k].detach().cpu().numpy(),  "./nocs_vis/%s_rgb_resized.png"%(instance_id[k]))

            pred_pose_ = np.dot(pred_pose[k].detach().cpu().numpy(), ref_pose[k].detach().cpu().numpy())
            
            iou_a = torch.ones(1).to(pred_pose[k].device)
            iou_q = torch.ones(1).to(pred_pose[k].device)
            gt_pose_ = gt_pose[k].float() @ ref_pose[k].float()
            cls_id = roi_class[k]
            pred_q = torch.from_numpy(pred_pose_).to(gt_pose_.device).float()
            evaluator.register_test({
                'iou_a' : iou_a.unsqueeze(0),
                'iou_q' : iou_q.unsqueeze(0),
                'gt_pose': gt_pose_.unsqueeze(0),
                'pred_pose': pred_q.unsqueeze(0),
                'pred_pose_rel': pred_pose.unsqueeze(0),
                'cls_id': [cls_id],
                'camera': [ref_K[k].detach().cpu().numpy()],
                'depth': [input_depth[k].detach().cpu().numpy()],
                'instance_id': [instance_id],
            })

            if args.visualize:
                frame_name = instance_id[k]
                save_path = os.path.join("./nocs_vis/", frame_name +"_pred_pose_only.png")
                draw_image = draw_pose_from_axis(orig_RGB[k].detach().cpu().numpy().copy(), ref_K[k].detach().cpu().numpy(), pred_pose_, gt_pose_.detach().cpu().numpy(), save_path)


    evaluator.test_summary()
    print(evaluator.get_latex_str())

if __name__ == '__main__':
    main()
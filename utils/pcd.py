import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import numpy as np
from typing import Tuple, Optional
from torch import Tensor
from torch.nn.functional import cosine_similarity
from .misc import torch_sample_select
from torch.distributions import Categorical
from utils import coordinates
import cv2


def get_diameter(pcd : np.ndarray) -> float:

    xyz = pcd[:,:3]
    maxs, mins = np.max(xyz,axis=0), np.min(xyz,axis=0)
    return max(maxs-mins)

def pdist(A, B, dist_type='L2'):
    if dist_type == 'L2':
        D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
        return torch.sqrt(D2 + 1e-7)
    elif dist_type == 'SquareL2':
        return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
    elif dist_type == 'inv_norm_cosine':
        return 0.5 * (-1*cosine_similarity(A.unsqueeze(1), B.unsqueeze(0),dim=2) + 1)
    elif dist_type == 'cosine':
        return 0.5 * (cosine_similarity(A.unsqueeze(1), B.unsqueeze(0),dim=2) + 1)
    else:
        raise NotImplementedError('Not implemented')

def lift_pcd(depth : Tensor, camera : Tensor, xy_idxs : Optional[Tuple]=None):

    '''
    Given a depth image and relative camera, lifts the depth to a point cloud.
    If depth has 4 channel, the last 3 are used as RGB and an RGB point cloud is produced in output.
    Image size is implicitly given as depth image size.
    Optionally a set of xy coordinates can be passed to lift only these points
    '''

    H, W, D = depth.shape

    d = depth[:,:,0].to(torch.float32)

    if xy_idxs is not None:
        xmap = xy_idxs[0].to(d.device).long()
        ymap = xy_idxs[1].to(d.device).long()
        pt2 = d[ymap, xmap]
        xmap = xmap.to(torch.float32)
        ymap = ymap.to(torch.float32)
    else:
        # make coordinate grid
        xs = torch.linspace(0, W-1, steps=W)
        ys = torch.linspace(0, H-1, steps=H)
        xmap, ymap = torch.meshgrid(xs,ys, indexing='xy')

        xmap = xmap.flatten().to(d.device).to(torch.float32)
        ymap = ymap.flatten().to(d.device).to(torch.float32)
        pt2 = d.flatten()

    # get camera info
    fx = camera[0]
    fy = camera[4]
    cx = camera[2]
    cy = camera[5]
    # perform lifting
    pt0 = (xmap - cx) * pt2 / fx
    pt1 = (ymap - cy) * pt2 / fy
    pcd_depth = torch.stack((pt0, pt1, pt2),dim=1) 
    
    if D > 1:
        feats = depth[ymap.long(),xmap.long(),1:]
        if xy_idxs is None:
            feats = feats.reshape(H*W,D-1)
        pcd_depth = torch.cat([pcd_depth, feats], dim=1)
    return pcd_depth

def get_pcd_bbox(pcd : Tensor, margin : float = 0. ) -> Tensor:

    min_x, max_x = torch.min(pcd[:, 0]) - margin, torch.max(pcd[:, 0]) + margin
    min_y, max_y = torch.min(pcd[:, 1]) - margin, torch.max(pcd[:, 1]) + margin
    min_z, max_z = torch.min(pcd[:, 2]) - margin, torch.max(pcd[:, 2]) + margin

    bbox_3d = Tensor([
        [min_x,min_y,min_z],
        [max_x,min_y,min_z],
        [min_x,max_y,min_z],
        [max_x,max_y,min_z],
        [min_x,min_y,max_z],
        [max_x,min_y,max_z],
        [min_x,max_y,max_z],
        [max_x,max_y,max_z]
    ])

    return bbox_3d

def crop_pcd(pcd : Tensor, bbox : Tensor) -> Tensor:

    max_x, min_x = torch.max(bbox[:,0]), torch.min(bbox[:,0])
    max_y, min_y = torch.max(bbox[:,1]), torch.min(bbox[:,1])
    max_z, min_z = torch.max(bbox[:,2]), torch.min(bbox[:,2])

    idx_x = torch.bitwise_and(pcd[:,0] <= max_x, pcd[:,0] >= min_x)
    idx_y = torch.bitwise_and(pcd[:,1] <= max_y, pcd[:,1] >= min_y)
    idx_z = torch.bitwise_and(pcd[:,2] <= max_z, pcd[:,2] >= min_z)

    mask = torch.all(torch.stack((idx_x, idx_y, idx_z),dim=1),dim=1)

    return mask

def torch_transform_pcd(pcd : Tensor, r : Tensor, t : Tensor) -> Tensor:
    '''
    Rotates a batch of (B,N,3) point clouds
    '''
    pcd = pcd.to(torch.double).transpose(1,2)
    r = r.to(torch.double)
    t = t.to(torch.double)
    rotated = torch.bmm(r, pcd) + t.unsqueeze(2)
    
    return rotated.transpose(1,2).to(torch.float)

def np_transform_pcd(pcd: np.ndarray, r : np.ndarray, t: np.ndarray) -> np.ndarray:
    
    pcd = pcd.astype(np.float16)
    r = r.astype(np.float16)
    t = t.astype(np.float16)
    rot_pcd = np.dot(np.asarray(pcd), r.T) + t
    return rot_pcd 

def compute_add(pcd : np.ndarray, pred_pose : np.ndarray, gt_pose : np.ndarray) -> np.ndarray:

        pred_r, pred_t = pred_pose[:3,:3], pred_pose[:3,3]
        gt_r, gt_t = gt_pose[:3,:3], gt_pose[:3,3]

        model_pred = np_transform_pcd(pcd, pred_r, pred_t)
        model_gt = np_transform_pcd(pcd, gt_r, gt_t)

        # ADD computation
        add = np.mean(np.linalg.norm(model_pred - model_gt, axis=1))

        return add

def project_points(v : np.ndarray, k : np.ndarray) -> np.ndarray:
    if len(v.shape) == 1:
        v = np.expand_dims(v, 0)
    
    assert len(v.shape) == 2, '  wrong dimension, expexted shape 2.'
    assert v.shape[1] == 3, ' expected 3d points, got ' + str(v.shape[0]) + ' ' + str(v.shape[1]) +'d points instead.' 

    k = k.astype(np.float16)
    p = np.matmul(k, v.T)
    p[0] = p[0] / (p[2] + 1e-6)
    p[1] = p[1] / (p[2] + 1e-6)
    
    return p[:2].T

def sample_pcd(pcd : Tensor, n_points : int) -> Tensor:
    
    '''
    Performs pcd subsampling
    '''

    pcd_points = pcd.shape[0]

    #if pcd_points > n_points:
    uniform_dist = torch.ones(pcd_points, dtype=torch.float).to(pcd.device)
    inds_choosen = torch.multinomial(uniform_dist, n_points, replacement=True)
    pcd = pcd[inds_choosen]

    return pcd

def nn_correspondences(feats1: Tensor, feats2: Tensor, mask1: Tensor, mask2: Tensor, threshold: float, max_corrs: int, subsample_source:int, corrs_device: str) -> Tensor:
    '''
    Finds matches between two [D,H,W] feature maps
    Return correspondences in shape (y1,x1,y2,x2)
    '''

    orig_device = feats1.device
    roi1 = torch.nonzero(mask1 == 1).to(corrs_device)
    roi2 = torch.nonzero(mask2 == 1).to(corrs_device)

    if subsample_source is not None:
        if roi1.shape[0] > subsample_source:
            idxs = torch_sample_select(roi1, subsample_source)
            roi1 = roi1[idxs]

    roi_feats1 = feats1[:, roi1[:, 0], roi1[:, 1]].T.to(corrs_device)
    roi_feats2 = feats2[:, roi2[:, 0], roi2[:, 1]].T.to(corrs_device)
    # reduce memory usage on gpu
    if corrs_device == 'cuda':
        roi_feats1 = roi_feats1.to(torch.float16).cuda()
        roi_feats2 = roi_feats2.to(torch.float16).cuda()
    else:
        roi_feats1 = roi_feats1.to(torch.float32).cpu()
        roi_feats2 = roi_feats2.to(torch.float32).cpu()
        
    dist = pdist(roi_feats1, roi_feats2, 'inv_norm_cosine')
    min_dist = torch.amin(dist, dim=1)
    ro12_idxs = torch.argmin(dist, dim=1)
    valid_corr = torch.nonzero(min_dist < threshold)
    if valid_corr.shape[0] > 1:
        # get roi2 choosen as minimum
        roi2 = roi2[ro12_idxs]
        final_corrs = torch.cat((roi1[valid_corr.squeeze(1)], roi2[valid_corr.squeeze(1)]), dim=1)

        idxs = torch_sample_select(final_corrs, max_corrs)
        final_corrs = final_corrs[idxs].to(orig_device)
    else:
        final_corrs = None
    # these are in format (y1,x1,y2,x2)
    return final_corrs


def nn_correspondences_anchors(feats1: Tensor, feats2: Tensor, gt_corrs: Tensor, mask2: Tensor, device: str) -> Tensor:
    '''
    Finds matches between two [D,H,W] feature maps, starting from a set of predefined anchors on the first feature map
    Only coordinates of A from gt_corrs is used!
    Return correspondences in shape (y1,x1,y2,x2)
    '''
    orig_device = feats1.device
    roi1 = gt_corrs[:,:2].to(device)
    roi2 = torch.nonzero(mask2 == 1).to(device)

    roi_feats1 = feats1[:, roi1[:, 0], roi1[:, 1]].T.to(device)
    roi_feats2 = feats2[:, roi2[:, 0], roi2[:, 1]].T.to(device)
        
    dist = pdist(roi_feats1, roi_feats2, 'inv_norm_cosine')
    ro12_idxs = torch.argmin(dist, dim=1)
    # get roi2 choosen as minimum
    roi2 = roi2[ro12_idxs]
    final_corrs = torch.cat([roi1, roi2], dim=1)

    # these are in format (y1,x1,y2,x2)
    return final_corrs.to(orig_device)
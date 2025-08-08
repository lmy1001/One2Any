import os
import sys
import math
import torch
import numpy as np
from tqdm import tqdm
sys.path.append(os.getcwd())
from .pcd import np_transform_pcd, pdist
from torch.nn.functional import cosine_similarity
from sklearn.neighbors import KDTree
from typing import Tuple
from torch import Tensor
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from utils.misc import unique_matches
from torch.distributions import Categorical
from scipy.spatial import cKDTree

def mask_iou(mask1: Tensor, mask2: Tensor) -> Tensor:

    '''
    Both input as [B,H,W]
    '''

    assert mask1.shape == mask2.shape
    assert len(mask1.shape) == 3

    B, H, W = mask1.shape

    mask1 = mask1.view(B, H*W)
    mask2 = mask2.view(B, H*W)

    union = torch.logical_or(mask1, mask2)
    inters = torch.logical_and(mask1, mask2)

    union_areas = union.sum(1)
    inter_areas = inters.sum(1)

    ious = inter_areas / union_areas

    return ious


def get_entropy(probs: torch.Tensor, dim:int, norm:bool=False) -> torch.Tensor:
    '''
    Compute entropy along a given dimension, optionally normalizing it between 0 and 1
    probs is supposed to be a distribution along the given dimension
    '''

    entropy = (-1 * torch.mul(probs, torch.log(probs+1e-12))).sum(dim)

    if norm:
        size = probs.shape[dim]
        uniform_dist = Categorical(torch.ones(size)/size)
        max_entropy = uniform_dist.entropy()
        entropy = entropy / max_entropy
    
    return entropy

def compute_fmr(feats1 : Tensor, feats2 : Tensor, dist_th: float, inlier_th: float):
    '''
    Compute FMR between to correspondence sets (B,N,D), (B,N,D)
    '''
    assert feats1.shape == feats2.shape
    
    if len(feats1.shape) == 2:
        # handles unbatched case
        feats1, feats2 = feats1.unsqueeze(0), feats2.unsqueeze(0)

    # use inverted cosine similarity as distance metric
    dist_pos = .5 * (-1*F.cosine_similarity(feats1, feats2, dim=2) + 1)
    # ratio of inlier (i.e. pairs under a distance in the feature space) of each corr set
    inlier_ratio = ((dist_pos < dist_th).to(float)).mean(1)

    # mean os success/failure for each corr set 
    recall = (inlier_ratio > inlier_th).to(float)

    return recall

def pixel_match_loss(gt_matches: torch.Tensor, pred_matches: torch.Tensor) -> torch.Tensor:

    '''
    Return pixel match loss between two batches of predicted and ground truth matches (B,N,4)
    '''


    err = torch.zeros(pred_matches.shape[0])
    for i in range(pred_matches.shape[0]):
        gt_i, pred_i = gt_matches[i].cpu(), pred_matches[i]
        if gt_i.shape[0] > 0 and pred_i.shape[0] > 0:
            dists = (pdist(pred_i[:,:2], gt_i[:,:2]) + pdist(pred_i[:,2:], gt_i[:,2:])) / 2.
            r,c = linear_sum_assignment(dists)
            err[i] = dists[r,c].mean()
        else: 
            err[i] = 0.

    return err

def box_iou(box1 : np.ndarray, box2: np.ndarray) -> float:
    # determine the (x, y)-coordinates of the intersection rectangle
    
    # switch from (x,y,w,h) to (x1,y1,x2,y2)
    box1[2] = box1[2] + box1[0]
    box1[3] = box1[3] + box1[1]
    box2[2] = box2[2] + box2[0]
    box2[3] = box2[3] + box2[1]

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
    boxBArea = abs((box2[2] - box2[0]) * (box2[3] - box2[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def np_mask_iou(m1 : np.ndarray, m2: np.ndarray) -> float:

    m1, m2 = m1.astype(bool), m2.astype(bool)
    union = np.logical_or(m1,m2).sum()
    inter = np.logical_and(m1,m2).sum()

    iou = float(inter) / float(union)

    return iou

def class_cosine_similarty(feats_dict : dict) -> np.ndarray:

    '''
    Given a dict of N classes, each a Tensor [P,D] with P number of elements and D dimension,
    compute similarity scores between classes using normalized cosine similarity.
    '''

    keys = list(feats_dict.keys())
    N_CLS = len(keys)
    mat_sim = torch.zeros((len(keys), len(keys)))
    for i1, obj1 in enumerate(keys):
        
        for i2, obj2 in enumerate(keys):

            if obj1 == obj2:
                cs = self_cosine_similarity(feats_dict[obj1])
                N_ELEM = feats_dict[obj1].shape[0]
                avg_sim = (cs.sum() - N_CLS) / (N_ELEM*(N_ELEM-1))
            else:
                avg_sim = cross_cosine_similarity(feats_dict[obj1], feats_dict[obj2]).mean()
            mat_sim[i1, i2] = avg_sim

    # move cosine similariti to a [0,1] score
    mat_sim = (mat_sim + 1) / 2.

    return mat_sim.numpy()

def self_cosine_similarity(x : torch.Tensor) -> torch.Tensor:
    '''
    Given a [N,D] vector, returns [N,N] matrix with cosine similarity
    '''

    N = x.shape[0]

    c_matrix = torch.zeros((N,N))

    for i in range(N):
        c_matrix[i,:] = cosine_similarity(x, x[i,:])
    
    return c_matrix

def cross_cosine_similarity(x1 : torch.Tensor, x2 : torch.Tensor) -> torch.Tensor:
    '''
    Given two vectors [N1,D] and [N2,D], returns [N1,N2] cosine similarity matrix
    '''

    N1,N2 = x1.shape[0], x2.shape[0]
    assert x1.shape[1] == x2.shape[1]

    c_matrix = torch.zeros((N1,N2))
    for i in range(N2):
        c_matrix[i,:] = cosine_similarity(x1, x2[i,:])

    return c_matrix

def compute_adi(pcd : np.ndarray, pred_pose : np.ndarray, gt_pose : np.ndarray) -> np.ndarray:
    pred_r, pred_t = pred_pose[:3,:3], pred_pose[:3,3]
    gt_r, gt_t = gt_pose[:3,:3], gt_pose[:3,3]

    model_pred = np_transform_pcd(pcd, pred_r, pred_t)
    model_gt = np_transform_pcd(pcd, gt_r, gt_t)

    nn_index = cKDTree(model_pred)
    nn_dists, _ = nn_index.query(model_gt, k=1, workers=-1)
    e = nn_dists.mean()
    return e

def compute_add(pcd : np.ndarray, pred_pose : np.ndarray, gt_pose : np.ndarray) -> np.ndarray:

        pred_r, pred_t = pred_pose[:3,:3], pred_pose[:3,3]
        gt_r, gt_t = gt_pose[:3,:3], gt_pose[:3,3]

        model_pred = np_transform_pcd(pcd, pred_r, pred_t)
        model_gt = np_transform_pcd(pcd, gt_r, gt_t)

        # ADD computation
        add = np.mean(np.linalg.norm(model_pred - model_gt, axis=1))

        return add

def compute_adds(pcd : np.ndarray, pred_pose : np.ndarray, gt_pose : np.ndarray) -> np.ndarray:

        pred_r, pred_t = pred_pose[:3,:3], pred_pose[:3,3]
        gt_r, gt_t = gt_pose[:3,:3], gt_pose[:3,3]

        model_pred = np_transform_pcd(pcd, pred_r, pred_t)
        model_gt = np_transform_pcd(pcd, gt_r, gt_t)
        
        # ADD-S computation
        kdt = KDTree(model_gt, metric='euclidean')
        distance, _ = kdt.query(model_pred, k=1)
        adds = np.mean(distance)
        
        return adds

def compute_RT_distances(pose1 : np.ndarray, pose2 : np.ndarray):
    '''
    :param RT_1: [B, 4, 4]. homogeneous affine transformation
    :param RT_2: [B, 4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter
    Works in batched or unbatched manner. NB: assumes that translations are in Meters
    '''

    if pose1 is None or pose2 is None:
        return -1

    if len(pose1.shape) == 2:
        pose1 = np.expand_dims(pose1,axis=0)
        pose2 = np.expand_dims(pose2,axis=0)

    try:
        assert np.array_equal(pose1[:, 3, :], pose2[:, 3, :])
        assert np.array_equal(pose1[0, 3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(pose1[:, 3, :], pose2[:, 3, :])


    BS = pose1.shape[0]

    R1 = pose1[:, :3, :3] / np.cbrt(np.linalg.det(pose1[:, :3, :3]))[:,None,None]
    T1 = pose1[:, :3, 3]

    R2 = pose2[:, :3, :3] / np.cbrt(np.linalg.det(pose2[:, :3, :3]))[:,None,None]
    T2 = pose2[:, :3, 3]
    
    R = np.matmul(R1,R2.transpose(0,2,1))
    arccos_arg = (np.trace(R,axis1=1, axis2=2) - 1)/2
    arccos_arg = np.clip(arccos_arg, -1+1e-12, 1-1e-12)
    theta = np.arccos(arccos_arg) * 180/np.pi
    theta[np.isnan(theta)] = 180.
    shift = np.linalg.norm(T1-T2,axis=-1) * 100

    return theta, shift

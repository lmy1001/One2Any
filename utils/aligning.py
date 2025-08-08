'''
Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
RANSAC for Similarity Transformation Estimation

Written by Srinath Sridhar
'''

import numpy as np
import cv2
import itertools
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
import math
import torch.nn.functional as F

def get_inverse_pose_torch(pose):
    pose_inv_R = torch.linalg.inv(pose[:3, :3])
    pose_inv_t = -1 * pose_inv_R @ pose[:3, 3]
    pose_inv = torch.eye(4).to(pose.device)
    pose_inv[:3, :3] = pose_inv_R
    pose_inv[:3, 3] = pose_inv_t
    return pose_inv 

def compute_nocs_list(pose_list, pcl_c, input_MASK, ref_scale_matrix, ref_pose):
    nocs_pose = np.dot(curr_gt_pose, get_inverse_pose(ref_pose))       
    nocs_list = []
    idx_h, idx_w = np.where(input_MASK==1)
    for i in range(len(pose_list)):
        pcl_c[input_MASK]
    mask_3d = transform_coordinates_3d(pcl_c[idx_h, idx_w].transpose(), get_inverse_pose(nocs_pose)).transpose()
    mask_3d = transform_coordinates_3d(mask_3d.transpose(), ref_scale_matrix).transpose()
    mask_3d += 0.5
    nocs_gt = np.zeros_like(pcl_c)
    nocs_gt[idx_h, idx_w] = mask_3d
    return nocs_gt

def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
    
def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction."""
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def get_symmetry_transformations(dis_sym, con_sym, max_sym_disc_step=0.05, gt_pose=None, ref_pose=None, ref_scale_matrix=None):
    # Discrete symmetries.
    dis_sym = dis_sym.detach().cpu().numpy()
    con_sym = con_sym.detach().cpu().numpy()

    trans_disc = [{"R": np.eye(3), "t": np.array([0, 0, 0]).T}]  # Identity.
    if gt_pose is not None:
        gt_pose = gt_pose.detach().cpu().numpy()
        gt_pose_list = [gt_pose]
        if ref_pose is not None:
            ref_pose = ref_pose.detach().cpu().numpy()
            ref_scale_matrix = ref_scale_matrix.detach().cpu().numpy()
            gt_pose_for_nocs_list = [gt_pose]
    for k in range(len(dis_sym)):
        if np.array_equal(dis_sym[k], np.zeros((4, 4))):
            continue
        sym = dis_sym[k]
        sym_4x4 = np.reshape(sym, (4, 4))
        R = sym_4x4[:3, :3]
        t = sym_4x4[:3, 3]
        trans_disc.append({"R": R, "t": t})

    # Discretized continuous symmetries.
    trans_cont = []
    for k in range(len(con_sym)):
        if np.array_equal(con_sym[k], np.zeros(6)):
            continue
        axis = con_sym[k, :3]
        offset = con_sym[k, 3:]

        # (PI * diam.) / (max_sym_disc_step * diam.) = discrete_steps_count
        discrete_steps_count = int(np.ceil(np.pi / max_sym_disc_step))

        # Discrete step in radians.
        discrete_step = 2.0 * np.pi / discrete_steps_count

        for i in range(0, discrete_steps_count):
            R = rotation_matrix(i * discrete_step, axis)[:3, :3]
            t = -R.dot(offset) + offset
            trans_cont.append({"R": R, "t": t})

    # Combine the discrete and the discretized continuous symmetries.
    trans = []
    for tran_disc in trans_disc:
        if len(trans_cont):
            for tran_cont in trans_cont:
                R = tran_cont["R"].dot(tran_disc["R"])
                t = tran_cont["R"].dot(tran_disc["t"]) + tran_cont["t"]
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = t
                if gt_pose is not None:
                    cur_gt_pose = gt_pose @ ref_pose @ pose
                    cur_gt_pose = cur_gt_pose @ get_inverse_pose(ref_pose)
                    gt_pose_list.append(cur_gt_pose)
                    if ref_pose is not None:
                        cur_gt_pose_nocs = get_inverse_pose(cur_gt_pose)
                        cur_gt_pose_nocs = ref_scale_matrix @ cur_gt_pose_nocs
                        gt_pose_for_nocs_list.append(cur_gt_pose_nocs)
                trans.append(pose)
        else:
            pose = np.eye(4)
            pose[:3, :3] = tran_disc['R']
            pose[:3, 3] = tran_disc['t']
            if gt_pose is not None:
                cur_gt_pose = gt_pose @ ref_pose @ pose
                cur_gt_pose = cur_gt_pose @ get_inverse_pose(ref_pose)
                gt_pose_list.append(cur_gt_pose)
                if ref_pose is not None:
                    cur_gt_pose_nocs = get_inverse_pose(cur_gt_pose)
                    cur_gt_pose_nocs = ref_scale_matrix @ cur_gt_pose_nocs
                    gt_pose_for_nocs_list.append(cur_gt_pose_nocs)
        trans.append(pose)

    trans = np.stack(trans, axis=0)
    if gt_pose is not None:
        gt_pose_list = np.stack(gt_pose_list, axis=0)
        if ref_scale_matrix is not None:
            gt_pose_for_nocs_list = np.stack(gt_pose_for_nocs_list, axis=0)
            return trans, gt_pose_list, gt_pose_for_nocs_list
        else:
            trans, gt_pose_list
    return trans

def rotation_6d_to_matrix(rot_6d):
    """
    Given a 6D rotation output, calculate the 3D rotation matrix in SO(3) using the Gramm Schmit process

    For details: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf
     """

    bs,_  = rot_6d.shape
    rot_6d = rot_6d.view(-1, 6)
    m1 = rot_6d[:, 0:3]
    m2 = rot_6d[:, 3:6]

    x = F.normalize(m1, p=2, dim=1)
    z = torch.cross(x, m2, dim=1)
    z = F.normalize(z, p=2, dim=1)
    y = torch.cross(z, x, dim=1)
    rot_matrix = torch.cat((x.view(-1, 3, 1), y.view(-1, 3, 1), z.view(-1, 3, 1)), 2)  # Rotation Matrix lying in the SO(3)
    rot_matrix = rot_matrix.view(bs, 3, 3)
    return rot_matrix

def to_homo(pts):
  '''
  @pts: (N,3 or 2) will homogeneliaze the last dimension
  '''
  assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
  homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
  return homo


def adi_err(pred,gt,model_pts):
  """
  @pred: 4x4 mat
  @gt:
  @model: (N,3)
  """
  pred_pts = (pred@to_homo(model_pts).T).T[:,:3]
  gt_pts = (gt@to_homo(model_pts).T).T[:,:3]
  nn_index = cKDTree(pred_pts)
  nn_dists, _ = nn_index.query(gt_pts, k=1, workers=-1)
  e = nn_dists.mean()
  return e

def add_err(pred,gt,model_pts):
  """
  Average Distance of Model Points for objects with no indistinguishable views
  - by Hinterstoisser et al. (ACCV 2012).
  """
  pred_pts = (pred@to_homo(model_pts).T).T[:,:3]
  gt_pts = (gt@to_homo(model_pts).T).T[:,:3]
  e = np.linalg.norm(pred_pts - gt_pts, axis=1).mean()
  return e

def compute_auc(rec, max_val=0.1):
  '''https://github.com/wenbowen123/iros20-6d-pose-tracking/blob/2df96b720e8e499b9f0d5fcebfbae2bcfa51ab19/eval_ycb.py#L45
  '''
  if len(rec)==0:
    return 0
  rec = np.sort(np.array(rec))
  n = len(rec)
  prec = np.arange(1,n+1) / float(n)
  rec = rec.reshape(-1)
  prec = prec.reshape(-1)
  index = np.where(rec<max_val)[0]
  rec = rec[index]
  prec = prec[index]
  if len(rec) == 0:
    return 0
  mrec=[0, *list(rec), max_val]
  mpre=[0, *list(prec), prec[-1]]

  for i in range(1,len(mpre)):
    mpre[i] = max(mpre[i], mpre[i-1])
  mpre = np.array(mpre)
  mrec = np.array(mrec)
  i = np.where(mrec[1:]!=mrec[0:len(mrec)-1])[0] + 1
  ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) / max_val
  return ap

def save_array_to_image(array, save_name):
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    if array.max() <= 1.0:
        if array.min() < 0.0:
            array = (array +1.0) / 2.0 * 255.0
        elif array.min() >= 0.0:
            array = array * 255

    if array.shape[0] == 3 or array.shape[0] == 1:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = array.squeeze()
        image = Image.fromarray(array.astype(np.uint8), mode='L')
    else:
        image = Image.fromarray(array.astype(np.uint8))
    image.save(save_name)

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

def get_inverse_pose(pose):
    pose_inv_R = np.linalg.inv(pose[:3, :3])
    pose_inv_t = -np.dot(pose_inv_R, pose[:3, 3])
    pose_inv = np.eye(4)
    pose_inv[:3, :3] = pose_inv_R
    pose_inv[:3, 3] = pose_inv_t
    return pose_inv 

def get_pts_in_w(pts, depth, camera, pose=None):
    K = camera
    pts_ = pts.astype(np.uint32)
    kp_depths = depth[pts_[:, 1], pts_[:, 0], None]
    kps_3d = np.einsum(
                "ij,pj->pi",
                np.linalg.inv(K[:3, :3]),
                np.pad(pts_[:, :2], ((0, 0), (0, 1)), constant_values=1),
            ) * kp_depths[None]
    
    if pose is None:
        return kps_3d[0]
    kps_3d = np.einsum(
            "ij,pj->pi",
            np.linalg.inv(pose)[:3],
            np.pad(kps_3d, ((0, 0), (0, 1)), constant_values=1),
    )
    kps_3d = kps_3d[0]
    return kps_3d

def get_pts_mask_in_w(cur_mask, cur_depth, cur_camera):
    cur_mask_grid_h, cur_mask_grid_w = np.where(cur_mask==1)
    pts_mask_cur = np.zeros((len(cur_mask_grid_h), 2))
    pts_mask_cur[:, 1] = cur_mask_grid_h
    pts_mask_cur[:, 0] = cur_mask_grid_w
    pts_mask_cur_w = get_pts_in_w(pts_mask_cur, cur_depth, cur_camera)
    return pts_mask_cur_w

def save_to_ply(pts, save_path):
    points_np = np.array([(point[0], point[1], point[2]) for point in pts],
                     dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])


    vertex = PlyElement.describe(points_np, 'vertex')

    ply_data = PlyData([vertex])
    ply_data.write(save_path)

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([1, 4], dtype=np.int32)
    for i in range(1):
        m = mask[:, :]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        w = abs(x2-x1)
        h = abs(y2-y1)
        boxes = np.array([x1, y1, w, h])
    return boxes.astype(np.int32)

def depth2xyzmap(depth, K):
    invalid_mask = (depth<0.1)
    H,W = depth.shape[:2]
    vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
    zs = depth.reshape(-1)
    xs = (us-K[0,2])*zs/K[0,0]
    ys = (vs-K[1,2])*zs/K[1,1]
    pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
    xyz_map = pts.reshape(H,W,3).astype(np.float32)
    xyz_map[invalid_mask] = 0
    return xyz_map.astype(np.float32)


def zoom_out(im, mask_orig, res=480, interpolate=cv2.INTER_NEAREST): 
    img_orig = np.zeros((mask_orig.shape[0], mask_orig.shape[1], 3))
    x,y,w,h = extract_bboxes(mask_orig)
    center = np.array((x+0.5*w, y+0.5*h))
    base_ratio=1.5
    wh_max=480
    wh = max(w, h) * base_ratio
    if wh_max != None:
        wh = min(wh, wh_max)
    c_w, c_h = center
    c_w, c_h, s, res = int(c_w), int(c_h), int(wh), int(res)
    max_h, max_w = mask_orig.shape[0:2]
    crop_min_h, crop_min_w = max(0, c_h - s // 2), max(0, c_w - s // 2)
    crop_max_h, crop_max_w = min(max_h, c_h + s // 2), min(max_w, c_w + s // 2)
    up = s // 2 - (c_h - crop_min_h)
    down = s // 2 + (crop_max_h-c_h)
    left = s // 2 - (c_w - crop_min_w)
    right = s // 2 + (crop_max_w - c_w)
    im_deresize = cv2.resize(im, (s, s), interpolation=interpolate)
    img_orig[crop_min_h:crop_max_h, crop_min_w:crop_max_w] = im_deresize[up:down, left:right]
    return img_orig

def pose_estimation_from_pnp(nocs, input_MASK, camera, scale_matrix):
    nocs = nocs.detach().cpu().numpy()
    input_MASK = input_MASK.detach().cpu().numpy()
    camera = camera.detach().cpu().numpy()
    scale_matrix = scale_matrix.detach().cpu().numpy()
    pred_poses = np.zeros((nocs.shape[0], 4, 4))

    for i in range(input_MASK.shape[0]):
        pred_nocs = nocs[i] - 0.5  # [-0.5, 0.5]
        idx_h, idx_w = np.where(input_MASK[i] == 1)
        pred_nocs_coord = pred_nocs[idx_h, idx_w]
        pred_nocs_coord = transform_coordinates_3d(pred_nocs_coord.transpose(),
                                                   get_inverse_pose(scale_matrix[i])).transpose()

        points2D = np.stack((idx_w, idx_h), axis=1)

        # Solve PnP to estimate the pose
        if len(points2D) < 4:
            pred_pose =np.eye(4)
            pred_poses[i] = pred_pose
        else:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=pred_nocs_coord,
                imagePoints=points2D.astype(np.float32),
                cameraMatrix=camera[i],
                distCoeffs=np.zeros(5),
                reprojectionError=2.0,
                iterationsCount=100,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                for reprojectionthres in [5.0, 10.0, 15.0, 20.0]:
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        objectPoints=pred_nocs_coord,
                        imagePoints=points2D.astype(np.float32),
                        cameraMatrix=camera[i],
                        distCoeffs=np.zeros(5),
                        reprojectionError=reprojectionthres,
                        iterationsCount=100,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    if success:
                        break
            _, refined_rvec, refined_tvec = cv2.solvePnP(
                objectPoints=pred_nocs_coord,
                imagePoints=points2D.astype(np.float32),
                cameraMatrix=camera[i],
                distCoeffs=np.zeros(5),
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            rotation_matrix, _ = cv2.Rodrigues(refined_rvec)
            pred_pose = np.eye(4)
            pred_pose[:3, :3] = rotation_matrix
            pred_pose[:3, 3] = refined_tvec[:, 0]
            pred_poses[i] = pred_pose
    return pred_poses

def pose_estimation_from_nocs(nocs, input_pcl, input_MASK, camera, scale_matrix):
    nocs = nocs.detach().cpu().numpy()
    input_pcl = input_pcl.detach().cpu().numpy()
    input_MASK = input_MASK.detach().cpu().numpy()
    camera = camera.detach().cpu().numpy()
    scale_matrix =scale_matrix.detach().cpu().numpy()
    pred_poses = np.zeros((nocs.shape[0], 4, 4))
    for i in range(input_MASK.shape[0]):
        pred_nocs = nocs[i] - 0.5 #[-0.5, 0.5]
        idx_h, idx_w = np.where(input_MASK[i]==1)
        pred_nocs_coord = pred_nocs[idx_h, idx_w]
        pred_nocs_coord = transform_coordinates_3d(pred_nocs_coord.transpose(), get_inverse_pose(scale_matrix[i])).transpose()
        pts_w_cur = input_pcl[i, idx_h, idx_w]
        if len(pts_w_cur) < 5:
            pred_poses[i][:3, :3] = np.eye(3)

        else:
            Scales, Rotation, Translation, pred_pose = estimateSimilarityTransform(pred_nocs_coord, pts_w_cur, no_scale=True)
            if Scales is None:
                pred_pose = np.eye(4)
                pred_poses[i] = pred_pose
            else:
                pred_pose[:3, :3] = pred_pose[:3, :3].transpose()
                pred_poses[i] = pred_pose
    return pred_poses

def estimateSimilarityTransform(source: np.array, target: np.array, verbose=False, no_scale=False):
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))

    # Auto-parameter selection based on source-target heuristics
    TargetNorm = np.mean(np.linalg.norm(target, axis=1))
    SourceNorm = np.mean(np.linalg.norm(source, axis=1))
    RatioTS = (TargetNorm / SourceNorm)
    RatioST = (SourceNorm / TargetNorm)
    PassT = RatioST if(RatioST>RatioTS) else RatioTS
    StopT = PassT / 100
    nIter = 100
    if verbose:
        print('Pass threshold: ', PassT)
        print('Stop threshold: ', StopT)
        print('Number of iterations: ', nIter)

    SourceInliersHom, TargetInliersHom, BestInlierRatio = getRANSACInliers(SourceHom, TargetHom, MaxIterations=nIter, PassThreshold=PassT, StopThreshold=StopT, no_scale=no_scale)

    if(BestInlierRatio < 0.1):
        print('[ WARN ] - Something is wrong. Small BestInlierRatio: ', BestInlierRatio)
        SourceInliersHom = SourceHom
        TargetInliersHom = TargetHom
        return None, None, None, None

    Scales, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(SourceInliersHom, TargetInliersHom, no_scale=no_scale)

    if verbose:
        print('BestInlierRatio:', BestInlierRatio)
        print('Rotation:\n', Rotation)
        print('Translation:\n', Translation)
        print('Scales:', Scales)

    return Scales, Rotation, Translation, OutTransform

def estimateRestrictedAffineTransform(source: np.array, target: np.array, verbose=False):
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))

    RetVal, AffineTrans, Inliers = cv2.estimateAffine3D(source, target)
    # We assume no shear in the affine matrix and decompose into rotation, non-uniform scales, and translation
    Translation = AffineTrans[:3, 3]
    NUScaleRotMat = AffineTrans[:3, :3]
    # NUScaleRotMat should be the matrix SR, where S is a diagonal scale matrix and R is the rotation matrix (equivalently RS)
    # Let us do the SVD of NUScaleRotMat to obtain R1*S*R2 and then R = R1 * R2
    R1, ScalesSorted, R2 = np.linalg.svd(NUScaleRotMat, full_matrices=True)

    if verbose:
        print('-----------------------------------------------------------------------')
    # Now, the scales are sort in ascending order which is painful because we don't know the x, y, z scales
    # Let's figure that out by evaluating all 6 possible permutations of the scales
    ScalePermutations = list(itertools.permutations(ScalesSorted))
    MinResidual = 1e8
    Scales = ScalePermutations[0]
    OutTransform = np.identity(4)
    Rotation = np.identity(3)
    for ScaleCand in ScalePermutations:
        CurrScale = np.asarray(ScaleCand)
        CurrTransform = np.identity(4)
        CurrRotation = (np.diag(1 / CurrScale) @ NUScaleRotMat).transpose()
        CurrTransform[:3, :3] = np.diag(CurrScale) @ CurrRotation
        CurrTransform[:3, 3] = Translation
        # Residual = evaluateModel(CurrTransform, SourceHom, TargetHom)
        Residual = evaluateModelNonHom(source, target, CurrScale,CurrRotation, Translation)
        if verbose:
            # print('CurrTransform:\n', CurrTransform)
            print('CurrScale:', CurrScale)
            print('Residual:', Residual)
            print('AltRes:', evaluateModelNoThresh(CurrTransform, SourceHom, TargetHom))
        if Residual < MinResidual:
            MinResidual = Residual
            Scales = CurrScale
            Rotation = CurrRotation
            OutTransform = CurrTransform

    if verbose:
        print('Best Scale:', Scales)

    if verbose:
        print('Affine Scales:', Scales)
        print('Affine Translation:', Translation)
        print('Affine Rotation:\n', Rotation)
        print('-----------------------------------------------------------------------')

    return Scales, Rotation, Translation, OutTransform

def getRANSACInliers(SourceHom, TargetHom, MaxIterations=100, PassThreshold=200, StopThreshold=1, no_scale=False):
    BestResidual = 1e10
    BestInlierRatio = 0
    BestInlierIdx = np.arange(SourceHom.shape[1])
    for i in range(0, MaxIterations):
        # Pick 5 random (but corresponding) points from source and target
        RandIdx = np.random.randint(SourceHom.shape[1], size=5)
        _, _, _, OutTransform = estimateSimilarityUmeyama(SourceHom[:, RandIdx], TargetHom[:, RandIdx], no_scale=no_scale)
        Residual, InlierRatio, InlierIdx = evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold)
        if Residual < BestResidual:
            BestResidual = Residual
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
        if BestResidual < StopThreshold:
            break

    return SourceHom[:, BestInlierIdx], TargetHom[:, BestInlierIdx], BestInlierRatio

def evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold):
    Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    InlierIdx = np.where(ResidualVec < PassThreshold)
    nInliers = np.count_nonzero(InlierIdx)
    InlierRatio = nInliers / SourceHom.shape[1]
    return Residual, InlierRatio, InlierIdx[0]

def evaluateModelNoThresh(OutTransform, SourceHom, TargetHom):
    Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual

def evaluateModelNonHom(source, target, Scales, Rotation, Translation):
    RepTrans = np.tile(Translation, (source.shape[0], 1))
    TransSource = (np.diag(Scales) @ Rotation @ source.transpose() + RepTrans.transpose()).transpose()
    Diff = target - TransSource
    ResidualVec = np.linalg.norm(Diff, axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual

def testNonUniformScale(SourceHom, TargetHom):
    OutTransform = np.matmul(TargetHom, np.linalg.pinv(SourceHom))
    ScaledRotation = OutTransform[:3, :3]
    Translation = OutTransform[:3, 3]
    Sx = np.linalg.norm(ScaledRotation[0, :])
    Sy = np.linalg.norm(ScaledRotation[1, :])
    Sz = np.linalg.norm(ScaledRotation[2, :])
    Rotation = np.vstack([ScaledRotation[0, :] / Sx, ScaledRotation[1, :] / Sy, ScaledRotation[2, :] / Sz])
    print('Rotation matrix norm:', np.linalg.norm(Rotation))
    Scales = np.array([Sx, Sy, Sz])

    return Scales, Rotation, Translation, OutTransform

def estimateSimilarityUmeyama(SourceHom, TargetHom, no_scale=False):
    # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]

    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()

    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints

    if np.isnan(CovMatrix).any():
        print('nPoints:', nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        raise RuntimeError('There are NANs in the input.')

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]

    Rotation = np.matmul(U, Vh).T # Transpose is the one that works

    varP = np.var(SourceHom[:3, :], axis=1).sum()
    ScaleFact = 1/varP * np.sum(D) # scale factor
    Scales = np.array([ScaleFact, ScaleFact, ScaleFact])
    ScaleMatrix = np.diag(Scales)
    if no_scale:
        Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(Rotation)
        OutTransform = np.identity(4)
        OutTransform[:3, :3] = Rotation
        OutTransform[:3, 3] = Translation
    else:
        Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(ScaleFact*Rotation)
        OutTransform = np.identity(4)
        OutTransform[:3, :3] = ScaleMatrix @ Rotation
        OutTransform[:3, 3] = Translation

    return Scales, Rotation, Translation, OutTransform

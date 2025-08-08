import os
import sys
sys.path.append(os.getcwd())
import json
from .metrics import *
import pickle
from typing import Optional
from bop_toolkit_lib.pose_error import my_mssd, my_mspd, vsd
from bop_toolkit_lib.misc import format_sym_set, get_symmetry_transformations
from utils.pcd import get_diameter
from utils.misc import sorted_alphanumeric
import numpy as np
from plyfile import PlyData
from torch import Tensor
from typing import Any, Dict, Tuple, Sequence, Union, List


def get_obj_rendering(root: str, obj_id: int) -> dict:
    '''
    returns object usable for vispy rendering
    argument obj_model is expected to have the following fields:
      - pts : (N,3) xyz points in mm
      - normals: (N,3) normals
      - faces: (M,3) polygon faces needed for rendering 
    '''

    pcd = PlyData.read(os.path.join(root,'obj_{:06d}.ply'.format(obj_id)))
    # these are already in mm    
    xs = pcd['vertex']['x']
    ys = pcd['vertex']['y']
    zs = pcd['vertex']['z']
    if 'nx' in pcd['vertex']:
        nxs = pcd['vertex']['nx']
        nys = pcd['vertex']['ny']
        nzs = pcd['vertex']['nz']

    raw_vertexs = np.asarray(pcd['face']['vertex_indices'])
    faces = np.stack([vert for vert in raw_vertexs],axis=0)

    xyz = np.stack((xs,ys,zs), axis=1)
    if 'nc' in pcd['vertex']:
        normals = np.stack((nxs,nys,nzs), axis=1)
    
        return {
        'pts': xyz,
        'normals': normals,
        'faces': faces
        }
    else:
        return {
        'pts': xyz,
        'faces': faces
        }

def get_obj_rendering_nocs(root: str, obj_id: str) -> dict:
    '''
    returns object usable for vispy rendering
    argument obj_model is expected to have the following fields:
      - pts : (N,3) xyz points in mm
      - normals: (N,3) normals
      - faces: (M,3) polygon faces needed for rendering 
    '''

    pts, normals, faces = list(), list(), list()
    basepath = os.path.join(root,'obj_models','real_test', obj_id)

    with open(basepath + '_vertices.txt') as f:
        lines = [line.split(' ') for line in f.readlines()]
        for line in lines:
            pts.append([float(line[0]), float(line[1]), float(line[2])])

    with open(basepath + '_normals.txt') as f:
        lines = [line.split(' ') for line in f.readlines()]
        for line in lines:
            normals.append([float(line[0]), float(line[1]), float(line[2])])

    with open(basepath + '.obj') as f:
        lines = [line.split(' ')[1:] for line in f.readlines() if line.startswith('f')]
        for line in lines:
            f1, f2, f3 = int(line[0].split('/')[0]),int(line[1].split('/')[0]),int(line[2].split('/')[0])
            faces.append([f1,f2,f3])

    return {
        'pts': np.asarray(pts) * 1000,
        'normals': np.asarray(normals),
        'faces': np.asarray(faces),
    }

def process_tensor(t : Tensor) -> Tensor:
    '''
    Processes tensor
    '''

    return t.clone().detach().cpu()

def dict_from_list(instance_list_file):
    '''
    Produces a dictionary from a partition instance list
    '''    
    instances = dict()
    
    with open(instance_list_file,'r') as f:
        preds_lines = f.readlines()

    for line in preds_lines:
        part, id_a, id_q, cls_name = line.split(',')
        scene_a, img_a = id_a.strip(' ').split(' ') 
        scene_q, img_q = id_q.strip(' ').split(' ')
        
        _, cls_name = cls_name.strip(' ').split(' ')
        cls_name = cls_name.strip('\n')
        new_id_a = '{}_{}_{}'.format(scene_a, img_a, cls_name)
        new_id_q = '{}_{}_{}'.format(scene_q, img_q, cls_name)
        instances[new_id_a] = part
        instances[new_id_q] = part

    return instances

def dict_from_gt(gt_file):
    '''
    Formats a ground truth file to a common representation
    '''    

    new_gt = dict()
    f = open(gt_file,'rb')
    gt = pickle.load(f)

    for k,v in gt.items():
        tokens = k.split('_')
        cls_name = '_'.join(tokens[5:])
        scene_a, img_a, scene_q, img_q = tokens[:4]
        new_id = '{}_{}_{}_{}_{}'.format(scene_a, img_a, scene_q, img_q, cls_name)
        v['gt'][:3,3] = v['gt'][:3,3] / 1000.
        new_gt[new_id] = v
    
    return new_gt

def dict_from_preds(perf_file):
    '''
    Produces a dictionart from a prediction file
    '''    
    preds = dict()
    
    with open(perf_file,'r') as f:
        preds_lines = f.readlines()

    for line in preds_lines:
        id_a, id_q, pose_line = line.split(',')
        scene_a, img_a, obj_a = id_a.split(' ') 
        scene_q, img_q, _ = id_q.split(' ')
        pose = np.asarray([float(p) for p in pose_line.split(' ')]).reshape(3,4)
        new_id = '{}_{}_{}_{}_{}'.format(scene_a, img_a, scene_q, img_q, obj_a)
        
        preds[new_id] = pose
    return preds


class Evaluator(object):
    '''
    Helper class used to evaluate pose metrics
    '''
    def __init__(self, data_path: str, compute_vsd: bool=True, compute_iou: bool=True):
        
        super().__init__()
        self.exp_tag = "./results/"
        self.data_path = data_path
        self.mssd_rec = np.arange(0.05, 0.51, 0.05)
        self.mspd_rec = np.arange(5, 51, 5)
        self.compute_vsd = compute_vsd
        self.compute_iou = compute_iou

        if self.compute_vsd:
            from bop_toolkit_lib.renderer_vispy import RendererVispy
            if 'tless' in data_path:
                self.renderer = RendererVispy(720, 540, mode='depth') #for tless data, it's 720, 540
            else:
                self.renderer = RendererVispy(640, 480, mode='depth')
            self.vsd_taus = list(np.arange(0.05, 0.51, 0.05))
            self.vsd_rec = np.arange(0.05, 0.51, 0.05)
            self.vsd_delta = 15.

        # used to compute pose recall from error in rotation and translation
        self.pose_recall_th = [(5,5),(10,5),(10,10)]
        self.metrics = {}
        if 'toyl' in data_path:
            obj_models, obj_diams, obj_symms = self.get_obj_data(self.data_path, model_name='models_bop')
        if 'nocs' in data_path:
            obj_models, obj_diams, obj_symms = self.get_obj_data_nocs(self.data_path)
        else:
            obj_models, obj_diams, obj_symms = self.get_obj_data(self.data_path, model_name='./models_eval/')
        self.add_object_info(obj_models, obj_diams, obj_symms)
        self.init_test()
    
    def get_obj_data(self, root: str, model_name='models_bop'):# -> Tuple[dict,dict,dict]:

        '''
        Returns complete information about a dataset point clouds
        '''
        obj_data_dir = os.path.join(root, model_name)
        obj_files = [file for file in os.listdir(obj_data_dir) if '.ply' in file]
        obj_models, obj_diams, obj_symm = dict(), dict(), dict()
        
        with open(os.path.join(obj_data_dir,'models_info.json')) as f:
            models_info = json.load(f)

        for obj_file in obj_files:

            obj_id = int(os.path.splitext(obj_file[4:])[0])
            model_info = models_info[str(obj_id)]
        
            obj_models[obj_id] = get_obj_rendering(obj_data_dir, obj_id)
            obj_diams[obj_id] = model_info['diameter']
            obj_symm[obj_id] = get_symmetry_transformations(model_info, max_sym_disc_step=0.05)
        
        return (obj_models, obj_diams, obj_symm)

    def get_obj_data_nocs(self, root: str):# -> Tuple[dict,dict,dict]:

        '''
        Returns complete information about a dataset point clouds
        '''
        obj_models, obj_diams, obj_symm = dict(), dict(), dict()

        obj_data_dir = os.path.join(root, 'obj_models','real_test')
        with open(os.path.join(obj_data_dir,'models_info.json')) as f:
            models_info = json.load(f)
    
        for obj_name, model_info in models_info.items():
            obj_models[obj_name] = get_obj_rendering_nocs(root, obj_name) #get_obj_pcd(root, split, obj_file)
            obj_diams[obj_name] = model_info['diameter']
            obj_symm[obj_name] = get_symmetry_transformations(model_info, max_sym_disc_step=0.05)
        
        return (obj_models, obj_diams, obj_symm)

    def add_object_info(self, obj_models: dict, obj_diams: dict, obj_symms: dict):
        # these are supposed to be in mm!
        self.obj_models = obj_models
        self.obj_diams = obj_diams
        self.obj_symms = {k: format_sym_set(sym_set) for k, sym_set in obj_symms.items()}

        if self.compute_vsd:
            for obj_id, obj in self.obj_models.items():
                self.renderer.my_add_object(obj, obj_id)

    def get_obj_info(self, obj_id):
        '''
        Get object ID
        '''
        return self.obj_models[obj_id], self.obj_diams[obj_id], self.obj_symms[obj_id]

    
    def clear(self):
        '''
        Remove all entries
        '''
        self.metrics = {}
        self.counts = {}

    def init_training(self):
        '''
        Initializes dict for metric storage at training time (only FMR)
        '''
        self.clear()
        if self.compute_iou:
            self.metrics['Anchor IoU'] = []
            self.metrics['Query IoU'] = []
            self.metrics['Mean IoU'] = []
            self.metrics['IoU > .25'] = []
            self.metrics['IoU > .5'] = []
            self.metrics['IoU > .75'] = []

    def init_validation(self):
        '''
        Initializes dict for metric storage at validation time (only FMR, ADD, FMR, pose recall)
        '''

        self.init_training()
        # init metrics
        self.metrics['R error'] = []
        self.metrics['T error'] = []
        self.metrics['ADD(S)-0.1d'] = []
        self.metrics['ADD-AUC'] = []
        self.metrics['ADI-AUC'] = []

        # BOP metrics
        if self.compute_vsd:
            self.metrics['AR'] = []
            self.metrics['VSD'] = []
        self.metrics['MSSD'] = []
        self.metrics['MSPD'] = []
        
        # init counts
        self.counts['Missing segm'] = []
        self.counts['Failed pose'] = []
        self.counts['Zero pose'] = []

        for r_th, t_th in self.pose_recall_th:
            self.metrics[f'Recall ({r_th}deg, {t_th}cm)'] = []

    def init_test(self):
        '''
        Initializes dict for metric storage at validation time (all validation + instance ids)
        '''
        self.init_validation()
        self.metrics['instance_id'] = []
        self.metrics['cls_id'] = []


    def register_train(self, results: dict, clear: bool=False):
        '''
        Register metrics at training time (only FMR)
        '''

        if clear:
            self.clear()
            self.init_training()
        
        if self.compute_iou:

            if 'iou_a' in results.keys() and 'iou_q' in results.keys():
                mean_iou = (results['iou_a'] + results['iou_q']) / 2.
                mean_iou = process_tensor(mean_iou).numpy()
            elif 'iou' in results.keys():
                mean_iou = process_tensor(results['iou']).numpy()
            
            iou_a = process_tensor(results['iou_a']).numpy()
            iou_q = process_tensor(results['iou_q']).numpy()
            
            self.metrics['Anchor IoU'].extend(iou_a.tolist())
            self.metrics['Query IoU'].extend(iou_q.tolist())
            self.metrics['Mean IoU'].extend(mean_iou.tolist())
            self.metrics['IoU > .25'].extend((mean_iou > 0.25).astype(int).tolist())
            self.metrics['IoU > .5'].extend((mean_iou > 0.5).astype(int).tolist())
            self.metrics['IoU > .75'].extend((mean_iou > 0.75).astype(int).tolist())


    def register_eval(self, results: dict, clear: bool=False):
        '''
        Register metrics at validation time (FMR, pose errors, ADD)
        '''

        self.register_train(results, clear) # this registers FMR
        pred_poses = process_tensor(results['pred_pose']).numpy()
        gt_poses = process_tensor(results['gt_pose']).numpy()
        pred_poses_rel = process_tensor(results['pred_pose_rel']).numpy()

        for idx, pred_pose_rel in enumerate(pred_poses_rel):
            # work on counts
            # if this function is called, something has been segmented
            self.counts['Missing segm'].append(0)
            # happens for some reason               
            zero_pose = int(np.count_nonzero(pred_pose_rel) <= 1)
            # pose fails when predictions is identity
            failed_pose = int((pred_pose_rel == np.eye(4)).all())
            self.counts['Failed pose'].append(failed_pose)
            self.counts['Zero pose'].append(zero_pose)

            if zero_pose == 1:
                pred_poses[idx] = np.eye(4)

        # compute error in pose
        err_R, err_T = compute_RT_distances(pred_poses, gt_poses)
        self.metrics['R error'].extend(err_R.tolist())
        self.metrics['T error'].extend(err_T.tolist())

        # iterate over different pose recalls
        for r_th, t_th in self.pose_recall_th:
            succ_r, succ_t = err_R <= r_th, err_T <= t_th
            succ_pose = np.logical_and(succ_r, succ_t).astype(float)
            self.metrics[f'Recall ({r_th}deg, {t_th}cm)'].extend(succ_pose.tolist())

        # iterate over pose and compute ADD-0.1d
        for cls_id, pred_pose, pred_pose_rel, gt_pose, camera, depth, instance_id in zip(results['cls_id'], pred_poses, pred_poses_rel, gt_poses, results['camera'], results['depth'], results['instance_id']):
            obj_model, obj_diam, obj_sym = self.get_obj_info(int(cls_id)) #nocs: str(), toyl: int()

            add_diam = obj_diam / 1e3
            if obj_sym.shape[0] > 1:
                adds = compute_adds(obj_model['pts']/1000.0, pred_pose, gt_pose)
            else:
                adds = compute_add(obj_model['pts']/1000.0, pred_pose, gt_pose)

            adi = compute_adi(obj_model['pts'] / 1000., pred_pose, gt_pose)
            add = compute_add(obj_model['pts']/1000.0, pred_pose, gt_pose)

            self.metrics['ADD(S)-0.1d'].append(float(adds <= add_diam*0.1))

            self.metrics['ADD-AUC'].append(float(add))
            self.metrics['ADI-AUC'].append(float(adi))

            pred_pose, gt_pose = pred_pose.astype(np.float16), gt_pose.astype(np.float16)
            
            pred_r, pred_t = pred_pose[:3,:3], np.expand_dims(pred_pose[:3,3],axis=1) * 1000
            gt_r, gt_t = gt_pose[:3,:3], np.expand_dims(gt_pose[:3,3],axis=1) * 1000

            # compute BOP metrics
            mspd_err = my_mspd(pred_r, pred_t, gt_r, gt_t, camera.reshape(3,3), obj_model['pts'], obj_sym)
            mssd_err = my_mssd(pred_r, pred_t, gt_r, gt_t, obj_model['pts'], obj_sym)

            # MSSD recalls depends on object diameters
            # MSPD instead is fixed, as it depends on the image size
            mssd_cur_rec = self.mssd_rec * obj_diam
            mean_mssd = (mssd_err < mssd_cur_rec).mean()
            mean_mspd = (mspd_err < self.mspd_rec).mean()
            self.metrics['MSSD'].append(mean_mssd)
            self.metrics['MSPD'].append(mean_mspd)

            if self.compute_vsd:
                depth *= 1e3 #has to be align with translation
                vsd_errs = vsd(pred_r, pred_t, gt_r, gt_t, depth, camera.reshape(3,3), self.vsd_delta, self.vsd_taus, True, obj_diam, self.renderer, int(cls_id)) #int(toyl), str() nocs
                vsd_errs = np.asarray(vsd_errs)
                all_vsd_recs = np.stack([vsd_errs < rec_i  for rec_i in self.vsd_rec],axis=1)
                mean_vsd = all_vsd_recs.mean()
                self.metrics['VSD'].append(mean_vsd)
                self.metrics['AR'].append((mean_mssd + mean_mspd + mean_vsd)/3.)

    def register_test(self, results: dict, clear: bool=False):
        self.register_eval(results, clear)
        self.metrics['cls_id'].extend(results['cls_id'])
        self.metrics['instance_id'].extend(results['instance_id'])

    def register_valid_failure(self, results):
        '''
        This is used to register an automatic failure due to wrong detection
        '''
        self.metrics['R error'].append(0.)
        self.metrics['T error'].append(0.)
        self.metrics['ADD(S)-0.1d'].append(0.)
        self.metrics['ADI-0.1d'].append(0.)
        
        if self.compute_vsd:
            self.metrics['VSD'].append(0.)
            self.metrics['AR'].append(0.)
        self.metrics['MSSD'].append(0.)
        self.metrics['MSPD'].append(0.)
        
        if self.compute_iou:
            iou_a = process_tensor(results['iou_a']).numpy()
            iou_q = process_tensor(results['iou_q']).numpy()
            
            self.metrics['Anchor IoU'].extend(iou_a.tolist())
            self.metrics['Query IoU'].extend(iou_q.tolist())
            
            self.metrics['Mean IoU'].append(0.)
            self.metrics['IoU > .25'].append(0.)
            self.metrics['IoU > .5'].append(0.)
            self.metrics['IoU > .75'].append(0.)

        self.counts['Missing segm'].append(1)
        self.counts['Failed pose'].append(0)
        self.counts['Zero pose'].append(0)

        #self.metrics[f'FMR_{self.fmr_dist_th}_{self.fmr_inlier_th}'].extend([-1]) 
        for r_th, t_th in self.pose_recall_th:
            self.metrics[f'Recall ({r_th}deg, {t_th}cm)'].extend([0])

    def get_auc(self, values, idxs, max_val=0.1):
        rec = np.asarray(values)[idxs]
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

    def register_test_failure(self, results: dict):
        '''
        This is used to register an automatic failure due to wrong detection
        '''
        self.register_valid_failure(results)
        self.metrics['cls_id'].extend(results['cls_id'])
        self.metrics['instance_id'].extend(results['instance_id'])


    def test_summary(self):
        # # print result for each class
        classes = np.unique(self.metrics['cls_id'])
        for cls_id in classes.tolist():
            means = self.get_obj_means(cls_id)
        
            if self.compute_vsd:
                latex_str = f"{cls_id} & {means['AR']*100:.1f} & {means['VSD']*100:.1f} & {means['MSSD']*100:.1f} & {means['MSPD']*100:.1f} & {means['ADD(S)-0.1d']*100:.1f} & \
                                                   {means['Recall (5deg, 5cm)']} & {means['Recall (10deg, 5cm)']} & {means['Recall (10deg, 10cm)']} & {means['ADI-AUC']*100:.1f} & {means['ADD-AUC']*100:.1f}"
            else:
                latex_str = f"{cls_id} & - & - & {means['MSSD']*100:.1f} & {means['MSPD']*100:.1f} & {means['ADD(S)-0.1d']*100:.1f} &  \
                               {means['Recall (5deg, 5cm)']} & {means['Recall (10deg, 5cm)']} & {means['Recall (10deg, 10cm)']} & {means['ADI-AUC']*100:.1f} & {means['ADD-AUC']*100:.1f}"
            if self.compute_iou:
                latex_str += f" {means['Mean IoU']*100:.1f} \\\\"
            else:
                latex_str += " - \\\\"
            print(latex_str)
            
            

    def save(self, file):
        all_dict = dict()
        all_dict.update(self.metrics)
        all_dict.update(self.counts)
        json.dump(all_dict, file)

    def get_log_means(self):
        '''
        Returns mean of each metric registered so far.
        Used only for the wandb logger
        '''
        means = {}
        for name, value in self.metrics.items():
            if name not in ['cls_id', 'instance_id'] and len(value) > 0:
                mean = np.asarray(value).mean()
                means[name] = mean

        return means

    def get_means(self):
        '''
        Returns mean of each metric registered so far
        '''
        means = {}
        for name, value in self.metrics.items():
            if name not in ['cls_id', 'instance_id'] and len(value) > 0:
                mean = np.asarray(value).mean()
                means[name] = mean
        return means

    def get_obj_means(self, cls_id):
        '''
        Returns mean of each metric registered so far, given an object id
        '''
        means = {}
        for name, value in self.metrics.items():
            if name not in ['cls_id', 'instance_id'] and len(value) > 0:
                idxs = np.asarray(self.metrics['cls_id']) == cls_id
                if name in ['ADI-AUC', 'ADD-AUC']:
                    means[name] = self.get_auc(self.metrics[name], idxs)
                else:
                    mean = np.asarray(value)[idxs].mean()
                    means[name] = mean
        return means

    def get_latex_str(self):
        '''
        Returns mean of each metric in format for a latex table
        '''

        means = self.get_means()
        if self.compute_vsd:
            latex_str = f"{self.exp_tag} & {means['AR']*100:.1f} & {means['VSD']*100:.1f} & {means['MSSD']*100:.1f} & {means['MSPD']*100:.1f} & {means['ADD(S)-0.1d']*100:.1f} &  \
                    {means['Recall (5deg, 5cm)']} & {means['Recall (10deg, 5cm)']} & {means['Recall (10deg, 10cm)']} & {means['ADI-AUC']*100:.1f} & {means['ADD-AUC']*100:.1f}"

        else:
            latex_str = f"{self.exp_tag} & - & - & {means['MSSD']*100:.1f} & {means['MSPD']*100:.1f} & {means['ADD(S)-0.1d']*100:.1f}  &  \
                {means['Recall (5deg, 5cm)']} & {means['Recall (10deg, 5cm)']} & {means['Recall (10deg, 10cm)']} & {means['ADI-AUC']*100:.1f} & {means['ADD-AUC']*100:.1f}"
            
        if self.compute_iou:
            latex_str += f" {means['Mean IoU']*100:.1f} \\\\ \n"
        else:
            latex_str += " - \\\\ \n"
        print(latex_str)
        return latex_str










    


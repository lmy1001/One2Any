import re
import logging
import numpy as np
from typing import Tuple, Sequence
import torch
from torch import Tensor
import os
from omegaconf import DictConfig
from typing import Union
from time import sleep
from datetime import datetime
from os import makedirs, listdir
from os.path import isdir, join
import subprocess

def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle: str, label=1, shape=(4,4)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)  # Needed to align to RLE direction


def rescale_recenter_coords(coords : Tensor, orig_scale : Sequence, new_scale : Sequence, crop_box : Sequence) -> Tensor:
    '''
    Maps coordinated in a crop back to the ones of the original images
    Returns a copy.
    
    orig_scale : H,W of current coordinates (domain of input coordinates)
    new_Scale : H,W of new coordinates (domain of output coordinates)
    crop_box : Y,X,H,W of original crop box
    coord_list : [N,2] list of YX coordinates
    '''
    new_coords = coords.clone()
    xs, ys = new_coords[:,1].to(torch.float32), new_coords[:,0].to(torch.float32)

    # obtain coordinate relative to the original crop size
    scaled_ys = (ys / orig_scale[0]) * crop_box[2]
    scaled_xs = (xs / orig_scale[1]) * crop_box[3]
    # add coordinate in image shape and clip for safety
    scaled_xs = torch.clip(scaled_xs + crop_box[1], 0, new_scale[1]-1)
    scaled_ys = torch.clip(scaled_ys + crop_box[0], 0, new_scale[0]-1)

    final_coords = torch.stack([scaled_ys, scaled_xs], dim=1)

    return final_coords

def crop_coords(coords : Tensor, crop_box : Union[Sequence, Tensor, np.ndarray], new_scale : Sequence) -> Tuple[Tensor,Tensor]:
    '''
    Maps coordinates from an image to a crop, which is then resized to a specific size.
    
    crop_box : Y,X,H,W of crop box
    new_scale : H,W of new coordinates (domain of output coordinates)
    coord_list : [N,2] list of YX coordinates
    '''
    new_coords = coords.clone()
    xs, ys = new_coords[:,1], new_coords[:,0]

    # obtain coordinate relative to the original crop size
    scaled_xs = ((xs - crop_box[1]) * (new_scale[1] / crop_box[3])).to(torch.int)
    scaled_ys = ((ys - crop_box[0]) * (new_scale[0] / crop_box[2])).to(torch.int)
    
    valid_xs = torch.logical_and(scaled_xs >= 0, scaled_xs < new_scale[1])
    valid_ys = torch.logical_and(scaled_ys >= 0, scaled_ys < new_scale[0])
    valid = torch.logical_and(valid_xs, valid_ys)

    final_coords = torch.stack([scaled_ys, scaled_xs], dim=1)

    return final_coords, valid


def rescale_coords(coords: Tensor, orig_scale: Tuple[int,int], new_scale: Tuple[int,int]) -> Tensor:
    '''
    Given a list of 2D coordinates, rescales them by the original and new scale.
    Coordinates are assumed to be in YX order, scales included.
    Returns a copy.
    Works in batch or single element.
    '''
    new_coords = coords.clone()
    to_squeeze = False
    if len(new_coords.shape) == 2:
        to_squeeze = True
        new_coords = new_coords.unsqueeze(0)

    assert new_coords.shape[-1] == 2 or new_coords.shape[-1] == 4, " works only with 2D keypoints or 2D correspondences"

    new_coords[:,:,0] = new_coords[:,:,0] * (new_scale[0] / orig_scale[0]) 
    new_coords[:,:,1] = new_coords[:,:,1] * (new_scale[1] / orig_scale[1])
    new_coords[:,:,0] = torch.clamp(new_coords[:,:,0], 0, new_scale[0]-1)
    new_coords[:,:,1] = torch.clamp(new_coords[:,:,1], 0, new_scale[1]-1)

    if new_coords.shape[-1] == 4:
        new_coords[:,:,2] = new_coords[:,:,2] * (new_scale[0] / orig_scale[0]) 
        new_coords[:,:,3] = new_coords[:,:,3] * (new_scale[1] / orig_scale[1])
        new_coords[:,:,2] = torch.clamp(new_coords[:,:,2], 0, new_scale[0]-1)
        new_coords[:,:,3] = torch.clamp(new_coords[:,:,3], 0, new_scale[1]-1)
    
    if to_squeeze:
        new_coords = new_coords.squeeze(0)

    return new_coords

def args2dict(config: DictConfig, init_key=None) -> dict:
    '''
    Recursively converts an Hydra configuration file in a one-level dictionary
    '''
    cur_dict = {}
    for k,v in config.items():

        if isinstance(v, DictConfig):
            if init_key:
                sub_dict = args2dict(v, init_key=f'{init_key}.{k}')
            else:
                sub_dict = args2dict(v, init_key=k)
            cur_dict.update(sub_dict)
        else:
            if init_key:
                cur_dict[f'{init_key}.{k}'] = v
            else:
                cur_dict[k] = v
    
    return cur_dict


def unique_matches(matches: torch.Tensor) -> torch.Tensor:

    '''
    Given a set of 2D-2D matches (Nx4), returns only unique matches
    '''

    assert matches.max() < 999, ' cannot handle indexes > 999'
    matches = matches.to(torch.int)
    match_set = set()
    for match in matches:
        match_code = f'{match[0]:03d}-{match[1]:03d}-{match[2]:03d}-{match[3]:03d}'
        match_set.add(match_code)

    match_list = list()
    for match in match_set:
        match_idxs = [int(m) for m in match.split('-')]
        match_list.append(match_idxs)
    
    return torch.tensor(match_list).to(torch.float)


def get_param_numbers(param_list : Sequence[Tensor]) -> int:
    '''
    Given a list of Tensors, considered a parameter lists, returns the number of parameters
    '''

    total_size = 0
    # iterate over tensor
    for param in param_list:
        
        param_size = 1
        # iterate over tensor size
        for n in list(param.shape):
            param_size *= n
        
        total_size += param_size

    return total_size


def set_deterministic_seed(seed : int):
    '''
    Set a seed for reproducibilty. Impacts numpy, pytorch and cuda backend.
    This is meant to be used during model evaluation
    '''
    print('SETTING SEED: ', seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def coords_3d_to_2d(pts : Tensor, camera : Tensor) -> Tensor:
    '''
    Given a tensor [N,3] of xyz coordinates and a camera matrix, projects to 2d the coordinates (y,x)
    '''

    fx = camera[0]
    fy = camera[4]
    cx = camera[2]
    cy = camera[5]
 
    x,y,z = pts[:,0], pts[:,1], pts[:,2]
    
    x_2d = ((x/z) * fx + cx).to(torch.int16)
    y_2d = ((y/z) * fy + cy).to(torch.int16)

    return torch.stack([y_2d, x_2d],dim=-1)

def box_from_mask(mask : np.ndarray, id : int) -> Tuple[int,int,int,int]:
    '''
    Returns a 2d box [y1,x1,y2,x2] of the pixels with the specified id value
    '''
    ys,xs = np.nonzero(mask == id)
    if ys.shape[0] > 0 and xs.shape[0] > 0:
        y1, x1 = min(ys), min(xs)
        y2, x2 = max(ys), max(xs)
    else:
        y1,x1,y2,x2 = 0,0,2,2

    return (y1,x1,y2,x2)

def unique_indexes(indexes : Tensor) -> Tensor:
    '''
    Given a list of [N,2] 2d indexes, removes duplicates
    '''

    N = torch.max(indexes) + 1
    encoded = indexes[:,0] * N + indexes[:,1]
    unique_encoded = torch.unique(encoded)
    decoded_y = (unique_encoded / N).int() 
    decoded_x = (unique_encoded % N).int()

    return torch.stack([decoded_y, decoded_x], dim=1) 

def torch_sample_select(t : torch.Tensor, n : int) -> torch.Tensor:
    '''
    Samples exactly n elements from a Tensor of shape [N,D1,....,Dm]
    Uses replacement only if n > N.
    Returns indexes
    '''

    N = t.shape[0]
    uniform_dist = torch.ones(N,dtype=float).to(t.device)
    if n > N:
        return torch.multinomial(uniform_dist, n, replacement=True).to(t.device)
    else:
        return torch.multinomial(uniform_dist, n, replacement=False).to(t.device)

def get_crop_coordinates(crop_wh : Sequence, crop_box : Sequence, coord_list : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Opposite of get_original_coordinates above
    crop_wh : W,H of final crop (domain of output coordinates)
    crop_box : X,Y,W,H of original crop box
    coord_list : [N,2] list of coordinates
    Also returns a mask showing which coordinates are still valid (i.e. inside the newly cropped box)
    '''
    x,y,w,h = crop_box
    # max_dim = max(w,h)
    # if w < max_dim:
    #     extra_w = max_dim - w
    #     x1 = int(x - extra_w/2.)
    #     x = x1
    # elif h < max_dim:
    #     extra_h = max_dim - h
    #     y1 = int(y - extra_h/2.)
    #     y = y1

    xs, ys = coord_list[:,1], coord_list[:,0]
    xs -= x
    ys -= y
    # obtain coordinate relative to the original crop size
    scaled_xs = (xs / w) * crop_wh[0]
    scaled_ys = (ys / h) * crop_wh[1]
    valid_x = np.logical_and(scaled_xs >= 0, scaled_xs < crop_wh[0])
    valid_y = np.logical_and(scaled_ys >= 0, scaled_ys < crop_wh[1])
    valid = np.logical_and(valid_x, valid_y)

    final_coords = np.stack([scaled_ys, scaled_xs], axis=1).astype(int)

    return final_coords, valid
    
def class_map(classes : list) -> list:
    '''
    Given a list of N numerical indexes, performs mapping from 0 to N-1
    '''
    
    new_classes = list()
    m = dict()
    cur_idx = 0

    for c in classes:   
    # already in the map, add related value
        if c in m.keys():
            new_classes.append(m[c])
    # not in map, must add it 
        else:
            m[c] = cur_idx
            new_classes.append(cur_idx)
            cur_idx += 1
    
    return new_classes

def np_normalize(v : np.ndarray) -> np.ndarray:
    
    assert len(v.shape) == 1, ' Works only with 1-dimensional arrays!'
    
    v = v.astype(float)
    vmax, vmin = np.max(v), np.min(v)

    return (v - vmax) / (vmin -vmax)

def sorted_alphanumeric(data):
    '''
    https://gist.github.com/SeanSyue/8c8ff717681e9ecffc8e43a686e68fd9
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_logger(path_log):
    '''
    https://www.toptal.com/python/in-depth-python-logging
    '''
    # Get logger
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)

    # Get formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # Get file handler and add it to logger
    fh = logging.FileHandler(path_log, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Get console handler  and add it to logger
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.propagate = False

    return logger

def generate_random_rotation():
    # Generate rotation
    anglex = np.random.uniform() * np.pi * 2
    angley = np.random.uniform() * np.pi * 2
    anglez = np.random.uniform() * np.pi * 2

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
    R_ab = Rx @ Ry @ Rz
    
    return R_ab

def init_storage_folders(args, creation_wait_timeout):
    
    r"""
    This function init the directory store where store the results
    This include the directory where storing the checkpoints and logs
    """

    checkpoints_output_folder = join(args.exp_root, args.exp_name, "models")
    logs_output_folder = join(args.exp_root, args.exp_name, "runs")
    results_output_folder = join(args.exp_root, args.exp_name, "results")

    print("Checkpoint folder: {}".format(checkpoints_output_folder))
    print("Logs folder: {}".format(logs_output_folder))
    print("Results folder: {}".format(results_output_folder))
        
    makedirs(checkpoints_output_folder, exist_ok=False)
    makedirs(logs_output_folder, exist_ok=False)
    makedirs(results_output_folder, exist_ok=False)
    makedirs(join(results_output_folder,'pcds'), exist_ok=False)
    makedirs(join(results_output_folder,'viz'), exist_ok=False)
    subprocess.call('touch {}/__init__.py'.format(join(args.exp_root, args.exp_name)), shell=True)

    start_wait_time = datetime.now()
    time_expired = True
    while (datetime.now() - start_wait_time).total_seconds() < creation_wait_timeout:
        if isdir(checkpoints_output_folder) and isdir(logs_output_folder) and isdir(results_output_folder):
            time_expired = False
            break
        sleep(0.2)
    if time_expired:
        raise TimeoutError("Time expire during the control of the creation of the directory")
    
    print("Output directories was correctly created by the master")

    return checkpoints_output_folder, logs_output_folder, results_output_folder

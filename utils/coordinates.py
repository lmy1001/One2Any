import torch
from torch import Tensor
from typing import Union, Tuple

def scale_coords(coords: Tensor, source_scale: Union[Tensor,Tuple], target_scale: Union[Tensor,Tuple]) -> Tensor:
    '''
    All measures are Y,X. Returns a copy.
    '''
    new_coords = coords.clone().to(torch.float32)
    new_coords[:,0] = new_coords[:,0] * (target_scale[0] / source_scale[0])
    new_coords[:,1] = new_coords[:,1] * (target_scale[1] / source_scale[1])

    return new_coords

def crop_coords(coords: Tensor, crop_origin: Union[Tensor,Tuple]) -> Tensor:
    '''
    Ignores box dimension, this has to be check after. All measures in Y,X. Returns a copy
    '''

    new_coords = coords.clone().to(torch.float32)
    new_coords[:,0] = new_coords[:,0] - crop_origin[0]
    new_coords[:,1] = new_coords[:,1] - crop_origin[1]
    return new_coords

def decrop_coords(coords: Tensor, crop_origin: Union[Tensor,Tuple]) -> Tensor:
    '''
    Ignores box dimension, this has to be check after. All measures in Y,X
    '''

    new_coords = coords.clone().to(torch.float32)
    new_coords[:,0] = new_coords[:,0] + crop_origin[0]
    new_coords[:,1] = new_coords[:,1] + crop_origin[1]

    return new_coords

def get_valid_coords(coords: Tensor, bounds:Union[Tensor,Tuple]) -> Tensor:
    '''
    Return boolean mask of valid coords based on given bounds. All measures in Y,X
    '''

    ys = coords[:,0]
    xs = coords[:,1]

    valid_y = torch.logical_and(ys >=0, ys<bounds[0])
    valid_x = torch.logical_and(xs >=0, xs<bounds[1])
    valid = torch.logical_and(valid_x, valid_y)

    return valid
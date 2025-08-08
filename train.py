import os
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import SmoothL1Loss, L1Loss
import torch.nn.functional as F
from einops import rearrange
from scipy.spatial.transform import Rotation as R
from tensorboardX import SummaryWriter

from models.model import One2Any
from models.optimizer import build_optimizers
import utils.logging as logging

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions
import glob
import utils.utils as utils
from utils.aligning import pose_estimation_from_nocs, adi_err, add_err, compute_auc, rotation_6d_to_matrix, get_inverse_pose_torch, save_to_ply, save_array_to_image, get_symmetry_transformations

def load_model(ckpt, model, optimizer=None):
    ckpt_dict = torch.load(ckpt, map_location='cpu')
    # keep backward compatibility
    if 'model' not in ckpt_dict and 'optimizer' not in ckpt_dict:
        state_dict = ckpt_dict
    else:
        state_dict = ckpt_dict['model']
    weights = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            weights[key[len('module.'):]] = value
        else:
            weights[key] = value

    model.load_state_dict(weights)
    optimizer = None

    if optimizer is not None:
        optimizer_state = ckpt_dict['optimizer']
        optimizer.load_state_dict(optimizer_state)


def main():
    opt = TrainOptions()
    args = opt.initialize().parse_args()
    print(args)
    
    utils.init_distributed_mode_torchrun(args)
    print(args)
    device = torch.device(args.gpu)

    maxlrstr = str(args.max_lr).replace('.', '')
    minlrstr = str(args.min_lr).replace('.', '')
    layer_decaystr = str(args.layer_decay).replace('.', '')
    weight_decaystr = str(args.weight_decay).replace('.', '')
    num_filter = str(args.num_filters[0]) if args.num_deconv > 0 else ''
    num_kernel = str(args.deconv_kernels[0]) if args.num_deconv > 0 else ''
    name = [args.dataset, args.data_name, str(args.batch_size), 'deconv'+str(args.num_deconv), \
        str(num_filter), str(num_kernel), str(args.scale_size), maxlrstr, minlrstr, \
        layer_decaystr, weight_decaystr, str(args.epochs)]

    # Logging
    if args.rank == 0:
        exp_name = args.exp_name
        log_dir = os.path.join(args.log_dir, exp_name)
        logging.check_and_make_dirs(log_dir)
        writer = SummaryWriter(logdir=log_dir)
        log_txt = os.path.join(log_dir, 'logs.txt')  
        logging.log_args_to_txt(log_txt, args)

        global result_dir
        result_dir = os.path.join(log_dir, 'results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    else:
        log_txt = None
        log_dir = None
        writer = None
    model = One2Any(args=args)

    # CPU-GPU agnostic settings
    
    cudnn.benchmark = True
    model.to(device)
    model_without_ddp = model
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    # Dataset setting
    dataset_kwargs = {
        'dataset_name': args.dataset, 
        'data_path': args.data_path, 
        'data_name': args.data_name, 
        'data_type': args.data_train, 
    }
    dataset_kwargs['scale_size'] = args.scale_size

    train_dataset = get_dataset(**dataset_kwargs)
    dataset_kwargs['data_type'] = args.data_val

    val_dataset = get_dataset(**dataset_kwargs, is_train=False)

    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=utils.get_world_size(), rank=args.rank, shuffle=True, 
    )

    sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=utils.get_world_size(), rank=args.rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size,
                                               sampler=sampler_train,
                                               num_workers=args.workers, 
                                               pin_memory=True, 
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=args.batch_size, 
                                             sampler=sampler_val,
                                             pin_memory=True)
    
    # Training settings
    criterion_o = SmoothL1Loss(beta=0.1)

    optimizer = build_optimizers(model, dict(type='AdamW', lr=args.max_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
                constructor='LDMOptimizerConstructor',
                paramwise_cfg=dict(layer_decay_rate=args.layer_decay, no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale'])))

    start_ep = 1
    if args.resume_from:
        if torch.cuda.device_count() > 1:
            load_model(args.resume_from, model.module, optimizer)
        else:
            load_model(args.resume_from, model, optimizer)
        print(f'resumed from ckpt {args.resume_from}')
        start_ep = 14 #####your resume checkpoint epoch
    if args.auto_resume:
        ckpt_list = glob.glob(f'{log_dir}/epoch_*_model.ckpt')
        idx = [int(ckpt.split('/')[-1].split('_')[-2]) for ckpt in ckpt_list]
        if len(idx) > 0:
            idx.sort(key=lambda x: -int(x))
            ckpt = f'{log_dir}/epoch_{idx[0]}_model.ckpt'
            load_model(ckpt, model.module, optimizer)
            resume_ep = int(idx[0])
            print(f'resumed from epoch {resume_ep}, ckpt {ckpt}')
            start_ep = resume_ep

    global global_step
    iterations = len(train_loader)
    global_step = iterations * (start_ep - 1)

    # Perform experiment
    for epoch in range(start_ep, args.epochs + 1):
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        loss_train_nocs = train(train_loader, model, criterion_o, criterion_r, log_txt, optimizer=optimizer, 
                           device=device, epoch=epoch, args=args, writer=writer)

        if args.rank == 0:
            if args.save_model:
                torch.save(
                    {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    os.path.join(log_dir, 'last.ckpt'))
        
        loss_nocs_val = validate(val_loader, model, criterion_o, criterion_r,
                            device=device, epoch=epoch, args=args, writer=writer)

        if args.rank == 0:
            torch.save(
                {
                    'model': model_without_ddp.state_dict(),
                },
            os.path.join(log_dir, f'epoch_{epoch}_model.ckpt'))
    


def train(train_loader, model, criterion_o, criterion_r, log_txt, optimizer, device, epoch, args, writer=None):    
    global global_step
    model.train()
    nocs_loss = logging.AverageMeter()
    half_epoch = args.epochs // 2
    iterations = len(train_loader)
    result_lines = []
    
    use_gt_nocs = True
    for batch_idx, batch in enumerate(train_loader):     
        global_step += 1

        if global_step < iterations * half_epoch:
            current_lr = args.max_lr - (args.max_lr - args.min_lr) * (global_step /
                                            iterations/half_epoch) ** 0.9
        else:
            current_lr = args.min_lr

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr*param_group['lr_scale']

        input_RGB = batch['image'].to(device)
        input_MASK = batch['mask'].to(device).to(bool)
        pcl_c = batch['roi_pcl'].to(device)

        ref_data = batch['ref_data']
        ref_image = ref_data['ref_rgb'].to(device)
        ref_mask = ref_data['ref_mask'].to(device).unsqueeze(1).int().float()
        ref_nocs = ref_data['ref_nocs'].to(device).float()
        ref_cond = torch.cat([ref_image, ref_nocs, ref_mask], dim=1).to(device)
        nocs = batch['nocs'].to(device)

        preds = model(input_RGB, input_MASK, ref_cond=ref_cond)
        nocs = nocs.permute(0, 2, 3, 1)
        gt_pose = batch['nocs_pose'].to(device)
        ref_pose = batch['ref_data']['ref_pose'].to(device)
        ref_scale_matrix = batch['ref_data']['ref_scale_matrix'].to(device)
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
            curr_pcl_m = curr_gt_nocs - 0.5  # nocs to pcl
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

        optimizer.zero_grad()
        loss_o = criterion_o(torch.cat(pred_nocs_list), torch.cat(gt_nocs_list))
        loss = loss_o
        nocs_loss.update(loss_o.detach().item(), input_RGB.size(0))
        loss.backward()
        
        if args.rank == 0:
            if batch_idx % args.print_freq == 0:
                result_line = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Loss: {loss}, loss_o: {loss_o},  Mov Avg Loss: {ma_loss},' \
                    'LR: {lr}\n'.format(
                        epoch, batch_idx, iterations,
                        loss=loss,
                        loss_o=loss_o, 
                        ma_loss=nocs_loss.avg, 
                        lr=current_lr, 
                    )
                result_lines.append(result_line)
                print(result_line)
                if writer is not None:
                    cur_iter = epoch * iterations + batch_idx
                    writer.add_scalar('Training nocs loss', nocs_loss.avg, cur_iter)
        optimizer.step()

        del cur_gt_pose_list, curr_gt_nocs_set

    if args.rank == 0:
        with open(log_txt, 'a') as txtfile:
            txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
            for result_line in result_lines:
                txtfile.write(result_line)   

    return nocs_loss.avg


def validate(val_loader, model, criterion_o, criterion_r, device, epoch, args, writer=None):    
    model.eval()
    nocs_loss = logging.AverageMeter()
    iterations = len(val_loader)
    for batch_idx, batch in enumerate(val_loader):      
        input_RGB = batch['image'].to(device)
        ob_name = batch['roi_class']
        input_MASK = batch['mask'].to(device).to(bool)
        pcl_c = batch['roi_pcl'].to(device)

        ref_data = batch['ref_data']
        ref_image = ref_data['ref_rgb'].to(device) #[0,1]
        ref_mask = ref_data['ref_mask'].to(device).unsqueeze(1).int().float()
        ref_scale_matrix = ref_data['ref_scale_matrix'].to(device)
        ref_nocs = ref_data['ref_nocs'].to(device).float()
        ref_cond = torch.cat([ref_image, ref_nocs, ref_mask], dim=1).to(device)
        nocs = batch['nocs'].to(device).permute(0, 2, 3, 1)
        gt_pose = batch['nocs_pose'].to(device)
        ref_pose = ref_data['ref_pose'].to(device)

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
            curr_pcl_m = curr_gt_nocs - 0.5  # nocs to pcl
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
        
        loss_o = criterion_o(torch.cat(pred_nocs_list), torch.cat(gt_nocs_list))
        loss = loss_o
        nocs_loss.update(loss_o.detach().item(), input_RGB.size(0))
       
        if args.rank == 0:
            if batch_idx % (args.print_freq * 10) == 0:
                result_line = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Val Loss: {loss},   loss_o: {loss_o}, '\
                    'Mov Avg Val Loss {ma_loss}, \n'.format(
                        epoch, batch_idx, iterations,
                        loss=loss, 
                        loss_o=loss_o,
                        ma_loss=nocs_loss.avg, 
                    )
                print(result_line)
                
                if writer is not None:
                    cur_iter = epoch * iterations + batch_idx
                    writer.add_scalar('Validation nocs loss', nocs_loss.avg, cur_iter)
        del cur_gt_pose_list, curr_gt_nocs_set
    return nocs_loss.avg


if __name__ == '__main__':
    main()

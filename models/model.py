import os
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch.nn.functional as F

from models.base import UNetWrapper, TextAdapterDepth
from einops import rearrange

class OV9DEncoder(nn.Module):
    def __init__(self, 
                 data_path, data_name='oo3d9dsingle',
                 out_dim=1024, ldm_prior=[320, 640, 1280+1280], sd_path=None, 
                 ):
        super().__init__()


        self.layer1 = nn.Sequential(
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, ldm_prior[0]),
            nn.ReLU(),
            nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(sum(ldm_prior), out_dim, 1),
            nn.GroupNorm(16, out_dim),
            nn.ReLU(),
        )

        self.apply(self._init_weights)

        config = OmegaConf.load('./v1-inference.yaml')
        if sd_path is None:
            config.model.params.ckpt_path = os.path.join('./pretrained_model/', 'checkpoints/v1-5-pruned-emaonly.ckpt')
        else:
            config.model.params.ckpt_path = f'../{sd_path}'

        sd_model = instantiate_from_config(config.model)
        self.encoder_vq = sd_model.first_stage_model

        self.unet = UNetWrapper(sd_model.model, use_attn=False)
    
        del sd_model.cond_stage_model
        del self.encoder_vq.decoder
        del self.unet.unet.diffusion_model.out

        for param in self.encoder_vq.parameters():
            param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, feats):
        x =  self.ldm_to_net[0](feats[0])
        for i in range(3):
            if i > 0:
                x = x + self.ldm_to_net[i](feats[i])
            x = self.layers[i](x)
            x = self.upsample_layers[i](x)
        return self.out_conv(x)

    def forward(self, x, c_crossattn):

        with torch.no_grad():
            latents = self.encoder_vq.encode(x).mode().detach()
        
        t = torch.ones((x.shape[0],), device=x.device).long()
        
        outs = self.unet(latents, t,  c_crossattn=[c_crossattn])
        feats = [outs[0], outs[1], torch.cat([outs[2], F.interpolate(outs[3], scale_factor=2)], dim=1)]
        x = torch.cat([self.layer1(feats[0]), self.layer2(feats[1]), feats[2]], dim=1)
        return self.out_layer(x)

class One2Any(nn.Module):
    def __init__(self, args=None):
        super().__init__()

        embed_dim = 192
        
        channels_in = embed_dim*8
        channels_out = embed_dim
        text_dim = 768
        
        self.encoder = OV9DEncoder(
            data_path=args.data_path, 
            data_name=args.data_name, 
            out_dim=channels_in, 
            )
        
        self.text_adapter = TextAdapterDepth(c_in=7, text_dim=text_dim)

        if args.dino:
            self.decoder = Decoder(channels_in+1024, channels_out, args)
            self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            for p in self.dino.parameters():
                p.requires_grad = False
            self.register_buffer("pixel_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), False)
            self.register_buffer("pixel_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), False)
        else:
            self.decoder = Decoder(channels_in, channels_out, args)
            self.dino = None
        self.decoder.init_weights()
        
        self.last_layer_nocs = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 3, kernel_size=3, stride=1, padding=1))

    def forward(self, x,  mask, ref_cond=None, gt_nocs=None, coarse_trans=None):    
        b, c, h, w = x.shape

        if self.dino:
            with torch.no_grad():
                pad_h, pad_w = torch.ceil(torch.tensor(h/14)).long()*14, torch.ceil(torch.tensor(w/14)).long()*14
                pad_x = torch.zeros((b, c, pad_h, pad_w)).to(x)
                pad_x[:, :, 0:h, 0:w] = x
                pad_x = (pad_x - self.pixel_mean) / self.pixel_std
                dino_feature = self.dino.get_intermediate_layers(pad_x, 
                                                                 n=self.dino.n_blocks,
                                                                 reshape=True)[-1]
                range_h = (torch.arange(h//32) * 32) / pad_h * 2 - 1
                range_w = (torch.arange(w//32) * 32) / pad_w * 2 - 1
                grid_h, grid_w = torch.meshgrid(range_h, range_w, indexing='ij')
                grid = torch.stack([grid_w, grid_h], dim=-1)
                grid = torch.stack([grid]*b, dim=0).to(x.device)
                dino_feature = torch.nn.functional.grid_sample(dino_feature, grid, align_corners=True)

        x = x * 2.0 - 1.0  
        c_crossattn = self.text_adapter(ref_cond)
        conv_feats = self.encoder(x, c_crossattn)

        if self.dino:
            conv_feats = torch.cat([conv_feats, dino_feature], dim=1)
        
        out = self.decoder([conv_feats])
        out_nocs = self.last_layer_nocs(out)
        out_nocs = torch.clip(out_nocs, 0, 1)

        res = {
            'pred_nocs': out_nocs,
        }
        return res


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.deconv = args.num_deconv
        self.in_channels = in_channels

        self.deconv_layers = self._make_deconv_layer(
            args.num_deconv,
            args.num_filters,
            args.deconv_kernels,
        )
        
        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=args.num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(
            build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, conv_feats):
        out = self.deconv_layers(conv_feats[0])
        out = self.conv_layers(out)

        out = self.up(out)
        out = self.up(out)

        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        
        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)


# Standard Library
from collections.abc import Iterable

# Import from third library
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import json
import torch

import clip
from PIL import Image
import numpy as np
import torch.nn.functional as F
import math
from lib.utils.box_ops import (box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh)


def iou_overlaps2(b1, b2, eps=1e-9):
    """
    Arguments:
        - b1: dts, [n, >=4] (x1, y1, x2, y2, ...)
        - b1: gts, [n, >=4] (x1, y1, x2, y2, ...)

    Returns:
        intersection-over-union pair-wise, generalized iou.
    """
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    iou_dict = {}
    # only for giou loss
    lt1 = torch.max(b1[:, :2], b2[:, :2])
    rb2 = torch.min(b1[:, 2:4], b2[:, 2:4])
    wh1 = (rb2 - lt1).clamp(min=0)
    inter_area = wh1[:, 0] * wh1[:, 1]
    union_area = area1
    iou = inter_area / torch.clamp(union_area, min=0)
    iou_dict['iou'] = iou
    return iou_dict

def xcorr_depthwise(x, kernel):
    """ depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, stride=kernel.size(2), groups=batch * channel, padding=0)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out

class CLIPVIT(nn.Module):
    """
        layer0 <-> Conv1, ..., layer4 <-> Conv5

        You can configure output layer and its related strides.
    """

    def __init__(self, num_features=768, template_sz=112, search_sz=224, window_sz=16, foveal_sz=64, net_name='ViT-B/16', **kwargs):
        super(CLIPVIT, self).__init__()
        model, self.preprocess = clip.load(net_name, device='cpu', jit=False)
        self.visual = model.visual
        self.pos_fc = nn.Sequential(nn.Linear(3, num_features // 2),
                                    nn.ReLU(),
                                    nn.Linear(num_features // 2, num_features),
                                    nn.Sigmoid())
        self.num_features = num_features

        self.tz = template_sz // window_sz
        self.sz = search_sz // window_sz
        self.wz = window_sz
        with torch.no_grad():
            y1 = torch.arange(0, self.wz * self.tz, self.wz).unsqueeze(1).repeat(1, self.tz).reshape(-1)
            y2 = y1.clone() + self.wz
            x1 = torch.arange(0, self.wz * self.tz, self.wz).unsqueeze(0).repeat(self.tz, 1).reshape(-1)
            x2 = x1.clone() + self.wz
            self.all_anchor = torch.stack([x1, y1, x2, y2], 1).unsqueeze(0).cuda() / (self.wz * self.tz)
            self.all_pos = torch.stack([(x1 + x2) / 2, (y1 + y2) / 2], 1).unsqueeze(0).cuda() / (self.wz * self.tz)

        self.shift_flag = ((template_sz - foveal_sz) % (foveal_sz * 2) == 0)
        self.foveal_sz = foveal_sz // self.wz


    def forward(self, input):
        xt0, x0, annot = input
        x = self.visual.conv1(x0)
        xt = self.visual.conv1(xt0)
        bz_ = xt0.size(0)
        H = int(x.shape[2])
        W = H
        C = x.shape[1]
        Ht = int(xt.shape[2])
        Wt = Ht

        cur_anchor = self.all_anchor.repeat(bz_, 1, 1).reshape(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(annot).unsqueeze(1).repeat(1, self.tz ** 2, 1).view(-1, 4).clamp(min=0.0, max=1.0)
        iou_cur = iou_overlaps2(cur_anchor, gt_boxes_vec)['iou']
        iou_cur_re = (iou_cur.reshape(bz_, self.tz ** 2) + 1) / 2


        '''foveal window-patch embedding'''
        if self.shift_flag:
            xts = torch.zeros(xt0.size()).cuda()
            xts[:, :, :-(self.wz // 2), :-(self.wz // 2)] = xt0[:, :, (self.wz // 2):, (self.wz // 2):].clone()
            xtsp = self.visual.conv1(xts)
            pad_t = (self.tz - self.foveal_sz) // 2
            xtsp = xtsp[:, :, pad_t:-pad_t, pad_t:-pad_t]
        else:
            pad_t = (self.tz - self.foveal_sz) * self.wz // 2
            xts = xt0[:, :, pad_t:-pad_t, pad_t:-pad_t]
            xtsp = self.visual.conv1(xts)
        xtsp = xtsp.reshape(bz_, C, -1)
        xtsp = xtsp.permute(0, 2, 1)
        '''end foveal window'''

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        xt = xt.reshape(xt.shape[0], xt.shape[1], -1)
        xt = xt.permute(0, 2, 1)

        # import pdb
        # pdb.set_trace()
        pos_H = int(math.sqrt(self.visual.positional_embedding[1:].size(0)))
        pos_embeds = F.interpolate(
            self.visual.positional_embedding[1:].reshape(1, pos_H, pos_H, -1).permute(0, 3, 1, 2), size=(H, W),
            mode="bilinear")

        '''distinguashable position embedding'''
        iou_cur_re = iou_cur_re.reshape(-1, 1)
        all_pos = self.all_pos.repeat(bz_, 1, 1).reshape(-1, 2)
        input_pos = torch.cat([iou_cur_re, all_pos], 1)
        pos_embedt = self.pos_fc(input_pos.to(x.dtype)).reshape(bz_, self.tz**2, -1) * 2 - 1
        pos_embedt = pos_embedt.reshape(bz_, self.tz, self.tz, -1).permute(0, 3, 1, 2)


        pad_pos_t1 = torch.zeros([bz_, C, (self.tz + 1), (self.tz + 1)], dtype=x.dtype, device=x.device)
        pad_pos_t2 = torch.zeros([bz_, C, (self.tz + 1), (self.tz + 1)], dtype=x.dtype, device=x.device)
        pad_pos_t1[:, :, :self.tz, :self.tz] = pos_embedt
        pad_pos_t2[:, :, 1:, 1:] = pos_embedt
        pad_t = (self.tz - self.foveal_sz) // 2 + 1
        pad_pos_tsp = (pad_pos_t1 + pad_pos_t2)[:, :, pad_t:(pad_t + self.foveal_sz), pad_t:(pad_t + self.foveal_sz)] / 2

        pos_embedt = pos_embedt.reshape(bz_, -1, Ht * Wt).permute(0, 2, 1).to(torch.float32)
        pos_embeds = pos_embeds.reshape(1, -1, H * W).permute(0, 2, 1).to(torch.float32)
        pad_pos_tsp = pad_pos_tsp.reshape(bz_, -1, self.foveal_sz ** 2).permute(0, 2, 1).to(torch.float32)
        '''end position embedding'''


        xtsp = xtsp + pad_pos_tsp

        pos_embeds = torch.cat([self.visual.positional_embedding[0].reshape(1, 1, x.shape[2]), pos_embeds], dim=1)

        xt = xt + pos_embedt

        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                   dtype=x.dtype, device=x.device), x],
                      dim=1)
        x = x + pos_embeds.to(x.dtype)

        x = torch.cat([xt, x, xtsp], dim=1)



        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)

        x = self.visual.ln_post(x)

        return x[:, 0:-(self.foveal_sz ** 2)]


    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype


    def freeze_layer(self):
        layers = [self.model]
        layer = layers[0]
        layer.eval()
        for param in layer.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        # self.freeze_layer()
        return self
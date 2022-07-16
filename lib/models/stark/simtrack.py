"""
Basic STARK Model (Spatial-only).
"""
import torch
from torch import nn

from lib.utils.misc import NestedTensor

from .backbone import build_backbone_simtrack
from .transformer import build_transformer
from .head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh


class SimTrack(nn.Module):
    """ This is the base class for Transformer Tracking """
    def __init__(self, backbone, box_head,
                 aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        hidden_dim = box_head.channel
        self.bottleneck = nn.Linear(backbone.num_features, hidden_dim)
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER":
            self.feat_sz_s = int(backbone.sz)
            self.feat_len_s = int(backbone.sz ** 2)
        # import pdb
        # pdb.set_trace()

    def forward(self, img=None, seq_dict=None, mode="backbone", run_box_head=True, run_cls_head=False):
        if mode == "backbone":
            return self.forward_backbone(img)
        elif mode == "head":
            return self.forward_head(seq_dict, run_box_head=run_box_head, run_cls_head=run_cls_head)
        else:
            raise ValueError

    def forward_backbone(self, input):
        '''
        :param input: list [template_img, search_img, template_anno]
        :return:
        '''
        # Forward the backbone
        output_back = self.backbone(input)  # features & masks, position embedding for the search
        # Adjust the shapes
        return self.adjust(output_back)

    def forward_head(self, seq_dict, run_box_head=True, run_cls_head=False):
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder
        # output_embed, enc_mem = self.transformer(seq_dict["feat"], seq_dict["mask"], self.query_embed.weight,
        #                                          seq_dict["pos"], return_encoder_output=True)
        output_embed = seq_dict[0]['feat']
        # Forward the corner head
        out, outputs_coord = self.forward_box_head(output_embed)
        return out, outputs_coord, output_embed

    def forward_box_head(self, memory):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""

        # adjust shape
        enc_opt = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        # run the corner head
        outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        out = {'pred_boxes': outputs_coord_new}
        return out, outputs_coord_new


    def adjust(self, output_back: list):
        """
        """
        src_feat = output_back
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(1, 0, 2)  # HWxBxC
        return {"feat": feat_vec}

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]


def build_simtrack(cfg):
    backbone = build_backbone_simtrack(cfg)  # backbone and positional encoding are built together
    box_head = build_box_head(cfg)
    model = SimTrack(
        backbone,
        box_head,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE
    )

    return model

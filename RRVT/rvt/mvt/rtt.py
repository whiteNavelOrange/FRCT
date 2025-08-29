import torch

import torch.nn.functional as F

from torch import nn

from rvt.mvt.attn import (
    Conv2DBlock,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    cache_fn,
    DenseBlock,
    FeedForward,
    FixedPositionalEncoding,
    SpatialSoftmax1D,
    SpatialSoftmax2D
)
from einops import repeat

class RTT(nn.Module):
    def __init__(
            self,
            depth = 6, 
            input_dim_before_seq = 128, 
            atten_dim = 128, 
            cross_heads = 1, 
            cross_dim_head = 64, 
            self_heads = 8,
            self_dim_head = 64,
            attn_dropout = 0.1,
            img_size = 224,
            num_rot = 72
            ):
        super().__init__()
        self.depth = depth
        self.input_dim_before_seq = input_dim_before_seq

        self.ins_preprocess = DenseBlock(self.input_dim_before_seq, atten_dim)

        self.one_preprocess = DenseBlock(10 + 3 + 2 + 2, atten_dim)

        self.cross_attend_relative_blocks = nn.ModuleList([
            PreNorm(atten_dim, Attention(atten_dim,
                                          self.input_dim_before_seq,
                                          heads=cross_heads,
                                          dim_head=cross_dim_head,
                                          dropout=attn_dropout),
                    context_dim=self.input_dim_before_seq),
            PreNorm(atten_dim, FeedForward(atten_dim))
        ])
        self.relative_latents = nn.Parameter(torch.randn(2, atten_dim))

        self.self_attend_relative_blocks = nn.ModuleList([])
        for i in range(depth):
            attend_relative_block = nn.ModuleList([
                PreNorm(atten_dim, Attention(atten_dim,
                                            heads=self_heads,
                                            dim_head=self_dim_head,
                                            dropout=attn_dropout)),
                PreNorm(atten_dim, FeedForward(atten_dim))
            ])
            self.self_attend_relative_blocks.append(attend_relative_block)

        self.final_relative = PreNorm(atten_dim, (DenseBlock(atten_dim, 10 + 3 + 2 + 2, None, 'tanh')))

        self.classify_arm = PreNorm(atten_dim, (DenseBlock(atten_dim, 2, None, None)))

        self.ss_t = SpatialSoftmax2D(img_size, img_size, 5)

        self.ss_r = SpatialSoftmax1D(3, num_rot)

    def forward(self, ins, trans_image, feats_x, feats_y, feats_z, feats_ex, mask=None):
        ins = self.ins_preprocess(ins)
        
        bs = ins.shape[0]

        relative_token = repeat(self.relative_latents, 'n d -> b n d', b=bs)

        trans_image_pos = self.ss_t(trans_image)

        feats_rot = torch.cat([feats_x, feats_y, feats_z], dim=-1)

        rot_pose = self.ss_r(feats_rot.contiguous())

        trans_rot_pose = torch.cat([trans_image_pos, rot_pose, feats_ex], dim=-1).view(-1, 1, 10 + 3 + 2 + 2)

        trans_rot_pose = self.one_preprocess(trans_rot_pose)

        x_relative = torch.cat([trans_rot_pose, relative_token], dim=1)

        cross_attn_rela, cross_ff_rela = self.cross_attend_relative_blocks
        # cross_attn_rela1, cross_ff_rela1 = self.cross_attend_relative_blocks1
        # cross_attn_rela2, cross_ff_rela2 = self.cross_attend_relative_blocks2
            # encoder cross attention
        x_relative = cross_attn_rela(x_relative, context=ins, mask=mask) + x_relative
        x_relative = cross_ff_rela(x_relative) + x_relative
        # x_relative = cross_attn_rela1(x_relative, context=ins, mask=mask) + x_relative
        # x_relative = cross_ff_rela1(x_relative) + x_relative
        # x_relative = cross_attn_rela2(x_relative, context=ins, mask=mask) + x_relative
        # x_relative = cross_ff_rela2(x_relative) + x_relative
        # self-attention layers
        for self_attn, self_ff in self.self_attend_relative_blocks:
            x_relative = self_attn(x_relative) + x_relative
            x_relative = self_ff(x_relative) + x_relative

        trans_rot_pose_relative = x_relative[:, 1, :]

        C = x_relative[:, 2, :]

        trans_rot_pose_relative = self.final_relative(trans_rot_pose_relative)

        C = self.classify_arm(C)

        out = {
            "relative_pose": trans_rot_pose_relative,
            "C": C
        }

        return out
    




class RTT2(nn.Module):
    def __init__(
            self,
            depth = 3, 
            input_dim_before_seq = 128, 
            atten_dim = 128, 
            cross_heads = 1, 
            cross_dim_head = 64, 
            self_heads = 8,
            self_dim_head = 64,
            attn_dropout = 0.1,
            img_size = 224,
            num_rot = 72
            ):
        super().__init__()
        self.depth = depth
        self.input_dim_before_seq = input_dim_before_seq

        self.ins_preprocess = DenseBlock(self.input_dim_before_seq, atten_dim)
        
        self.self_attend_ins_blocks = nn.ModuleList([])
        for i in range(depth):
            attend_relative_block = nn.ModuleList([
                PreNorm(atten_dim, Attention(atten_dim,
                                            heads=self_heads,
                                            dim_head=self_dim_head,
                                            dropout=attn_dropout)),
                PreNorm(atten_dim, FeedForward(atten_dim))
            ])
            self.self_attend_ins_blocks.append(attend_relative_block)

        self.one_preprocess = DenseBlock(10 + 3 + 2 + 2, atten_dim)

        # self.self_attend_pose_blocks = nn.ModuleList([])
        # for i in range(depth):
        #     attend_relative_block = nn.ModuleList([
        #         PreNorm(atten_dim, Attention(atten_dim,
        #                                     heads=self_heads,
        #                                     dim_head=self_dim_head,
        #                                     dropout=attn_dropout)),
        #         PreNorm(atten_dim, FeedForward(atten_dim))
        #     ])
        #     self.self_attend_pose_blocks.append(attend_relative_block)

        self.cross_attend_relative_blocks = nn.ModuleList([
            PreNorm(atten_dim, Attention(atten_dim,
                                          self.input_dim_before_seq,
                                          heads=cross_heads,
                                          dim_head=cross_dim_head,
                                          dropout=attn_dropout),
                    context_dim=self.input_dim_before_seq),
            PreNorm(atten_dim, FeedForward(atten_dim))
        ])
        self.relative_latents = nn.Parameter(torch.randn(2, atten_dim))

        self.self_attend_relative_blocks = nn.ModuleList([])
        for i in range(depth):
            attend_relative_block = nn.ModuleList([
                PreNorm(atten_dim, Attention(atten_dim,
                                            heads=self_heads,
                                            dim_head=self_dim_head,
                                            dropout=attn_dropout)),
                PreNorm(atten_dim, FeedForward(atten_dim))
            ])
            self.self_attend_relative_blocks.append(attend_relative_block)

        self.final_relative = PreNorm(atten_dim, (DenseBlock(atten_dim, 10 + 3 + 2 + 2, None, 'tanh')))

        self.classify_arm = PreNorm(atten_dim, (DenseBlock(atten_dim, 2, None, None)))

        self.ss_t = SpatialSoftmax2D(img_size, img_size, 5)

        self.ss_r = SpatialSoftmax1D(3, num_rot)

    def forward(self, ins, trans_image, feats_x, feats_y, feats_z, feats_ex, mask=None):
        ins = self.ins_preprocess(ins)
        
        bs = ins.shape[0]

        relative_token = repeat(self.relative_latents, 'n d -> b n d', b=bs)

        trans_image_pos = self.ss_t(trans_image)

        feats_rot = torch.cat([feats_x, feats_y, feats_z], dim=-1)

        rot_pose = self.ss_r(feats_rot.contiguous())

        trans_rot_pose = torch.cat([trans_image_pos, rot_pose, feats_ex], dim=-1).view(-1, 1, 10 + 3 + 2 + 2)

        trans_rot_pose = self.one_preprocess(trans_rot_pose)

        x_relative = torch.cat([trans_rot_pose, relative_token], dim=1)

        for self_attn, self_ff in self.self_attend_ins_blocks:
            ins = self_attn(ins) + ins
            ins = self_ff(ins) + ins  


        # for self_attn, self_ff in self.self_attend_pose_blocks:
        #     x_relative = self_attn(x_relative) + x_relative
        #     x_relative = self_ff(x_relative) + x_relative

        cross_attn_rela, cross_ff_rela = self.cross_attend_relative_blocks
        # cross_attn_rela1, cross_ff_rela1 = self.cross_attend_relative_blocks1
        # cross_attn_rela2, cross_ff_rela2 = self.cross_attend_relative_blocks2
            # encoder cross attention
        x_relative = cross_attn_rela(x_relative, context=ins, mask=mask) + x_relative
        x_relative = cross_ff_rela(x_relative) + x_relative
        # x_relative = cross_attn_rela1(x_relative, context=ins, mask=mask) + x_relative
        # x_relative = cross_ff_rela1(x_relative) + x_relative
        # x_relative = cross_attn_rela2(x_relative, context=ins, mask=mask) + x_relative
        # x_relative = cross_ff_rela2(x_relative) + x_relative
        # self-attention layers
        for self_attn, self_ff in self.self_attend_relative_blocks:
            x_relative = self_attn(x_relative) + x_relative
            x_relative = self_ff(x_relative) + x_relative

        trans_rot_pose_relative = x_relative[:, 1, :]

        C = x_relative[:, 2, :]

        trans_rot_pose_relative = self.final_relative(trans_rot_pose_relative)

        C = self.classify_arm(C)

        out = {
            "relative_pose": trans_rot_pose_relative,
            "C": C
        }

        return out
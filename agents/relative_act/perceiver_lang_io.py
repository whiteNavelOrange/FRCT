# Perceiver IO implementation adpated for manipulation
# Source: https://github.com/lucidrains/perceiver-pytorch
# License: https://github.com/lucidrains/perceiver-pytorch/blob/main/LICENSE

from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops.layers.torch import Reduce

from helpers.network_utils import DenseBlock, SpatialSoftmax3D, Conv3DBlock, Conv3DUpsampleBlock, SpatialSoftmax1D

# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


# PerceiverIO adapted for 6-DoF manipulation



class PerceiverVoxelLang2RobotsEncoder(nn.Module):
    def __init__(
            self,
            depth,                    # number of self-attention layers
            iterations,               # number cross-attention iterations (PerceiverIO uses just 1)
            voxel_size,               # N voxels per side (size: N*N*N)
            initial_dim,              # 10 dimensions - dimension of the input sequence to be encoded
            low_dim_size,             # 4 dimensions - proprioception: {gripper_open, left_finger, right_finger, timestep}
            layer=0,
            num_rotation_classes=72,  # 5 degree increments (5*72=360) for each of the 3-axis
            num_grip_classes=2,       # open or not open
            num_collision_classes=2,  # collisions allowed or not allowed
            input_axis=3,             # 3D tensors have 3 axes
            num_latents=512,          # number of latent vectors
            im_channels=64,           # intermediate channel size
            latent_dim=512,           # dimensions of latent vectors
            relative_dim=128,
            cross_heads=1,            # number of cross-attention heads
            latent_heads=8,           # number of latent heads
            cross_dim_head=64,
            latent_dim_head=64,
            activation='relu',
            weight_tie_layers=False,
            pos_encoding_with_lang=True,
            input_dropout=0.1,
            attn_dropout=0.1,
            decoder_dropout=0.0,
            lang_fusion_type='seq',
            voxel_patch_size=9,
            voxel_patch_stride=8,
            no_skip_connection=False,
            no_perceiver=False,
            no_language=False,
            final_dim=64,
    ):
        super().__init__()
        self.depth = depth
        self.layer = layer
        self.init_dim = int(initial_dim)
        self.iterations = iterations
        self.input_axis = input_axis
        self.voxel_size = voxel_size
        self.low_dim_size = low_dim_size
        self.im_channels = im_channels
        self.pos_encoding_with_lang = pos_encoding_with_lang
        self.lang_fusion_type = lang_fusion_type
        self.voxel_patch_size = voxel_patch_size
        self.voxel_patch_stride = voxel_patch_stride
        self.num_rotation_classes = num_rotation_classes
        self.num_grip_classes = num_grip_classes
        self.num_collision_classes = num_collision_classes
        self.final_dim = final_dim
        self.input_dropout = input_dropout
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout
        self.no_skip_connection = no_skip_connection
        self.no_perceiver = no_perceiver
        self.no_language = no_language

        # patchified input dimensions
        spatial_size = voxel_size // self.voxel_patch_stride  # 100/5 = 20

        # 64 voxel features + 64 proprio_right features + 64 proprio_left features (+ 64 lang goal features if concattenated)
        self.input_dim_before_seq = self.im_channels * 4 if self.lang_fusion_type == 'concat' else self.im_channels * 3

        # CLIP language feature dimensions
        lang_feat_dim, lang_emb_dim, lang_max_seq_len = 1024, 512, 77

        # learnable positional encoding
        if self.pos_encoding_with_lang:
            self.pos_encoding = nn.Parameter(torch.randn(1,
                                                         lang_max_seq_len + spatial_size ** 3,
                                                         self.input_dim_before_seq))
        else:
            # assert self.lang_fusion_type == 'concat', 'Only concat is supported for pos encoding without lang.'
            self.pos_encoding = nn.Parameter(torch.randn(1,
                                                         spatial_size, spatial_size, spatial_size,
                                                         self.input_dim_before_seq))

        # voxel input preprocessing 1x1 conv encoder
        self.input_preprocess = Conv3DBlock(
            self.init_dim, self.im_channels, kernel_sizes=1, strides=1,
            norm=None, activation=activation,
        )

        # patchify conv
        self.patchify = Conv3DBlock(
            self.input_preprocess.out_channels, self.im_channels,
            kernel_sizes=self.voxel_patch_size, strides=self.voxel_patch_stride,
            norm=None, activation=activation)

        # language preprocess
        if self.lang_fusion_type == 'concat':
            self.lang_preprocess = nn.Linear(lang_feat_dim, self.im_channels)
        elif self.lang_fusion_type == 'seq':
            self.lang_preprocess = nn.Linear(lang_emb_dim, self.im_channels * 3)

        # proprioception
        if self.low_dim_size > 0:
            self.proprio_preprocess = DenseBlock(
                self.low_dim_size, self.im_channels, norm=None, activation=activation,
            )

        # pooling functions
        self.local_maxp = nn.MaxPool3d(3, 2, padding=1)
        self.global_maxp = nn.AdaptiveMaxPool3d(1)

        # 1st 3D softmax
        self.ss0 = SpatialSoftmax3D(
            self.voxel_size, self.voxel_size, self.voxel_size, self.im_channels)
        flat_size = self.im_channels * 4

        # latent vectors (that are randomly initialized)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.relative_latents = nn.Parameter(torch.randn(2, relative_dim))
        self.fc_befor_atten_one = DenseBlock(self.input_dim_before_seq, self.input_dim_before_seq)
        self.fc_befor_atten_relative = DenseBlock(self.input_dim_before_seq, self.input_dim_before_seq)
        # encoder cross attention
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim,
                                          self.input_dim_before_seq,
                                          heads=cross_heads,
                                          dim_head=cross_dim_head,
                                          dropout=input_dropout),
                    context_dim=self.input_dim_before_seq),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])
        # self.cross_attend_blocks1 = nn.ModuleList([
        #     PreNorm(latent_dim, Attention(latent_dim,
        #                                   self.input_dim_before_seq,
        #                                   heads=cross_heads,
        #                                   dim_head=cross_dim_head,
        #                                   dropout=input_dropout),
        #             context_dim=self.input_dim_before_seq),
        #     PreNorm(latent_dim, FeedForward(latent_dim))
        # ])
        # self.cross_attend_blocks2 = nn.ModuleList([
        #     PreNorm(latent_dim, Attention(latent_dim,
        #                                   self.input_dim_before_seq,
        #                                   heads=cross_heads,
        #                                   dim_head=cross_dim_head,
        #                                   dropout=input_dropout),
        #             context_dim=self.input_dim_before_seq),
        #     PreNorm(latent_dim, FeedForward(latent_dim))
        # ])

        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads,
                                                    dim_head=latent_dim_head, dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        # self attention layers
        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.process_relative = nn.Linear(3 + 3 + 2, relative_dim)
        self.cross_attend_relative_blocks = nn.ModuleList([
            PreNorm(relative_dim, Attention(relative_dim,
                                          self.input_dim_before_seq,
                                          heads=cross_heads,
                                          dim_head=cross_dim_head,
                                          dropout=input_dropout),
                    context_dim=self.input_dim_before_seq),
            PreNorm(relative_dim, FeedForward(relative_dim))
        ])
        # self.cross_attend_relative_blocks1 = nn.ModuleList([
        #     PreNorm(relative_dim, Attention(relative_dim,
        #                                   self.input_dim_before_seq,
        #                                   heads=cross_heads,
        #                                   dim_head=cross_dim_head,
        #                                   dropout=input_dropout),
        #             context_dim=self.input_dim_before_seq),
        #     PreNorm(relative_dim, FeedForward(relative_dim))
        # ])
        # self.cross_attend_relative_blocks2 = nn.ModuleList([
        #     PreNorm(relative_dim, Attention(relative_dim,
        #                                   self.input_dim_before_seq,
        #                                   heads=cross_heads,
        #                                   dim_head=cross_dim_head,
        #                                   dropout=input_dropout),
        #             context_dim=self.input_dim_before_seq),
        #     PreNorm(relative_dim, FeedForward(relative_dim))
        # ])
        self.attend_relative_blocks = nn.ModuleList([])

        for i in range(depth):
            attend_relative_block = nn.ModuleList([
                PreNorm(relative_dim, Attention(relative_dim,
                                            heads=latent_heads,
                                            dim_head=latent_dim_head,
                                            dropout=attn_dropout)),
                PreNorm(relative_dim, FeedForward(relative_dim))
            ])
            self.attend_relative_blocks.append(attend_relative_block)

        self.final_relative = PreNorm(relative_dim, (DenseBlock(relative_dim, 3 + 3 + 2, None, 'tanh')))

        self.classify_arm = PreNorm(relative_dim, (DenseBlock(relative_dim, 2, None, None)))

        # decoder cross attention
        self.decoder_cross_attn = PreNorm(self.input_dim_before_seq, Attention(self.input_dim_before_seq,
                                                                               latent_dim,
                                                                               heads=cross_heads,
                                                                               dim_head=cross_dim_head,
                                                                               dropout=decoder_dropout),
                                          context_dim=latent_dim)

        # upsample conv
        self.up0 = Conv3DUpsampleBlock(
            self.input_dim_before_seq, self.final_dim,
            kernel_sizes=self.voxel_patch_size, strides=self.voxel_patch_stride,
            norm=None, activation=activation,
        )

        # 2nd 3D softmax
        self.ss1 = SpatialSoftmax3D(
            spatial_size, spatial_size, spatial_size,
            self.input_dim_before_seq)

        flat_size += self.input_dim_before_seq * 4

        self.ss_T = SpatialSoftmax3D(voxel_size, voxel_size, voxel_size, 1)
        
        self.ss_R = SpatialSoftmax1D(3, self.num_rotation_classes)
        # final 3D softmax
        self.final = Conv3DBlock(
            self.im_channels if (self.no_perceiver or self.no_skip_connection) else self.im_channels * 2,
            self.im_channels,
            kernel_sizes=3,
            strides=1, norm=None, activation=activation)

        self.trans_decoder = Conv3DBlock(
            self.final_dim, 1, kernel_sizes=3, strides=1,
            norm=None, activation=None,
        )

        # rotation, gripper, and collision MLP layers
        if self.num_rotation_classes > 0:
            self.ss_final = SpatialSoftmax3D(
                self.voxel_size, self.voxel_size, self.voxel_size,
                self.im_channels)

            flat_size += self.im_channels * 4

            self.dense0 =  DenseBlock(flat_size, 256, None, activation)
            self.dense1 = DenseBlock(256, self.final_dim, None, activation)

            self.rot_grip_collision_ff = DenseBlock(self.final_dim,
                                                    self.num_rotation_classes * 3 + \
                                                    self.num_grip_classes + \
                                                    self.num_collision_classes,
                                                    None, None)

    def encode_text(self, x):
        with torch.no_grad():
            text_feat, text_emb = self._clip_rn50.encode_text_with_embeddings(x)

        text_feat = text_feat.detach()
        text_emb = text_emb.detach()
        text_mask = torch.where(x==0, x, 1)  # [1, max_token_len]
        return text_feat, text_emb

    def forward(
            self,
            ins,
            proprio_right,
            proprio_left,
            lang_goal_emb,
            lang_token_embs,
            prev_layer_voxel_grid,
            bounds,
            prev_layer_bounds,
            mask=None,
    ):
        # preprocess input
        d0 = self.input_preprocess(ins)                       # [B,10,100,100,100] -> [B,64,100,100,100]

        # aggregated features from 1st softmax and maxpool for MLP decoders
        feats = [self.ss0(d0.contiguous()), self.global_maxp(d0).view(ins.shape[0], -1)]

        # patchify input (5x5x5 patches)
        ins = self.patchify(d0)                               # [B,64,100,100,100] -> [B,64,20,20,20]

        b, c, d, h, w, device = *ins.shape, ins.device
        axis = [d, h, w]
        assert len(axis) == self.input_axis, 'input must have the same number of axis as input_axis'

        # concat proprio
        if self.low_dim_size > 0:
            p_right = self.proprio_preprocess(proprio_right)              # [B,4] -> [B,64]
            p_right = p_right.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
            ins = torch.cat([ins, p_right], dim=1)                  # [B,128,20,20,20]

            p_left = self.proprio_preprocess(proprio_left)              # [B,4] -> [B,64]
            p_left = p_left.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
            ins = torch.cat([ins, p_left], dim=1)                  # [B,192,20,20,20]

        # language ablation
        if self.no_language:
            lang_goal_emb = torch.zeros_like(lang_goal_emb)
            lang_token_embs = torch.zeros_like(lang_token_embs)

        # option 1: tile and concat lang goal to input
        if self.lang_fusion_type == 'concat':
            lang_emb = lang_goal_emb
            lang_emb = lang_emb.to(dtype=ins.dtype)
            l = self.lang_preprocess(lang_emb)
            l = l.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, d, h, w)
            ins = torch.cat([ins, l], dim=1)

        # channel last
        ins = rearrange(ins, 'b d ... -> b ... d')            # [B,20,20,20,192]

        # add pos encoding to grid
        if not self.pos_encoding_with_lang:
            ins = ins + self.pos_encoding

        ######################## NOTE #############################
        # NOTE: If you add positional encodings ^here the lang embs
        # won't have positional encodings. I accidently forgot
        # to turn this off for all the experiments in the paper.
        # So I guess those models were using language embs
        # as a bag of words :( But it doesn't matter much for
        # RLBench tasks since we don't test for novel instructions
        # at test time anyway. The recommend way is to add
        # positional encodings to the final input sequence
        # fed into the Perceiver Transformer, as done below
        # (and also in the Colab tutorial).
        ###########################################################

        # concat to channels of and flatten axis
        queries_orig_shape = ins.shape

        # rearrange input to be channel last
        ins = rearrange(ins, 'b ... d -> b (...) d')          # [B,8000,192]
        ins_wo_prev_layers = ins

        # option 2: add lang token embs as a sequence
        if self.lang_fusion_type == 'seq':
            l = self.lang_preprocess(lang_token_embs)         # [B,77,1024] -> [B,77,192]
            ins = torch.cat((l, ins), dim=1)                  # [B,8077,192]

        # add pos encoding to language + flattened grid (the recommended way)
        if self.pos_encoding_with_lang:
            ins = ins + self.pos_encoding

        ins_one = self.fc_befor_atten_one(ins)
        # ins_one = ins
        # batchify latents
        x = repeat(self.latents, 'n d -> b n d', b=b)

        cross_attn, cross_ff = self.cross_attend_blocks
        # cross_attn1, cross_ff1 = self.cross_attend_blocks1
        # cross_attn2, cross_ff2 = self.cross_attend_blocks2
        for it in range(self.iterations):
            # encoder cross attention
            x = cross_attn(x, context=ins_one, mask=mask) + x
            x = cross_ff(x) + x
            # x = cross_attn1(x, context=ins, mask=mask) + x
            # x = cross_ff1(x) + x
            # x = cross_attn2(x, context=ins, mask=mask) + x
            # x = cross_ff2(x) + x
            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # decoder cross attention
        latents = self.decoder_cross_attn(ins_one, context=x)

        # crop out the language part of the output sequence
        if self.lang_fusion_type == 'seq':
            latents = latents[:, l.shape[1]:]

        # reshape back to voxel grid
        latents = latents.view(b, *queries_orig_shape[1:-1], latents.shape[-1]) # [B,20,20,20,64]
        latents = rearrange(latents, 'b ... d -> b d ...')                      # [B,64,20,20,20]

        # aggregated features from 2nd softmax and maxpool for MLP decoders
        feats.extend([self.ss1(latents.contiguous()), self.global_maxp(latents).view(b, -1)])

        # upsample
        u0 = self.up0(latents)

        # ablations
        if self.no_skip_connection:
            u = self.final(u0)
        elif self.no_perceiver:
            u = self.final(d0)
        else:
            u = self.final(torch.cat([d0, u0], dim=1))

        # translation decoder
        trans_one = self.trans_decoder(u)

        # rotation, gripper, and collision MLPs
        rot_and_grip_out_one = None
        if self.num_rotation_classes > 0:
            feats.extend([self.ss_final(u.contiguous()), self.global_maxp(u).view(b, -1)])

            dense0 = self.dense0(torch.cat(feats, dim=1))
            dense1 = self.dense1(dense0)                     # [B,72*3+2+2]

            rot_and_grip_collision_out = self.rot_grip_collision_ff(dense1)
            rot_and_grip_out_one = rot_and_grip_collision_out[:, :-self.num_collision_classes]
            collision_out_one = rot_and_grip_collision_out[:, -self.num_collision_classes:]

        # trans_one_ = trans_one.detach()
        trans_pos = self.ss_T(trans_one)
        rot_one = rot_and_grip_out_one[:, :self.num_rotation_classes * 3]
        rot_one = rot_one.view(-1, 3, self.num_rotation_classes)
        grip_one = rot_and_grip_out_one[:, self.num_rotation_classes * 3:]
        rot_pos = self.ss_R(rot_one.contiguous())
        trans_and_rot_one = torch.cat([trans_pos, rot_pos, grip_one], dim=1).view(-1, 1, 3 + 3 + 2)
        trans_and_rot_one = self.process_relative(trans_and_rot_one)


        relative_token = repeat(self.relative_latents, 'n d -> b n d', b=b)

        x_relative = torch.cat([trans_and_rot_one, relative_token], dim=1)   
            
        ins_relative = self.fc_befor_atten_relative(ins)
        # ins_relative = ins

        cross_attn_rela, cross_ff_rela = self.cross_attend_relative_blocks
        # cross_attn_rela1, cross_ff_rela1 = self.cross_attend_relative_blocks1
        # cross_attn_rela2, cross_ff_rela2 = self.cross_attend_relative_blocks2
        for it in range(self.iterations):
            # encoder cross attention
            x_relative = cross_attn_rela(x_relative, context=ins_relative, mask=mask) + x_relative
            x_relative = cross_ff_rela(x_relative) + x_relative
            # x_relative = cross_attn_rela1(x_relative, context=ins, mask=mask) + x_relative
            # x_relative = cross_ff_rela1(x_relative) + x_relative
            # x_relative = cross_attn_rela2(x_relative, context=ins, mask=mask) + x_relative
            # x_relative = cross_ff_rela2(x_relative) + x_relative
            # self-attention layers
            for self_attn, self_ff in self.attend_relative_blocks:
                x_relative = self_attn(x_relative) + x_relative
                x_relative = self_ff(x_relative) + x_relative
        
        trans_and_rot_relative = x_relative[:, 1, :]
        C = x_relative[:, 2, :]
        trans_and_rot_relative = self.final_relative(trans_and_rot_relative)
        C = self.classify_arm(C)
        



        return trans_one, rot_and_grip_out_one, collision_out_one, trans_and_rot_relative, C

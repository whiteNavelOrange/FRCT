import copy
import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from pytorch3d import transforms as torch3d_tf
from yarr.agents.agent import Agent, ActResult, ScalarSummary, \
    HistogramSummary, ImageSummary, Summary

from voxel.voxel_grid import VoxelGrid
from helpers.utils import visualise_voxel
from voxel.augmentation import apply_se3_augmentation_2Robots_relative3, apply_se3_augmentation_2Robots_relative2, apply_se3_augmentation_2Robots_relative
from helpers.clip.core.clip import build_model, load_clip
import math
import transformers
from helpers.optim.lamb import Lamb

from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt

NAME = 'QAttentionAgent'



class QFunction2Robots(nn.Module):

    def __init__(self,
                 perceiver_encoder: nn.Module,
                 voxelizer: VoxelGrid,
                 bounds_offset: float,
                 rotation_resolution: float,
                 device,
                 training):
        super(QFunction2Robots, self).__init__()
        self._rotation_resolution = rotation_resolution
        self._voxelizer = voxelizer
        self._bounds_offset = bounds_offset
        self._qnet = perceiver_encoder.to(device)

        # distributed training
        if training:
            self._qnet = DDP(self._qnet, device_ids=[device])

    def _argmax_3d(self, tensor_orig):
        b, c, d, h, w = tensor_orig.shape  # c will be one
        idxs = tensor_orig.view(b, c, -1).argmax(-1)
        indices = torch.cat([((idxs // h) // d), (idxs // h) % w, idxs % w], 1)
        return indices

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies = None
        ignore_collision = None
        if q_rot_grip is not None:
            q_rot = torch.stack(torch.split(
                q_rot_grip[:, :-2],
                int(360 // self._rotation_resolution),
                dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [q_rot[:, 0:1].argmax(-1),
                 q_rot[:, 1:2].argmax(-1),
                 q_rot[:, 2:3].argmax(-1),
                 q_rot_grip[:, -2:].argmax(-1, keepdim=True)], -1)
            ignore_collision = q_collision[:, -2:].argmax(-1, keepdim=True)
        return coords, rot_and_grip_indicies, ignore_collision

    def forward(self, rgb_pcd, proprio_right, proprio_left, pcd, lang_goal_emb, lang_token_embs,
                bounds=None, prev_bounds=None, prev_layer_voxel_grid=None):
        # rgb_pcd will be list of list (list of [rgb, pcd])
        b = rgb_pcd[0][0].shape[0]
        pcd_flat = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1)

        # flatten RGBs and Pointclouds
        rgb = [rp[0] for rp in rgb_pcd]
        feat_size = rgb[0].shape[1]
        flat_imag_features = torch.cat(
            [p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in rgb], 1)

        # construct voxel grid
        voxel_grid = self._voxelizer.coords_to_bounding_voxel_grid(
            pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds)

        # swap to channels fist
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()

        # batch bounds if necessary
        if bounds.shape[0] != b:
            bounds = bounds.repeat(b, 1)

        # forward pass
        q_trans_one, \
        q_rot_and_grip_one,\
        q_ignore_collisions_one, \
        trans_and_rot_relative, \
        pred_arm = self._qnet(voxel_grid,
                                         proprio_right,
                                         proprio_left,
                                         lang_goal_emb,
                                         lang_token_embs,
                                         prev_layer_voxel_grid,
                                         bounds,
                                         prev_bounds)

        return q_trans_one, q_rot_and_grip_one, q_ignore_collisions_one, trans_and_rot_relative, pred_arm, voxel_grid


class QAttentionPerActBCAgent2Robots(Agent):

    def __init__(self,
                 layer: int,
                 coordinate_bounds: list,
                 perceiver_encoder: nn.Module,
                 camera_names: list,
                 batch_size: int,
                 voxel_size: int,
                 bounds_offset: float,
                 voxel_feature_size: int,
                 image_crop_size: int,
                 num_rotation_classes: int,
                 rotation_resolution: float,
                 lr: float = 0.0001,
                 lr_scheduler: bool = False,
                 training_iterations: int = 100000,
                 num_warmup_steps: int = 20000,
                 trans_loss_weight: float = 1.0,
                 rot_loss_weight: float = 1.0,
                 grip_loss_weight: float = 1.0,
                 collision_loss_weight: float = 1.0,
                 relative_loss_weight: float = 20.0,
                 arm_loss_weight: float = 1.0,
                 include_low_dim_state: bool = False,
                 image_resolution: list = None,
                 lambda_weight_l2: float = 0.0,
                 transform_augmentation: bool = True,
                 transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
                 transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
                 transform_augmentation_rot_resolution: int = 5,
                 optimizer_type: str = 'adam',
                 num_devices: int = 1,
                 wandb_run = None,
                 ):
        self._layer = layer
        self._coordinate_bounds = coordinate_bounds
        self._perceiver_encoder = perceiver_encoder
        self._voxel_feature_size = voxel_feature_size
        self._bounds_offset = bounds_offset
        self._image_crop_size = image_crop_size
        self._lr = lr
        self._lr_scheduler = lr_scheduler
        self._training_iterations = training_iterations
        self._num_warmup_steps = num_warmup_steps
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._relative_loss_weight = relative_loss_weight
        self._arm_loss_weight = arm_loss_weight
        self._include_low_dim_state = include_low_dim_state
        self._image_resolution = image_resolution or [128, 128]
        self._voxel_size = voxel_size
        self._camera_names = camera_names
        self._num_cameras = len(camera_names)
        self._batch_size = batch_size
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = torch.from_numpy(np.array(transform_augmentation_xyz))
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._optimizer_type = optimizer_type
        self._num_devices = num_devices
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution
        self._wandb_run = wandb_run

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self._name = NAME + '_layer' + str(self._layer)

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        if device is None:
            device = torch.device('cpu')

        self._voxelizer = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=self._batch_size if training else 1,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod(self._image_resolution) * self._num_cameras,
        )

        self._q = QFunction2Robots(self._perceiver_encoder,
                            self._voxelizer,
                            self._bounds_offset,
                            self._rotation_resolution,
                            device,
                            training).to(device).train(training)

        grid_for_crop = torch.arange(0,
                                     self._image_crop_size,
                                     device=device).unsqueeze(0).repeat(self._image_crop_size, 1).unsqueeze(-1)
        self._grid_for_crop = torch.cat([grid_for_crop.transpose(1, 0),
                                         grid_for_crop], dim=2).unsqueeze(0)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds,
                                               device=device).unsqueeze(0)

        if self._training:
            # optimizer
            if self._optimizer_type == 'lamb':
                self._optimizer = Lamb(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
            elif self._optimizer_type == 'adam':
                self._optimizer = torch.optim.Adam(
                    self._q.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                )
            else:
                raise Exception('Unknown optimizer type')

            # learning rate scheduler
            if self._lr_scheduler:
                self._scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                    self._optimizer,
                    num_warmup_steps=self._num_warmup_steps,
                    num_training_steps=self._training_iterations,
                    num_cycles=self._training_iterations // 10000,
                )

            # one-hot zero tensors
            self._action_trans_one_hot_zeros = torch.zeros((self._batch_size,
                                                            1,
                                                            self._voxel_size,
                                                            self._voxel_size,
                                                            self._voxel_size),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_x_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_y_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_rot_z_one_hot_zeros = torch.zeros((self._batch_size,
                                                            self._num_rotation_classes),
                                                            dtype=int,
                                                            device=device)
            self._action_grip_one_hot_zeros = torch.zeros((self._batch_size,
                                                           2),
                                                           dtype=int,
                                                           device=device)
            self._action_ignore_collisions_one_hot_zeros = torch.zeros((self._batch_size,
                                                                        2),
                                                                        dtype=int,
                                                                        device=device)

            # print total params
            logging.info('# Q Params: %d' % sum(
                p.numel() for name, p in self._q.named_parameters() \
                if p.requires_grad and 'clip' not in name))
        else:
            for param in self._q.parameters():
                param.requires_grad = False

            # load CLIP for encoding language goals during evaluation
            model, _ = load_clip("RN50", jit=False)
            self._clip_rn50 = build_model(model.state_dict())
            self._clip_rn50 = self._clip_rn50.float().to(device)
            self._clip_rn50.eval()
            del model

            self._voxelizer.to(device)
            self._q.to(device)

    def _extract_crop(self, pixel_action, observation):
        # Pixel action will now be (B, 2)
        # observation = stack_on_channel(observation)
        h = observation.shape[-1]
        top_left_corner = torch.clamp(
            pixel_action - self._image_crop_size // 2, 0,
            h - self._image_crop_size)
        grid = self._grid_for_crop + top_left_corner.unsqueeze(1)
        grid = ((grid / float(h)) * 2.0) - 1.0  # between -1 and 1
        # Used for cropping the images across a batch
        # swap fro y x, to x, y
        grid = torch.cat((grid[:, :, :, 1:2], grid[:, :, :, 0:1]), dim=-1)
        crop = F.grid_sample(observation, grid, mode='nearest',
                             align_corners=True)
        return crop

    def _preprocess_inputs(self, replay_sample):
        obs = []
        pcds = []
        self._crop_summary = []
        for n in self._camera_names:
            rgb = replay_sample['%s_rgb' % n]
            pcd = replay_sample['%s_point_cloud' % n]

            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _act_preprocess_inputs(self, observation):
        obs, pcds = [], []
        for n in self._camera_names:
            rgb = observation['%s_rgb' % n]
            pcd = observation['%s_point_cloud' % n]

            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _get_value_from_voxel_index(self, q, voxel_idx):
        b, c, d, h, w = q.shape
        q_trans_flat = q.view(b, c, d * h * w)
        flat_indicies = (voxel_idx[:, 0] * d * h + voxel_idx[:, 1] * h + voxel_idx[:, 2])[:, None].int()
        highest_idxs = flat_indicies.unsqueeze(-1).repeat(1, c, 1)
        chosen_voxel_values = q_trans_flat.gather(2, highest_idxs)[..., 0]  # (B, trans + rot + grip)
        return chosen_voxel_values

    def _get_value_from_rot_and_grip(self, rot_grip_q, rot_and_grip_idx):
        q_rot = torch.stack(torch.split(
            rot_grip_q[:, :-2], int(360 // self._rotation_resolution),
            dim=1), dim=1)  # B, 3, 72
        q_grip = rot_grip_q[:, -2:]
        rot_and_grip_values = torch.cat(
            [q_rot[:, 0].gather(1, rot_and_grip_idx[:, 0:1]),
             q_rot[:, 1].gather(1, rot_and_grip_idx[:, 1:2]),
             q_rot[:, 2].gather(1, rot_and_grip_idx[:, 2:3]),
             q_grip.gather(1, rot_and_grip_idx[:, 3:4])], -1)
        return rot_and_grip_values

    def _celoss(self, pred, labels):
        return self._cross_entropy_loss(pred, labels.argmax(-1))

    def _mseloss(self, pred, labels):

        return F.mse_loss(pred, labels)
    
    def circular_mse_loss(self, y_pred, y_true, reduction='mean'):
        diff = torch.abs(y_pred - y_true)
        circular_diff = torch.minimum(diff, 1 - diff)  # 使用 torch.minimum 逐元素取最小值
        loss = circular_diff ** 2
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss
        else:
            raise ValueError("Invalid reduction mode.")

    def _softmax_q_trans(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)

    def _softmax_q_rot_grip(self, q_rot_grip):
        q_rot_x_flat = q_rot_grip[:, 0*self._num_rotation_classes: 1*self._num_rotation_classes]
        q_rot_y_flat = q_rot_grip[:, 1*self._num_rotation_classes: 2*self._num_rotation_classes]
        q_rot_z_flat = q_rot_grip[:, 2*self._num_rotation_classes: 3*self._num_rotation_classes]
        q_grip_flat  = q_rot_grip[:, 3*self._num_rotation_classes:]

        q_rot_x_flat_softmax = F.softmax(q_rot_x_flat, dim=1)
        q_rot_y_flat_softmax = F.softmax(q_rot_y_flat, dim=1)
        q_rot_z_flat_softmax = F.softmax(q_rot_z_flat, dim=1)
        q_grip_flat_softmax = F.softmax(q_grip_flat, dim=1)

        return torch.cat([q_rot_x_flat_softmax,
                          q_rot_y_flat_softmax,
                          q_rot_z_flat_softmax,
                          q_grip_flat_softmax], dim=1)

    def _softmax_ignore_collision(self, q_collision):
        q_collision_softmax = F.softmax(q_collision, dim=1)
        return q_collision_softmax
    
    def _softmax_pred_arm(self, q_pred_arm):
        q_pred_softmax = F.softmax(q_pred_arm, dim=1)
        return q_pred_softmax
    

    
    def compute_another(self, action_gripper_pose_one, action_gripper_relative, bounds):
        # bounds = self._coordinate_bounds.to(self._device)
        bs = action_gripper_pose_one.shape[0]
        tran_relative = action_gripper_relative[:, :3] * self._voxel_size / 2.0
        trans_range = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        tran_relative = tran_relative * trans_range
        roll_pitch_yaw = action_gripper_relative[:, 3: 6] * math.pi
        
        
        tran_rot_relative = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=self._device)
        tran_rot_relative[:, :3, :3] = torch3d_tf.euler_angles_to_matrix(roll_pitch_yaw, "XYZ")
        tran_rot_relative[:, :3, 3] = tran_relative
        
        
        tran_rot_one = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=self._device)
        # action_gripper_pose_one = torch.from_numpy(action_gripper_pose_one).to(device=self._device)
        tran_rot_one[:, :3, 3] = action_gripper_pose_one[:, :3]
        quat_wxyz_one = torch.cat((action_gripper_pose_one[:, 6].unsqueeze(1),
                                          action_gripper_pose_one[:, 3:6]), dim=1)
        tran_rot_one[:, :3, :3] = torch3d_tf.quaternion_to_matrix(quat_wxyz_one)

        tran_rot_another = torch.bmm(tran_rot_one, tran_rot_relative)

        tran_another = tran_rot_another[:, :3, 3]

        rot_quat_wxyz_another = torch3d_tf.matrix_to_quaternion(tran_rot_another[:, :3, :3])
        rot_quat_xyzw_another = torch.cat([rot_quat_wxyz_another[:, 1:], rot_quat_wxyz_another[:, 0].unsqueeze(1)], dim=1)
        
        grip_another = F.softmax(action_gripper_relative[:, -2:] / 0.01, dim=-1)
        grip_one_hot = torch.tensor([0,1]).to(self._device).unsqueeze(0)
        grip_one_hot = grip_one_hot.repeat(bs, 1)
        grip_another = torch.sum(grip_one_hot * grip_another, dim=-1, keepdim=True)
        
        tran_rot_grip_another = torch.cat([tran_another, rot_quat_xyzw_another, grip_another], dim=1)

        return tran_rot_grip_another

    def act_compute_another(self, trans, eular, action_gripper_relative, bounds):
        bs = action_gripper_relative.shape[0]
        tran_relative = action_gripper_relative[:, :3] * self._voxel_size / 2.0
        trans_range = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        tran_relative = tran_relative * trans_range
        roll_pitch_yaw = action_gripper_relative[:, 3: 6] * math.pi
        
        
        tran_rot_relative = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=self._device)
        tran_rot_relative[:, :3, :3] = torch3d_tf.euler_angles_to_matrix(roll_pitch_yaw, "XYZ")
        tran_rot_relative[:, :3, 3] = tran_relative
        
        
        tran_rot_one = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=self._device)
        # action_gripper_pose_one = torch.from_numpy(action_gripper_pose_one).to(device=self._device)
        tran_rot_one[:, :3, 3] = trans
        tran_rot_one[:, :3, :3] = torch3d_tf.euler_angles_to_matrix(eular * self._rotation_resolution - 180, "XYZ")

        tran_rot_another = torch.bmm(tran_rot_one, tran_rot_relative)

        tran_another = tran_rot_another[:, :3, 3]

        rot_quat_wxyz_another = torch3d_tf.matrix_to_quaternion(tran_rot_another[:, :3, :3])
        rot_quat_xyzw_another = torch.cat([rot_quat_wxyz_another[:, 1:], rot_quat_wxyz_another[:, 0].unsqueeze(1)], dim=1)
        
        grip_another = F.softmax(action_gripper_relative[:, -2:] / 0.01, dim=-1)
        grip_one_hot = torch.tensor([0,1]).to(self._device).unsqueeze(0)
        grip_one_hot = grip_one_hot.repeat(bs, 1)
        grip_another = torch.sum(grip_one_hot * grip_another, dim=-1, keepdim=True)
        
        tran_rot_grip_another = torch.cat([tran_another, rot_quat_xyzw_another, grip_another], dim=1)

        return tran_rot_grip_another
    

    def compute_another2(self, action_gripper_pose_one, action_gripper_relative, bounds):
        # bounds = self._coordinate_bounds.to(self._device)
        bs = action_gripper_pose_one.shape[0]
        tran_relative = action_gripper_relative[:, :3] * self._voxel_size / 2.0
        trans_range = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        tran_relative = tran_relative * trans_range
        roll_pitch_yaw = action_gripper_relative[:, 3: 6] * math.pi
        
        
        tran_rot_relative = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=self._device)
        tran_rot_relative[:, :3, :3] = torch3d_tf.euler_angles_to_matrix(roll_pitch_yaw, "XYZ")
        # tran_rot_relative[:, :3, 3] = tran_relative
        
        
        tran_rot_one = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=self._device)
        # action_gripper_pose_one = torch.from_numpy(action_gripper_pose_one).to(device=self._device)
        tran_rot_one[:, :3, 3] = action_gripper_pose_one[:, :3]
        quat_wxyz_one = torch.cat((action_gripper_pose_one[:, 6].unsqueeze(1),
                                          action_gripper_pose_one[:, 3:6]), dim=1)
        tran_rot_one[:, :3, :3] = torch3d_tf.quaternion_to_matrix(quat_wxyz_one)

        tran_rot_another = torch.bmm(tran_rot_one, tran_rot_relative)

        tran_another = tran_rot_another[:, :3, 3] + tran_relative

        rot_quat_wxyz_another = torch3d_tf.matrix_to_quaternion(tran_rot_another[:, :3, :3])
        rot_quat_xyzw_another = torch.cat([rot_quat_wxyz_another[:, 1:], rot_quat_wxyz_another[:, 0].unsqueeze(1)], dim=1)
        
        grip_another = F.softmax(action_gripper_relative[:, -2:] / 0.01, dim=-1)
        grip_one_hot = torch.tensor([0,1]).to(self._device).unsqueeze(0)
        grip_one_hot = grip_one_hot.repeat(bs, 1)
        grip_another = torch.sum(grip_one_hot * grip_another, dim=-1, keepdim=True)
        
        tran_rot_grip_another = torch.cat([tran_another, rot_quat_xyzw_another, grip_another], dim=1)

        return tran_rot_grip_another
    

    def act_compute_another2(self, trans, eular, action_gripper_relative, bounds):
        bs = action_gripper_relative.shape[0]
        tran_relative = action_gripper_relative[:, :3] * self._voxel_size / 2.0
        trans_range = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        tran_relative = tran_relative * trans_range
        roll_pitch_yaw = action_gripper_relative[:, 3: 6] * math.pi
        
        
        tran_rot_relative = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=self._device)
        tran_rot_relative[:, :3, :3] = torch3d_tf.euler_angles_to_matrix(roll_pitch_yaw, "XYZ")
        # tran_rot_relative[:, :3, 3] = tran_relative
        
        
        tran_rot_one = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=self._device)
        # action_gripper_pose_one = torch.from_numpy(action_gripper_pose_one).to(device=self._device)
        tran_rot_one[:, :3, 3] = trans
        tran_rot_one[:, :3, :3] = torch3d_tf.euler_angles_to_matrix(eular * self._rotation_resolution - 180, "XYZ")

        tran_rot_another = torch.bmm(tran_rot_one, tran_rot_relative)

        tran_another = tran_rot_another[:, :3, 3] + tran_relative

        rot_quat_wxyz_another = torch3d_tf.matrix_to_quaternion(tran_rot_another[:, :3, :3])
        rot_quat_xyzw_another = torch.cat([rot_quat_wxyz_another[:, 1:], rot_quat_wxyz_another[:, 0].unsqueeze(1)], dim=1)
        
        grip_another = F.softmax(action_gripper_relative[:, -2:] / 0.01, dim=-1)
        grip_one_hot = torch.tensor([0,1]).to(self._device).unsqueeze(0)
        grip_one_hot = grip_one_hot.repeat(bs, 1)
        grip_another = torch.sum(grip_one_hot * grip_another, dim=-1, keepdim=True)
        
        tran_rot_grip_another = torch.cat([tran_another, rot_quat_xyzw_another, grip_another], dim=1)

        return tran_rot_grip_another

    def update_compute_another_euler(self, trans, eular, action_gripper_relative, bounds):
        bs = action_gripper_relative.shape[0]
        tran_relative = action_gripper_relative[:, :3] * self._voxel_size / 2.0
        trans_range = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        tran_relative = tran_relative * trans_range
        roll_pitch_yaw = action_gripper_relative[:, 3: 6] * math.pi
        
        
        tran_rot_relative = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=self._device)
        tran_rot_relative[:, :3, :3] = torch3d_tf.euler_angles_to_matrix(roll_pitch_yaw, "XYZ")
        # tran_rot_relative[:, :3, 3] = tran_relative
        
        
        tran_rot_one = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=self._device)
        # action_gripper_pose_one = torch.from_numpy(action_gripper_pose_one).to(device=self._device)
        tran_rot_one[:, :3, 3] = trans
        tran_rot_one[:, :3, :3] = torch3d_tf.euler_angles_to_matrix((eular * self._rotation_resolution - 180) / 180.0 * math.pi, "XYZ")

        tran_rot_another = torch.bmm(tran_rot_one, tran_rot_relative)

        tran_another = tran_rot_another[:, :3, 3] + tran_relative

        el = torch3d_tf.matrix_to_euler_angles(tran_rot_another[:, :3, :3], "ZYX") * 180 / math.pi + 180.0
        el = el / self._rotation_resolution
        el = torch.flip(el, dims=[1])
        grip_another = F.softmax(action_gripper_relative[:, -2:] / 0.01, dim=-1)
        grip_one_hot = torch.tensor([0,1]).to(self._device).unsqueeze(0)
        grip_one_hot = grip_one_hot.repeat(bs, 1)
        grip_another = torch.sum(grip_one_hot * grip_another, dim=-1, keepdim=True)
        
        tran_rot_grip_another = torch.cat([tran_another, el, grip_another], dim=1)


        return tran_rot_grip_another
    
    def update_compute_another_euler_relu(self, trans, eular, action_gripper_relative, bounds):
        bs = action_gripper_relative.shape[0]
        torch.clamp_(action_gripper_relative[:, :6], min=0, max=1)
        action_gripper_relative[:, :6] = action_gripper_relative[:, :6] * 2.0 - 1
        tran_relative = action_gripper_relative[:, :3] * self._voxel_size / 2.0
        trans_range = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        tran_relative = tran_relative * trans_range
        roll_pitch_yaw = action_gripper_relative[:, 3: 6] * math.pi
        
        
        tran_rot_relative = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=self._device)
        tran_rot_relative[:, :3, :3] = torch3d_tf.euler_angles_to_matrix(roll_pitch_yaw, "XYZ")
        # tran_rot_relative[:, :3, 3] = tran_relative
        
        
        tran_rot_one = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=self._device)
        # action_gripper_pose_one = torch.from_numpy(action_gripper_pose_one).to(device=self._device)
        tran_rot_one[:, :3, 3] = trans
        tran_rot_one[:, :3, :3] = torch3d_tf.euler_angles_to_matrix((eular * self._rotation_resolution - 180) / 180.0 * math.pi, "XYZ")

        tran_rot_another = torch.bmm(tran_rot_one, tran_rot_relative)

        tran_another = tran_rot_another[:, :3, 3] + tran_relative

        el = torch3d_tf.matrix_to_euler_angles(tran_rot_another[:, :3, :3], "ZYX") * 180 / math.pi + 180.0
        el = el / self._rotation_resolution
        el = torch.flip(el, dims=[1])
        grip_another = F.softmax(action_gripper_relative[:, -2:] / 0.01, dim=-1)
        grip_one_hot = torch.tensor([0,1]).to(self._device).unsqueeze(0)
        grip_one_hot = grip_one_hot.repeat(bs, 1)
        grip_another = torch.sum(grip_one_hot * grip_another, dim=-1, keepdim=True)
        
        tran_rot_grip_another = torch.cat([tran_another, el, grip_another], dim=1)


        return tran_rot_grip_another

    def update_compute_another_euler2(self, trans, eular, action_gripper_relative, bounds):
        bs = action_gripper_relative.shape[0]
        tran_relative = action_gripper_relative[:, :3] * self._voxel_size / 2.0
        trans_range = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        tran_relative = tran_relative * trans_range
        roll_pitch_yaw = (action_gripper_relative[:, 3: 6] * 180 + 180) / self._rotation_resolution
        


        tran_another = trans + tran_relative

        el = (eular + roll_pitch_yaw) % self._num_rotation_classes
        
        grip_another = F.softmax(action_gripper_relative[:, -2:] / 0.01, dim=-1)
        grip_one_hot = torch.tensor([0,1]).to(self._device).unsqueeze(0)
        grip_one_hot = grip_one_hot.repeat(bs, 1)
        grip_another = torch.sum(grip_one_hot * grip_another, dim=-1, keepdim=True)
        
        tran_rot_grip_another = torch.cat([tran_another, el, grip_another], dim=1)


        return tran_rot_grip_another
    

    def update_compute_another_euler3(self, trans, eular, action_gripper_relative, bounds):
        bs = action_gripper_relative.shape[0]
        tran_relative = action_gripper_relative[:, :3] * self._voxel_size / 2.0
        trans_range = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        tran_relative = tran_relative * trans_range
        roll_pitch_yaw = action_gripper_relative[:, 3: 6] * math.pi
        
        
        tran_rot_relative = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=self._device)
        tran_rot_relative[:, :3, :3] = torch3d_tf.euler_angles_to_matrix(roll_pitch_yaw, "XYZ")
        tran_rot_relative[:, :3, 3] = tran_relative
        
        
        tran_rot_one = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=self._device)
        # action_gripper_pose_one = torch.from_numpy(action_gripper_pose_one).to(device=self._device)
        tran_rot_one[:, :3, 3] = trans
        tran_rot_one[:, :3, :3] = torch3d_tf.euler_angles_to_matrix((eular * self._rotation_resolution - 180) / 180.0 * math.pi, "XYZ")

        tran_rot_another = torch.bmm(tran_rot_one, tran_rot_relative)

        tran_another = tran_rot_another[:, :3, 3]

        el = torch3d_tf.matrix_to_euler_angles(tran_rot_another[:, :3, :3], "ZYX") * 180 / math.pi + 180.0
        el = el / self._rotation_resolution
        el = torch.flip(el, dims=[1])
        grip_another = F.softmax(action_gripper_relative[:, -2:] / 0.01, dim=-1)
        grip_one_hot = torch.tensor([0,1]).to(self._device).unsqueeze(0)
        grip_one_hot = grip_one_hot.repeat(bs, 1)
        grip_another = torch.sum(grip_one_hot * grip_another, dim=-1, keepdim=True)
        
        tran_rot_grip_another = torch.cat([tran_another, el, grip_another], dim=1)


        return tran_rot_grip_another
    
    def move_pc_in_bound(self, pc, img_feat, bounds):
        """
        :param no_op: no operation
        """

        # 重塑点云为 [bs, H*W, 3]
        bs, _, H, W = pc.shape
        pc_flat = pc.permute(0, 2, 3, 1).reshape(bs, H*W, 3)  # [bs, H, W, 3] -> [bs, H*W, 3]
        
        # 重塑图像特征为 [bs, H*W, C]
        if img_feat is not None:
            C = img_feat.shape[1]
            feat_flat = img_feat.permute(0, 2, 3, 1).reshape(bs, H*W, C)
        else:
            feat_flat = None

        x_min, y_min, z_min, x_max, y_max, z_max = bounds
        
        # 创建无效点掩码 [bs, H*W]
        inv_pnt = (
            (pc_flat[:, :, 0] < x_min) | (pc_flat[:, :, 0] > x_max) |
            (pc_flat[:, :, 1] < y_min) | (pc_flat[:, :, 1] > y_max) |
            (pc_flat[:, :, 2] < z_min) | (pc_flat[:, :, 2] > z_max) |
            torch.isnan(pc_flat[:, :, 0]) |
            torch.isnan(pc_flat[:, :, 1]) |
            torch.isnan(pc_flat[:, :, 2])
        )

        # 过滤有效点
        filtered_pc = []
        filtered_feat = [] if img_feat is not None else None
        
        for i in range(bs):
            valid_mask = ~inv_pnt[i]
            # 点云数据 [valid_points, 3]
            filtered_pc.append(pc_flat[i][valid_mask])
            
            if feat_flat is not None:
                # 特征数据 [valid_points, C]
                filtered_feat.append(feat_flat[i][valid_mask])

        return filtered_pc, filtered_feat
    

    # baseline #2 v3
    def update(self, step: int, replay_sample: dict) -> dict:
        action_trans_right = replay_sample['right_trans_action_indicies'][:, self._layer * 3:self._layer * 3 + 3].int()
        action_rot_grip_right = replay_sample['right_rot_grip_action_indicies'].int()
        action_gripper_pose_right = replay_sample['right_gripper_pose']
        action_trans_left = replay_sample['left_trans_action_indicies'][:, self._layer * 3:self._layer * 3 + 3].int()
        action_rot_grip_left = replay_sample['left_rot_grip_action_indicies'].int()
        action_gripper_pose_left = replay_sample['left_gripper_pose']
        action_ignore_collisions = replay_sample['right_ignore_collisions'].int()
        lang_goal_emb = replay_sample['lang_goal_emb'].float()
        lang_token_embs = replay_sample['lang_token_embs'].float()
        prev_layer_voxel_grid = replay_sample.get('prev_layer_voxel_grid', None)
        prev_layer_bounds = replay_sample.get('prev_layer_bounds', None)
        device = self._device

        bounds = self._coordinate_bounds.to(device)
        if self._layer > 0:
            cp = replay_sample['attention_coordinate_layer_%d' % (self._layer - 1)]
            bounds = torch.cat([cp - self._bounds_offset, cp + self._bounds_offset], dim=1)

        proprio_right, proprio_left = None, None
        if self._include_low_dim_state:
            proprio_right = replay_sample['right_low_dim_state']
            proprio_left = replay_sample['left_low_dim_state']

        # NOTE: right now, we're feeding wrist and wrist2 images
        obs, pcd = self._preprocess_inputs(replay_sample)

        # batch size
        bs = pcd[0].shape[0]

        # SE(3) augmentation of point clouds and actions
        if self._transform_augmentation:
            # left and right arms need to have the same augmentations (only 1 pcd is returned).
            aug_succ, \
            action_trans_right, \
            action_rot_grip_right, \
            action_trans_left, \
            action_rot_grip_left, \
            pcd, \
            action_gripper_pose_right, \
            action_gripper_pose_left = apply_se3_augmentation_2Robots_relative(pcd,
                                         action_gripper_pose_right,
                                         action_trans_right,
                                         action_rot_grip_right,
                                         action_gripper_pose_left,
                                         action_trans_left,
                                         action_rot_grip_left,
                                         bounds,
                                         self._layer,
                                         self._transform_augmentation_xyz,
                                         self._transform_augmentation_rpy,
                                         self._transform_augmentation_rot_resolution,
                                         self._voxel_size,
                                         self._rotation_resolution,
                                         self._device)
            if not aug_succ:
                return {
                    'trans_loss': -1,
                    'rot_loss': -1,
                    'relative_loss': -1,
                    'total_loss': -1,
                    'prev_layer_voxel_grid': prev_layer_voxel_grid,
                    'prev_layer_bounds': prev_layer_bounds,
                }
            

        # print("label:")
        # print(action_gripper_pose_right)
        # print(action_gripper_pose_left)
        # print(action_trans_right)
        # print(action_trans_left)
        # forward pass
        q_trans_one, q_rot_grip_one, \
        q_collision_one, \
        q_trans_and_rot_relative, \
        q_arm_pred, \
        voxel_grid = self._q(obs,
                             proprio_right,
                             proprio_left,
                             pcd,
                             lang_goal_emb,
                             lang_token_embs,
                             bounds,
                             prev_layer_bounds,
                             prev_layer_voxel_grid)

        # argmax to choose best action
        coords_one, \
        rot_and_grip_indicies_one, \
        ignore_collision_indicies_one = self._q.choose_highest_action(q_trans_one, q_rot_grip_one, q_collision_one)

        

        
        # choose arm
        action_gripper_pose_one = action_gripper_pose_right
        action_trans_one = action_trans_right
        action_rot_grip_one = action_rot_grip_right

        # action_gripper_pose_another = torch.cat([action_gripper_pose_left, action_rot_grip_left[:, -1].unsqueeze(1)], dim=1)
        action_gripper_pose_another = torch.cat([action_gripper_pose_left[:, :3], action_rot_grip_left[:, :3] / self._num_rotation_classes, action_rot_grip_left[:, -1].unsqueeze(1)], dim=1)
        arm_choose = torch.tensor([[0, 1]]).to(device=self._device)
        arm_choose = arm_choose.repeat(bs, 1)
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        # attention_coordinate_one = bounds[:, :3] + res * coords_one + res / 2
        # rot_indicies_one = rot_and_grip_indicies_one[:, :3]
        
        attention_coordinate_one = bounds[:, :3] + res * action_trans_one + res / 2
        rot_indicies_one = action_rot_grip_one[:, :3]
        # coords_left, \
        # rot_and_grip_indicies_left, \
        # ignore_collision_indicies_left = self._q.choose_highest_action(q_trans_left, q_rot_grip_left, q_collision_left)
        # q_action_gripper_pose_another = self.compute_another2(action_gripper_pose_one, q_trans_and_rot_relative, bounds)

        q_action_gripper_pose_another = self.update_compute_another_euler(attention_coordinate_one, rot_indicies_one, q_trans_and_rot_relative, bounds)

        q_action_gripper_pose_another[:, 3:6] = q_action_gripper_pose_another[:, 3:6] / self._num_rotation_classes

        q_trans_loss, q_rot_loss, q_grip_loss, q_collision_loss, q_relative_loss, q_arm_loss = 0., 0., 0., 0., 0., 0.
        
        q_arm_loss = self._celoss(q_arm_pred, arm_choose)

        q_relative_trans_loss = self._mseloss(q_action_gripper_pose_another[:, :3], action_gripper_pose_another[:, :3])
        q_relative_rot_loss = self.circular_mse_loss(q_action_gripper_pose_another[:, 3:6], action_gripper_pose_another[:, 3:6])
        q_relative_grip_loss = self._mseloss(q_action_gripper_pose_another[:, 6:], action_gripper_pose_another[:, 6:])
        q_relative_loss = q_relative_trans_loss + q_relative_rot_loss + q_relative_grip_loss
        # q_relative_loss = self._mseloss(q_action_gripper_pose_another, action_gripper_pose_another)
        # q_zeros = torch.zeros(q_trans_and_rot_relative[:, :6].shape)
        # q_l2_loss = torch.mean((torch.maximum(torch.abs(q_trans_and_rot_relative[:, :6]) - 1, 0)) ** 2)
        # translation one-hot
        action_trans_one_hot_one = self._action_trans_one_hot_zeros.clone()
        for b in range(bs):
            gt_coord = action_trans_one[b, :].int()
            action_trans_one_hot_one[b, :, gt_coord[0], gt_coord[1], gt_coord[2]] = 1


        # translation loss
        q_trans_flat_one = q_trans_one.view(bs, -1)
        action_trans_one_hot_flat_one = action_trans_one_hot_one.view(bs, -1)
        q_trans_loss += self._celoss(q_trans_flat_one, action_trans_one_hot_flat_one)


        with_rot_and_grip_one = rot_and_grip_indicies_one is not None
        if with_rot_and_grip_one:
            # rotation, gripper, and collision one-hots
            action_rot_x_one_hot_one = self._action_rot_x_one_hot_zeros.clone()
            action_rot_y_one_hot_one = self._action_rot_y_one_hot_zeros.clone()
            action_rot_z_one_hot_one = self._action_rot_z_one_hot_zeros.clone()
            action_grip_one_hot_one = self._action_grip_one_hot_zeros.clone()
            action_ignore_collisions_one_hot = self._action_ignore_collisions_one_hot_zeros.clone()

            for b in range(bs):
                gt_rot_grip = action_rot_grip_one[b, :].int()
                action_rot_x_one_hot_one[b, gt_rot_grip[0]] = 1
                action_rot_y_one_hot_one[b, gt_rot_grip[1]] = 1
                action_rot_z_one_hot_one[b, gt_rot_grip[2]] = 1
                action_grip_one_hot_one[b, gt_rot_grip[3]] = 1

                gt_ignore_collisions = action_ignore_collisions[b, :].int()
                action_ignore_collisions_one_hot[b, gt_ignore_collisions[0]] = 1

            # flatten predictions
            q_rot_x_flat_one = q_rot_grip_one[:, 0*self._num_rotation_classes:1*self._num_rotation_classes]
            q_rot_y_flat_one = q_rot_grip_one[:, 1*self._num_rotation_classes:2*self._num_rotation_classes]
            q_rot_z_flat_one = q_rot_grip_one[:, 2*self._num_rotation_classes:3*self._num_rotation_classes]
            q_grip_flat_one =  q_rot_grip_one[:, 3*self._num_rotation_classes:]

            # rotation loss
            q_rot_loss += self._celoss(q_rot_x_flat_one, action_rot_x_one_hot_one)
            q_rot_loss += self._celoss(q_rot_y_flat_one, action_rot_y_one_hot_one)
            q_rot_loss += self._celoss(q_rot_z_flat_one, action_rot_z_one_hot_one)

            # gripper loss
            q_grip_loss += self._celoss(q_grip_flat_one, action_grip_one_hot_one)

            # collision loss
            q_collision_loss += self._celoss(q_collision_one, action_ignore_collisions_one_hot)


        combined_losses = (q_trans_loss * self._trans_loss_weight) + \
                          (q_rot_loss * self._rot_loss_weight) + \
                          (q_grip_loss * self._grip_loss_weight) + \
                          (q_collision_loss * self._collision_loss_weight) + \
                          (q_relative_loss * self._relative_loss_weight) + \
                          (q_arm_loss * self._arm_loss_weight)
        total_loss = combined_losses.mean()
        # print("trans:", q_trans_loss)
        # print("rot:", q_rot_loss)
        # print("relative:", q_relative_loss)
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        self._summaries = {
            'losses/total_loss': total_loss,
            'losses/trans_loss': q_trans_loss.mean(),
            'losses/rot_loss': q_rot_loss.mean() if with_rot_and_grip_one else 0.,
            'losses/grip_loss': q_grip_loss.mean() if with_rot_and_grip_one else 0.,
            'losses/collision_loss': q_collision_loss.mean() if with_rot_and_grip_one else 0.,
            'losses/relative_loss': q_relative_loss.mean()
        }

        if self._lr_scheduler:
            self._scheduler.step()
            self._summaries['learning_rate'] = self._scheduler.get_last_lr()[0]

        self._vis_voxel_grid = voxel_grid[0]
        self._vis_translation_qvalue_one = self._softmax_q_trans(q_trans_one[0])
        self._vis_max_coordinate_one = coords_one[0]
        self._vis_gt_coordinate_one = action_trans_one[0]

        # Note: PerAct doesn't use multi-layer voxel grids like C2FARM
        # stack prev_layer_voxel_grid(s) from previous layers into a list
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [voxel_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [voxel_grid]

        # stack prev_layer_bound(s) from previous layers into a list
        if prev_layer_bounds is None:
            prev_layer_bounds = [self._coordinate_bounds.repeat(bs, 1)]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        # debug: make sure voxel grid looks alright
        # visualise_voxel_2robots(
        #     self._vis_voxel_grid.detach().cpu().numpy(),
        #     self._vis_translation_qvalue_right.detach().cpu().numpy(),
        #     self._vis_max_coordinate_right.detach().cpu().numpy(),
        #     self._vis_gt_coordinate_right.detach().cpu().numpy(),
        #     self._vis_translation_qvalue_left.detach().cpu().numpy(),
        #     self._vis_max_coordinate_left.detach().cpu().numpy(),
        #     self._vis_gt_coordinate_left.detach().cpu().numpy(),
        #     show=True)
        if torch.isnan(total_loss).any():
            print(action_trans_right)
            print(action_trans_left)
            print(q_trans_and_rot_relative)
            print(q_trans_one)
            print(replay_sample['task'])
            
        return {
            'trans_loss': q_trans_loss.mean(),
            'rot_loss': q_rot_loss.mean(),
            'relative_loss': q_relative_loss.mean(),
            'total_loss': total_loss,
            'prev_layer_voxel_grid': prev_layer_voxel_grid,
            'prev_layer_bounds': prev_layer_bounds,
        }

    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        deterministic = True
        bounds = self._coordinate_bounds
        prev_layer_voxel_grid = observation.get('prev_layer_voxel_grid', None)
        prev_layer_bounds = observation.get('prev_layer_bounds', None)
        lang_goal_tokens = observation.get('lang_goal_tokens', None).long()

        # extract CLIP language embs
        with torch.no_grad():
            lang_goal_tokens = lang_goal_tokens.to(device=self._device)
            lang_goal_emb, lang_token_embs = self._clip_rn50.encode_text_with_embeddings(lang_goal_tokens[0])

        # voxelization resolution
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        max_rot_index = int(360 // self._rotation_resolution)
        proprio_right, proprio_left = None, None

        proprio_right = observation['right_low_dim_state']
        proprio_left = observation['left_low_dim_state']

        # NOTE: right now, we're feeding wrist and wrist2 images
        obs, pcd = self._act_preprocess_inputs(observation)

        # correct batch size and device
        obs = [[o[0][0].to(self._device), o[1][0].to(self._device)] for o in obs]
        proprio_right = proprio_right[0].to(self._device)
        proprio_left = proprio_left[0].to(self._device)
        pcd = [p[0].to(self._device) for p in pcd]
        lang_goal_emb = lang_goal_emb.to(self._device)
        lang_token_embs = lang_token_embs.to(self._device)
        bounds = torch.as_tensor(bounds, device=self._device)
        prev_layer_voxel_grid = prev_layer_voxel_grid.to(self._device) if prev_layer_voxel_grid is not None else None
        prev_layer_bounds = prev_layer_bounds.to(self._device) if prev_layer_bounds is not None else None

        # inference
        q_trans_one, q_rot_grip_one, \
        q_collision_one, \
        q_trans_and_rot_relative, \
        q_arm_pred, \
        voxel_grid = self._q(obs,
                           proprio_right,
                           proprio_left,
                           pcd,
                           lang_goal_emb,
                           lang_token_embs,
                           bounds,
                           prev_layer_bounds,
                           prev_layer_voxel_grid)

        # softmax Q predictions
        q_trans_one = self._softmax_q_trans(q_trans_one)
        q_rot_grip_one =  self._softmax_q_rot_grip(q_rot_grip_one) if q_rot_grip_one is not None else q_rot_grip_one
        q_ignore_collisions_one = self._softmax_ignore_collision(q_collision_one) \
            if q_collision_one is not None else q_collision_one
        q_arm_pred = self._softmax_pred_arm(q_arm_pred)
        # argmax Q predictions
        coords_one, \
        rot_and_grip_indicies_one, \
        ignore_collisions_one = self._q.choose_highest_action(q_trans_one, q_rot_grip_one, q_ignore_collisions_one)
        pred_arm_index = q_arm_pred[:, -2:].argmax(-1, keepdim=True)

        rot_grip_action_one = rot_and_grip_indicies_one if q_rot_grip_one is not None else None
        ignore_collisions_action_one = ignore_collisions_one.int() if ignore_collisions_one is not None else None
        pred_arm_index = pred_arm_index.int()
        
        coords_one = coords_one.int()
        attention_coordinate_one = bounds[:, :3] + res * coords_one + res / 2
        rot_action_one = rot_grip_action_one[:, :3]

        # pose_another = self.act_compute_another2(attention_coordinate_one, rot_grip_action_one[:, :3], q_trans_and_rot_relative, bounds)
        pose_another = self.update_compute_another_euler(attention_coordinate_one, rot_action_one, q_trans_and_rot_relative, bounds)
        if pred_arm_index[0] > 0.5:
            relative_arm = 'left'
        else:
            relative_arm = 'right'
        eps = 0.0001
        torch.clamp_(pose_another[:, 0], bounds[0,0] + eps, bounds[0,3] - eps)
        torch.clamp_(pose_another[:, 1], bounds[0,1] + eps, bounds[0,4] - eps)
        torch.clamp_(pose_another[:, 2], bounds[0,2] + eps, bounds[0,5] - eps)

        # stack prev_layer_voxel_grid(s) into a list
        # NOTE: PerAct doesn't used multi-layer voxel grids like C2FARM
        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [voxel_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [voxel_grid]

        if prev_layer_bounds is None:
            prev_layer_bounds = [bounds]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        observation_elements = {
            'attention_coordinate_one': attention_coordinate_one,
            'prev_layer_voxel_grid': prev_layer_voxel_grid,
            'prev_layer_bounds': prev_layer_bounds,
        }
        info = {
            'voxel_grid_depth%d' % self._layer: voxel_grid,
            'q_depth_one%d' % self._layer: q_trans_one,
            'voxel_idx_depth_one%d' % self._layer: coords_one,
        }
        self._act_voxel_grid = voxel_grid[0]
        self._act_max_coordinate_one = coords_one[0]
        self._act_qvalues_one = q_trans_one[0].detach()

        # debug: make sure voxel grid looks alright
        # visualise_voxel_2robots(
        #     vox_grid[0].detach().cpu().numpy(),
        #     q_trans_right[0].detach().cpu().numpy(),
        #     coords_right[0].detach().cpu().numpy(),
        #     None,
        #     q_trans_left[0].detach().cpu().numpy(),
        #     coords_left[0].detach().cpu().numpy(),
        #     None,
        #     show=True)

        return ActResult((coords_one, rot_grip_action_one, ignore_collisions_action_one, pose_another, relative_arm),
                         observation_elements=observation_elements,
                         info=info)

    def update_summaries(self) -> List[Summary]:
        try:
            summaries = [
                ImageSummary('%s/update_qattention' % self._name,
                            transforms.ToTensor()(visualise_voxel(
                                self._vis_voxel_grid.detach().cpu().numpy(),
                                self._vis_translation_qvalue_one.detach().cpu().numpy(),
                                self._vis_max_coordinate_one.detach().cpu().numpy(),
                                self._vis_gt_coordinate_one.detach().cpu().numpy())))
            ]
        except:
            # this exception can happen when the computer does not have a display or the display is not set up properly
            summaries = []

        wandb_dict = {}
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary('%s/%s' % (self._name, n), v))
            if self._wandb_run is not None:
                wandb_dict['%s/%s' % (self._name, n)] = v

        for (name, crop) in (self._crop_summary):
            crops = (torch.cat(torch.split(crop, 3, dim=1), dim=3) + 1.0) / 2.0
            summaries.extend([
                ImageSummary('%s/crops/%s' % (self._name, name), crops)])

        for tag, param in self._q.named_parameters():
            # assert not torch.isnan(param.grad.abs() <= 1.0).all()
            summaries.append(
                HistogramSummary('%s/gradient/%s' % (self._name, tag),
                                 param.grad))
            summaries.append(
                HistogramSummary('%s/weight/%s' % (self._name, tag),
                                 param.data))

        return summaries, wandb_dict

    def act_summaries(self) -> List[Summary]:
        return [
            ImageSummary('%s/act_Qattention' % self._name,
                         transforms.ToTensor()(visualise_voxel(
                             self._act_voxel_grid.cpu().numpy(),
                             self._act_qvalues_one.cpu().numpy(),
                             self._act_max_coordinate_one.cpu().numpy(),
                             )))]

    def load_weights(self, savedir: str):
        device = self._device if not self._training else torch.device('cuda:%d' % self._device)
        weight_file = os.path.join(savedir, '%s.pt' % self._name)
        state_dict = torch.load(weight_file, map_location=device)

        # load only keys that are in the current model
        merged_state_dict = self._q.state_dict()
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace('_qnet.module', '_qnet')
            if k in merged_state_dict:
                merged_state_dict[k] = v
            else:
                if '_voxelizer' not in k:
                    logging.warning("key %s not found in checkpoint" % k)
        if not self._training:
            # reshape voxelizer weights
            b = merged_state_dict["_voxelizer._ones_max_coords"].shape[0]
            merged_state_dict["_voxelizer._ones_max_coords"] = merged_state_dict[
                "_voxelizer._ones_max_coords"
            ][0:1]
            flat_shape = merged_state_dict["_voxelizer._flat_output"].shape[0]
            merged_state_dict["_voxelizer._flat_output"] = merged_state_dict[
                "_voxelizer._flat_output"
            ][0 : flat_shape // b]
            merged_state_dict["_voxelizer._tiled_batch_indices"] = merged_state_dict[
                "_voxelizer._tiled_batch_indices"
            ][0:1]
            merged_state_dict["_voxelizer._index_grid"] = merged_state_dict[
                "_voxelizer._index_grid"
            ][0:1]
        
        self._q.load_state_dict(merged_state_dict)
        print("loaded weights from %s" % weight_file)

    def load_weight(self, ckpt_file: str):
        device = self._device if not self._training else torch.device('cuda:%d' % self._device)
        state_dict = torch.load(ckpt_file, map_location=device)

        # load only keys that are in the current model
        merged_state_dict = self._q.state_dict()
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace('_qnet.module', '_qnet')
            if k in merged_state_dict:
                merged_state_dict[k] = v
            else:
                if '_voxelizer' not in k:
                    logging.warning("key %s not found in checkpoint" % k)
        self._q.load_state_dict(merged_state_dict)
        print("loaded weights from %s" % ckpt_file)

    def save_weights(self, savedir: str):
        torch.save(
            self._q.state_dict(), os.path.join(savedir, '%s.pt' % self._name))
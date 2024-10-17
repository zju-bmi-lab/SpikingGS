#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import sys
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn, Tensor
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from arguments import PipelineParams, ArgumentParser
import math
import torch.nn.functional as F
from scene.appearance_network import AppearanceNetwork

class BilateralFilter(torch.nn.Module):
    def __init__(self, d, sigmaColor, sigmaSpace):
        super(BilateralFilter, self).__init__()
        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace

    def forward(self, input):
        if input.dim() == 3:
            input = input.unsqueeze(0)
        if self.d == 0:
            return input
        N, C, H, W = input.size()
        
        # Prepare Gaussian spatial kernel
        grid = torch.stack(torch.meshgrid(torch.arange(-self.d, self.d + 1), torch.arange(-self.d, self.d + 1)), dim=-1)
        spatial_weight = torch.exp(-0.5 * (grid ** 2).sum(dim=-1, keepdim=True) / (self.sigmaSpace ** 2))
        spatial_weight = spatial_weight / spatial_weight.sum()
        spatial_weight = spatial_weight.to(input.device)
        
        # Pad the input for border handling
        padded_input = F.pad(input, (self.d, self.d, self.d, self.d), mode='reflect')
        
        # Apply bilateral filter
        output = torch.zeros_like(input)
        for i in range(-self.d, self.d + 1):
            for j in range(-self.d, self.d + 1):
                if i == 0 and j == 0:
                    continue
                shifted_input = padded_input[:, :, self.d + i:H + self.d + i, self.d + j:W + self.d + j]
                color_weight = torch.exp(-0.5 * (shifted_input - input) ** 2 / (self.sigmaColor ** 2))
                weight = spatial_weight[self.d + i, self.d + j] * color_weight
                output += weight * shifted_input
        
        
        return output

class EdgeAwareTV(nn.Module):
    """Edge Aware Smooth Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, depth: Tensor, rgb: Tensor):
        """
        Args:
            depth: [ H, W, 1]
            rgb: [ H, W, 3]
        """
        grad_depth_x = torch.abs(depth[:, :-1, :] - depth[:, 1:, :])
        grad_depth_y = torch.abs(depth[:-1, :, :] - depth[1:, :, :])

        grad_img_x = torch.mean(
            torch.abs(rgb[:, :-1, :] - rgb[:, 1:, :]), -1, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(rgb[:-1, :, :] - rgb[1:, :, :]), -1, keepdim=True
        )

        grad_depth_x *= torch.exp(-grad_img_x)
        grad_depth_y *= torch.exp(-grad_img_y)

        return grad_depth_x.mean() + grad_depth_y.mean()


class TVLoss(nn.Module):
    """TV loss"""

    def __init__(self):
        super().__init__()

    def forward(self, pred):
        """
        Args:
            pred: [ H, W, 3]

        Returns:
            tv_loss: [1]
        """
        h_diff = pred[:, :-1, :] - pred[:, 1:, :]
        w_diff = pred[:-1, :, :] - pred[1:, :, :]
        return torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))

class SpikingNeuron(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, thresh):
        out = (input >= thresh).float()
        ctx.save_for_backward(input, out, thresh)
        return out*input

    @staticmethod
    def backward(ctx, grad_output):
        input, out, thresh = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad = grad_input * out
        lm = 10.0
        k = 0.1
        grad_Vth = -lm * input * grad_input * ((k - torch.abs(input - thresh)) / k ** 2).clamp(min = 0)
        #print(torch.sum(grad_Vth.detach()))
        return grad, grad_Vth

class SurrGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, thresh):
        out = (input >= thresh).float()
        ctx.save_for_backward(input, out, thresh)
        return out*input

    @staticmethod
    def backward(ctx, grad_output):
        input, out, thresh = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad = grad_input * out
        lm = 10.0
        k = 0.1
        grad_Vth = -lm * input * grad_input * ((k - torch.abs(input - thresh)) / k ** 2).clamp(min = 0)
        return grad, grad_Vth

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.Vth_opa = torch.empty(0)
        self.Vth_pdf = torch.empty(0)

        SN = SpikingNeuron.apply
        self.spike_neuron = SN

        Surr = SurrGrad.apply
        self.surr = Surr
        
        self.bi_filter = None
        self.EdgeAwareTVLoss = EdgeAwareTV()
        self.TVLoss = TVLoss()

        self.appearance_network = AppearanceNetwork(3+64, 3).cuda()
        
        std = 1e-4
        self._appearance_embeddings = nn.Parameter(torch.empty(2048, 64).cuda())
        self._appearance_embeddings.data.normal_(0, std)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)


    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_Vth_pdf(self):
        #print(self.Vth_pdf.shape)
        return torch.abs(self.Vth_pdf)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_real_opa(self, t = 12., return_k=False):
        opacity = self.spike_neuron(self.get_opacity, self.Vth_opa)
        if not return_k:
            return opacity
        else:
            return opacity, opacity
    
    def get_apperance_embedding(self, idx):
        return self._appearance_embeddings[idx]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        scales[..., -1] -= 1e10 # squeeze z scaling
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True)) 
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.Vth_opa = nn.Parameter(torch.tensor([0.001]).to(device="cuda").requires_grad_(True))
        self.Vth_pdf = nn.Parameter(0.01 * torch.ones((self.get_xyz.shape[0], 1), dtype=torch.float, device="cuda"))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self.Vth_opa], 'lr': training_args.vth_lr, "name": "Vth_opa"},
            {'params': [self.Vth_pdf], 'lr': training_args.vth_lr, "name": "Vth_pdf"},
            {'params': [self._appearance_embeddings], 'lr': training_args.appearance_embeddings_lr, "name": "appearance_embeddings"},
            {'params': self.appearance_network.parameters(), 'lr': training_args.appearance_network_lr, "name": "appearance_network"}
        ]
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration, reset_freq, pipe):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "Vth_opa":
                if (iteration % reset_freq >= 0) and (iteration % reset_freq <= 300) and (iteration <= 18000):
                    lr = 0.0
                elif (iteration <= 18000):
                    lr = 0.0002
                elif (iteration <= 25000):
                    lr = 0.00014
                else:
                    lr = 0.00005
                lr *= 3
                if pipe.no_spike:
                    lr = 0
                param_group['lr'] = lr
            if param_group["name"] == "Vth_pdf":
                # if (iteration % reset_freq >= 0) and (iteration % reset_freq <= 300) and (iteration <= 19000):
                #     lr = 0.0
                # elif (iteration <= 14000):
                #     lr = 0.0002
                # else:
                #     lr = 0.00014
                # lr = 0.0001
                if (iteration % reset_freq >= 0) and (iteration % reset_freq <= 300) and (iteration <= 18000):
                    lr = 0.0
                else:
                    lr = 0.0002
                lr *= 3
                if pipe.no_cut:
                    lr = 0
                param_group['lr'] = lr
        return lr
        

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.get_opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def save_npz(self, path):
        mkdir_p(os.path.dirname(path))

        save_dict = dict()
        
        save_dict["xyz"] = self._xyz.detach().cpu().numpy()
        save_dict["opacity"] = self._opacity.detach().cpu().half().numpy()
        save_dict["scale"] = self._scaling.detach().cpu().numpy()
        save_dict["rotation"] = self._rotation.detach().cpu().numpy()
        save_dict["f_dc"] = self._features_dc.detach().transpose(1, 2).contiguous().cpu().numpy()
        save_dict["f_rest"] = self._features_rest.detach().transpose(1, 2).contiguous().cpu().numpy()
        save_dict["Vth_opa"] = self.Vth_opa.cpu().numpy()
        save_dict["Vth_pdf"] = self.Vth_pdf.cpu().numpy()
        
        np.savez(path, **save_dict)

    def reset_opacity_spike(self):
        _, k = self.get_real_opa(return_k=True)
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * max((self.Vth_opa.item()), 0.003)))
        if self.opacity_activation(opacities_new.max()) < self.Vth_opa:
            opacities_new = opacities_new + 1e-3

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_model(self, path):
        if os.path.isfile(path + '.npz'):
            path = path + '.npz'
            print("Loading ", path)
            load_dict = np.load(path, allow_pickle=True)

            self._xyz = nn.Parameter(torch.from_numpy(load_dict["xyz"]).cuda().float().requires_grad_(True))
            self._opacity = nn.Parameter(torch.from_numpy(load_dict["opacity"]).reshape(-1,1).cuda().float().requires_grad_(True))
            length = 3*(self.max_sh_degree + 1) ** 2 - 3
            shape_rest = (self._xyz.shape[0], length)
            shape_dc = (self._xyz.shape[0], 3, 1)
            self._features_dc = nn.Parameter(torch.from_numpy(load_dict["f_dc"]).cuda().transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(torch.from_numpy(load_dict["f_rest"]).cuda().transpose(1, 2).contiguous().requires_grad_(True))
            self._scaling = nn.Parameter(torch.from_numpy(load_dict["scale"]).cuda().requires_grad_(True))
            self._rotation = nn.Parameter(torch.from_numpy(load_dict["rotation"]).cuda().requires_grad_(True))
            self.Vth_opa = nn.Parameter(torch.from_numpy(load_dict["Vth_opa"]).cuda().requires_grad_(True))
            self.Vth_pdf = nn.Parameter(torch.from_numpy(load_dict["Vth_pdf"]).cuda().requires_grad_(True))
            self.active_sh_degree = self.max_sh_degree
        else:
            self.load_ply(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        
        state_dict = torch.load(path +".pth")
        module_list_state_dict = state_dict['module_list']
        self.Vth_opa = state_dict['Vth_opa']
        self.Vth_pdf = state_dict['Vth_pdf']
        

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # if group["name"] == 'Vth_opa':
            #   continue
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'Vth_opa':
              continue
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.Vth_pdf = optimizable_tensors["Vth_pdf"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'Vth_opa':
              continue
            if group["name"] in ["appearance_embeddings", "appearance_network"]:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_vpdf):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "Vth_pdf": new_vpdf}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.Vth_pdf = optimizable_tensors["Vth_pdf"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_scaling[:, -1] = -1e10
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_vpdf = self.Vth_pdf[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_vpdf)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_vpdf = self.Vth_pdf[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,new_opacities, new_scaling, new_rotation, new_vpdf)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, visibility_filter):
       
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        opacity = self.get_real_opa()
        prune_mask = (opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    def spike_prune(self, min_opacity):
        opacity = self.get_real_opa()
        pdf = self.get_Vth_pdf
        
        opacity_mask = (opacity < min_opacity).squeeze()
        spike_mask = torch.logical_or(opacity_mask, (pdf >= 1.5).squeeze())
        self.prune_points(spike_mask)
        torch.cuda.empty_cache()

    def last_prune(self, min_opacity):
        opacity = self.get_real_opa()
        pdf = self.get_Vth_pdf
        
        opacity_mask = (opacity < min_opacity).squeeze()
        spike_mask = torch.logical_or(opacity_mask, (pdf >= 1.0).squeeze())
        self.prune_points(spike_mask)
        torch.cuda.empty_cache()

    def prune_big(self, extent, max_screen_size):
        big_points_vs = self.max_radii2D > max_screen_size * 0.1
        big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(big_points_vs, big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def densify_and_clone_scale(self, grads, grad_threshold, scene_extent, scale):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        scale_reg = scale
        threshold = scale_reg / 200.
        max_values, _ = torch.max(self.get_scaling, dim=1)
        mask = (max_values > (scale_reg - threshold)) & (max_values < (scale_reg + threshold))
        # print(mask.shape)
        selected_pts_mask = torch.logical_or(selected_pts_mask,mask)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_vpdf = self.Vth_pdf[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,new_opacities, new_scaling, new_rotation, new_vpdf)

    def densify_and_prune_scale(self, max_grad, min_opacity, extent, max_screen_size, radii, visibility_filter, scale):
       
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone_scale(grads, max_grad, extent, scale)

        torch.cuda.empty_cache()

    def densify_and_scale_split(self, grad_threshold, min_opacity, scene_extent, max_screen_size, scale_factor, scene_mask, N=2, no_grad=False):
        assert scale_factor > 0
        n_init_points = self.get_xyz.shape[0]
        scale_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent * scale_factor
        if max_screen_size:
            scale_mask = torch.logical_or(
                scale_mask,
                self.max_radii2D > max_screen_size
            )
        scale_mask = torch.logical_and(scene_mask, scale_mask)
        if no_grad:
            selected_pts_mask = scale_mask
        else:
            # Extract points that satisfy the gradient condition
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[:grads.shape[0]] = grads.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, scale_mask)

        
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_scaling[:, -1] = -1e10
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_vpdf = self.Vth_pdf[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_vpdf)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # prune_filter = torch.logical_or(prune_filter, (self.get_real_opa < -1).squeeze())
        self.prune_points(prune_filter)

        torch.cuda.empty_cache()
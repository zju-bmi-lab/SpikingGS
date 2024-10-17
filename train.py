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

import os
import torch
from torchmetrics.image import PeakSignalNoiseRatio
from random import randint, seed
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_through
import sys
from scene import Scene, GaussianModel, BilateralFilter
from utils.general_utils import safe_state
import uuid
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torch import nn
from lpipsPyTorch import lpips
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from torch.utils.tensorboard import SummaryWriter
TENSORBOARD_FOUND = True
from utils.depth_utils import depths_to_points, depth_to_normal
import kornia

def culling(xyz, cams, expansion=2):
    cam_centers = torch.stack([c.camera_center for c in cams], 0).to(xyz.device)
    span_x = cam_centers[:, 0].max() - cam_centers[:, 0].min()
    span_y = cam_centers[:, 1].max() - cam_centers[:, 1].min() # smallest span
    span_z = cam_centers[:, 2].max() - cam_centers[:, 2].min()

    scene_center = cam_centers.mean(0)

    span_x = span_x * expansion
    span_y = span_y * expansion
    span_z = span_z * expansion

    x_min = scene_center[0] - span_x / 2
    x_max = scene_center[0] + span_x / 2

    y_min = scene_center[1] - span_y / 2
    y_max = scene_center[1] + span_y / 2

    z_min = scene_center[2] - span_x / 2
    z_max = scene_center[2] + span_x / 2


    valid_mask = (xyz[:, 0] > x_min) & (xyz[:, 0] < x_max) & \
                 (xyz[:, 1] > y_min) & (xyz[:, 1] < y_max) & \
                 (xyz[:, 2] > z_min) & (xyz[:, 2] < z_max)
    # print(f'scene mask ratio {valid_mask.sum().item() / valid_mask.shape[0]}')

    return valid_mask, scene_center

def prune_low_contribution_gaussians(gaussians, cameras, pipe, bg, K=5, prune_ratio=0.1):
    top_list = [None, ] * K
    for i, cam in enumerate(cameras):
        trans = render(cam, gaussians, pipe, bg, record_transmittance=True)
        if top_list[0] is not None:
            m = trans > top_list[0]
            if m.any():
                for i in range(K - 1):
                    top_list[K - 1 - i][m] = top_list[K - 2 - i][m]
                top_list[0][m] = trans[m]
        else:
            top_list = [trans.clone() for _ in range(K)]

    contribution = torch.stack(top_list, dim=-1).mean(-1)
    tile = torch.quantile(contribution, prune_ratio)
    prune_mask = contribution < tile
    gaussians.prune_points(prune_mask)
    torch.cuda.empty_cache()

def L1_loss_appearance(image, gt_image, gaussians, view_idx, return_transformed_image=False):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    # center crop the image
    origH, origW = image.shape[1:]
    H = origH // 32 * 32
    W = origW // 32 * 32
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, top:top+H, left:left+W]
    
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
    
    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down)
    transformed_image = mapping_image * crop_image
    if not return_transformed_image:
        return l1_loss(transformed_image, crop_gt_image)
    else:
        transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
        return transformed_image

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, scale_regu, lambda_opa, mask_normal):
    print("scal:", scale_regu, "ld_opa:", lambda_opa)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.bi_filter = BilateralFilter(d=opt.fil_width, sigmaColor=50, sigmaSpace=50)
    gaussians.bi_filter.to(device="cuda")
    scene = Scene(dataset, gaussians)
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    if pipe.no_spike:
        gaussians.Vth_opa = nn.Parameter(torch.tensor([0.005]).to(device="cuda").requires_grad_(False))
    if pipe.no_cut:
        gaussians.Vth_pdf.requires_grad_(False)
        gaussians.Vth_pdf *= 0

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    all_cameras = scene.getTrainCameras().copy()
    for idx, camera in enumerate(scene.getTrainCameras()):
        camera.idx = idx
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration, opt.opacity_reset_interval, pipe)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        view_mask = viewpoint_cam.image_mask.squeeze(0).to(device="cuda")
        
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = 0.
        # use L1 loss for the transformed image if using decoupled appearance
        if dataset.use_decoupled_appearance:
            Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.idx)
        else:
            Ll1 = l1_loss(image, gt_image)
        lv_opa = 1 / (gaussians.Vth_opa + 1e-6)
        lv_pdf = 1 / (gaussians.get_Vth_pdf.mean() + 1e-6)
        max_values, _ = torch.max(gaussians.get_scaling[: ,:2], dim=1)

        scale_reg = scale_regu
        max_values = max_values[max_values > scale_reg]
        
        lscale = torch.sum(max_values)
        ssim_loss = ssim(image, gt_image)
        ld_opa = opt.ld_opa
        ld_pdf = opt.ld_pdf
        ld_scale = 0.0005 
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        lambda_smooth = opt.lambda_smooth if iteration > -1 else 0.0
        lambda_tv_normal = 0.0 if (iteration > 0 and iteration <= 2000) else 0.0
        lambda_tv_depth = opt.lambda_tv_d if (iteration <= 4000) else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        surf_depth = render_pkg['surf_depth']

        depth_map = render_pkg["surf_depth"][0]
        if opt.depth_grad_thresh > 0:
            depth_map_for_grad = depth_map[None, None]
            sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=depth_map.dtype, device=depth_map.device).view(1, 1, 3, 3)
            sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=depth_map.dtype, device=depth_map.device).view(1, 1, 3, 3)

            depth_map_for_grad = F.pad(depth_map_for_grad, pad=(1, 1, 1, 1), mode="replicate")
            depth_grad_x = F.conv2d(depth_map_for_grad, sobel_kernel_x) / 3
            depth_grad_y = F.conv2d(depth_map_for_grad, sobel_kernel_y) / 3
            depth_grad_mag = torch.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2)
            depth_grad_weight = (depth_grad_mag < opt.depth_grad_thresh).float()
            depth_grad_mask_dilation = 1
            mask_di = depth_grad_mask_dilation
            depth_grad_weight = -1 * F.max_pool2d(-1 * depth_grad_weight, mask_di * 2 + 1, stride=1, padding=mask_di)
            depth_grad_weight = depth_grad_weight.squeeze().detach()
            normal_error = (1 - (rend_normal * surf_normal * view_mask.unsqueeze(0)).sum(dim=0)) * depth_grad_weight
            normal_error = normal_error[None]
        else:
            if (1 - mask_normal):
                normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            else:
                normal_error = (1 - (rend_normal * surf_normal * view_mask.unsqueeze(0)).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        _, h, w = gt_image.shape
        smooth_loss = kornia.losses.inverse_depth_smoothness_loss(surf_depth.unsqueeze(0), gt_image.unsqueeze(0))
        smooth_loss = lambda_smooth * smooth_loss
        tv_loss_depth = 0.
        tv_loss_normal = 0.
        tv_loss_depth += gaussians.TVLoss(surf_depth.permute(1, 2, 0))
        tv_loss_normal += gaussians.EdgeAwareTVLoss(rend_normal.permute(1, 2, 0), gt_image.permute(1, 2, 0))
        tv_loss = lambda_tv_depth * tv_loss_depth + lambda_tv_normal * tv_loss_normal


        if pipe.no_spike:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_loss) + ld_pdf * lv_pdf + ld_scale * lscale
        elif pipe.no_cut:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_loss) + ld_opa * lv_opa + ld_scale * lscale
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_loss) + ld_pdf * lv_pdf + ld_opa * lv_opa + ld_scale * lscale 
        loss += normal_loss + dist_loss + smooth_loss + tv_loss 
        if iteration % 2000 == 0 or iteration == 29999:  
            print("v_opa: {:.4f}".format(gaussians.Vth_opa.item()), 
                  "v_pdf: {:.4f}".format(gaussians.get_Vth_pdf.mean().item()), 
                  "num_gaussians:", gaussians._xyz.shape[0], 
                  "loss: {:.4f}".format(loss.item()), 
                  "norm: {:.4f}".format(normal_loss.item()),
                  "dist: {:.4f}".format(dist_loss.item()),
            )
        if iteration % 15000 == 0:
            mask = gaussians.get_opacity < 0.5
            print("mean_opac_low0.5:{:.4f}".format(gaussians.get_opacity[mask].mean().item()))
            mask = gaussians.get_opacity < 0.9
            print("mean_opac_low0.9:{:.4f}".format(gaussians.get_opacity[mask].mean().item()))
            mask = gaussians.get_opacity < 1.01
            print("mean_opac_low1.0:{:.4f}".format(gaussians.get_opacity[mask].mean().item()))
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, gaussians.Vth_opa.item(), scene.cameras_extent, size_threshold, radii, visibility_filter)
                if (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)) :
                    if pipe.no_spike:
                        gaussians.reset_opacity()
                    else:
                        gaussians.reset_opacity_spike()
            elif not pipe.no_spike:
                if iteration % opt.densification_interval == 0:
                    gaussians.spike_prune(gaussians.Vth_opa.item())
                if iteration == 29999:
                    gaussians.last_prune(gaussians.Vth_opa.item())
            
            # scale-based clone
            if iteration >= 15000 and iteration <= 25000 and iteration % 800 == 0 and not pipe.no_spike:
                gaussians.densify_and_prune_scale(opt.densify_grad_threshold, gaussians.Vth_opa.item(), scene.cameras_extent, size_threshold, radii, visibility_filter, scale_reg)

            # if iteration >= 25000 and iteration % 200 == 0 and not pipe.no_spike:
            #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #     gaussians.prune_big(scene.cameras_extent, size_threshold)
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10000, 20000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 10000, 20000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--scale', type=float, default=0.01)
    parser.add_argument('--lambda_opa', type=float, default=0.00002)
    parser.add_argument('--mask_normal', type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed(100)
    psnr = PeakSignalNoiseRatio().cuda()

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(pp.extract(args).exp_name)
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.scale, args.lambda_opa, args.mask_normal)

    # All done
    print("\nTraining complete.")

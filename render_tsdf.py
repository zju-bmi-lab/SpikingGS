import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import multiprocessing as mp
from typing import List
from skimage.morphology import binary_dilation, disk
import trimesh
import sklearn.neighbors as skln
from scipy.io import loadmat
import json
import numpy as np
import open3d as o3d
import os

from scene import Scene
from gaussian_renderer import render, render_through, render_skip_filter
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import math
from os import makedirs
from tqdm import tqdm

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):

        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj

def keep_largest_connected_component(mesh):
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)

    return mesh

def keep_large_connected_component(mesh, min_triangles=200):
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    small_clusters = np.where(cluster_n_triangles < min_triangles)[0]
    triangles_to_remove = np.isin(triangle_clusters, small_clusters)
    mesh.remove_triangles_by_mask(triangles_to_remove)

    return mesh

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
    mesh_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mesh")
    depthnorm_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depthnorm")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    makedirs(mesh_path, exist_ok=True)
    makedirs(depthnorm_path, exist_ok=True)

    depthmaps = []
    alphamaps = []
    rgbmaps = []
    normals = []
    depth_normals = []
    points = []
    # print(gaussians.get_scaling[:, 1].max())

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx >= 0 and idx < 300:
            render_pkg = render_skip_filter(view, gaussians, pipeline, background)
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            depth = render_pkg['surf_depth']
            depth_normal = render_pkg['surf_normal']
            # depth_normal = torch.nn.functional.normalize(render_pkg['surf_normal'], dim=0)
            point = render_pkg['surf_point']
            rgbmaps.append(rgb.cpu())
            depthmaps.append(depth.cpu())
            alphamaps.append(alpha.cpu())
            normals.append(normal.cpu())
            depth_normals.append(depth_normal.cpu())
            points.append(point.cpu())
            norm = depth.max()
            depth = depth / norm

            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rgb, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(normal*0.5+0.5, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(depth_normal*0.5+0.5, os.path.join(depthnorm_path, '{0:05d}'.format(idx) + ".png"))
    
    rgbmaps = torch.stack(rgbmaps, dim=0)
    depthmaps = torch.stack(depthmaps, dim=0)
    alphamaps = torch.stack(alphamaps, dim=0)
    depth_normals = torch.stack(depth_normals, dim=0)
    points = torch.stack(points, dim=0)

    voxel_size=0.004 
    sdf_trunc=0.02 
    depth_trunc=4.4
    print(f'voxel_size: {voxel_size}')
    print(f'sdf_trunc: {sdf_trunc}')
    print(f'depth_truc: {depth_trunc}')
    
    volume = o3d.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.integration.TSDFVolumeColorType.RGB8
        )

    for i, cam_o3d in tqdm(enumerate(to_cam_open3d(views)), desc="TSDF integration progress"):
        if i < 300:
            rgb = rgbmaps[i]
            depth = depthmaps[i]
            # print(rgbmaps.shape[0])
            mask_backgrond = 1
            if mask_backgrond and (views[i].image_mask is not None):
                adjusted_mask = views[i].image_mask[0, :, :].unsqueeze(0)
                depth[(adjusted_mask < 0.5)] = 0
            
            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

    mesh = volume.extract_triangle_mesh()
    mesh_name = 'fused_full.ply'
    o3d.io.write_triangle_mesh(os.path.join(mesh_path, mesh_name), mesh)
    
    largest_component_mesh = keep_largest_connected_component(mesh)
    mesh_name = 'fused.ply'
    o3d.io.write_triangle_mesh(os.path.join(mesh_path, mesh_name), largest_component_mesh)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        skip_test = True
        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)


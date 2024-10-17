import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
from skimage.morphology import binary_dilation, disk
import argparse
import json
import trimesh
from pathlib import Path
import subprocess
from os import makedirs, path

import sys
import render_utils as rend_util
from tqdm import tqdm

def load_dtu_camera(DTU):
    # Load projection matrix from file.
    camtoworlds = []
    for i in range(1, 64+1):
        fname = path.join(DTU, f'Calibration/cal18/pos_{i:03d}.txt')

        projection = np.loadtxt(fname, dtype=np.float32)

        # Decompose projection matrix into pose and camera matrix.
        camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[:3]
        camera_mat = camera_mat / camera_mat[2, 2]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_mat.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]
        pose = pose[:3]
        camtoworlds.append(pose)
    return camtoworlds

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def cull_scan(scan, mesh_path, result_mesh_file, instance_dir, dtu_Dataset_path):
    
    # load poses
    image_dir = '{0}/images'.format(instance_dir)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    n_images = len(image_paths)
    cam_file = '{0}/cameras.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())
    
    # load mask
    mask_dir = '{0}/mask'.format(instance_dir)
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    masks = []
    for p in mask_paths:
        mask = cv2.imread(p)
        masks.append(mask)

    # hard-coded image shape
    W, H = 1600, 1200

    # load mesh
    mesh = trimesh.load(mesh_path)
    # with open(Path(mask_dir) / '../../../../dtu_eval' / 'sfm' / f'sfm_scene_{str(scan)}.json') as f:
    #     sfm_scene = json.load(f)
    # bbox_transform = np.array(sfm_scene['bbox']['transform']).reshape(4, 4)
    # bbox_transform = bbox_transform.copy()
    # bbox_transform[[0, 1, 2], [0, 1, 2]] = bbox_transform[[0, 1, 2], [0, 1, 2]].max() / 2
    # # load transformation matrix

    # scale_m = bbox_transform  # [4, 4]
    # mesh.vertices = mesh.vertices * scale_m[0, 0] + scale_m[:3, 3][None]

    # mesh.export(result_mesh_file)
    # print(t)


    vertices = mesh.vertices
    # inv_scale_mat = np.linalg.inv(scale_mats[0])
    # vertices = vertices @ inv_scale_mat[:3, :3].T + inv_scale_mat[:3, 3][None]

    # project and filter
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()

    sampled_masks = []
    for i in tqdm(range(n_images),  desc="Culling mesh given masks"):
        pose = pose_all[i]
        w2c = torch.inverse(pose).cuda()
        intrinsic = intrinsics_all[i].cuda()

        with torch.no_grad():
            # transform and project
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1. ) & (pix_coords < 1.)).all(dim=-1).float()
            
            # dialate mask similar to unisurf
            maski = masks[i][:, :, 0].astype(np.float32) / 256.
            maski = torch.from_numpy(binary_dilation(maski, disk(15))).float()[None, None].cuda()

            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]

            sampled_mask = sampled_mask + (1. - valid)
            sampled_masks.append(sampled_mask)

    sampled_masks = torch.stack(sampled_masks, -1)
    # filter
    
    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)

    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)

    scale_mat = scale_mats[0]
    mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
    mesh.export(result_mesh_file)

    # transform vertices to world 
    # mesh.vertices = mesh.vertices @ inv_scale_mat[:3, :3].T + inv_scale_mat[:3, 3][None]

    # inv_scale_mat2 = np.linalg.inv(scale_m)
    # mesh.vertices = mesh.vertices @ inv_scale_mat2[:3, :3].T + inv_scale_mat2[:3, 3][None]
    directory, base_filename = os.path.split(result_mesh_file)
    new_base_filename = "cull_small.ply"
    new_result_mesh_file = os.path.join(directory, new_base_filename)
    mesh.export(new_result_mesh_file)
    del mesh
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the mesh.'
    )

    parser.add_argument('--input_mesh', type=str,  help='path to the mesh to be evaluated')
    parser.add_argument('--scan_id', type=str,  help='scan id of the input mesh')
    parser.add_argument('--output_dir', type=str, default='cull_mesh/0/culled_mesh.ply', help='path to the output folder')
    parser.add_argument('--mask_dir', type=str,  default='mask', help='path to uncropped mask')
    parser.add_argument('--DTU', type=str,  default='Offical_DTU_Dataset', help='path to the GT DTU point clouds')
    args = parser.parse_args()

    Offical_DTU_Dataset = args.DTU
    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    scan = args.scan_id
    ply_file = args.input_mesh
    print("cull mesh ....")
    result_mesh_file = os.path.join(out_dir, "culled_mesh.ply")
    print(result_mesh_file)
    cull_scan(scan, ply_file, result_mesh_file, instance_dir=os.path.join(args.mask_dir, f'scan{args.scan_id}'), dtu_Dataset_path=args.DTU)

    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # cmd = f"python {script_dir}/eval.py --data {result_mesh_file} --scan {scan} --mode mesh --dataset_dir {Offical_DTU_Dataset} --vis_out_dir {out_dir}"
    # os.system(cmd)
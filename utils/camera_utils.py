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

from scene.cameras import Camera
import numpy as np
import torch
import torchvision
from torchvision.transforms.functional import InterpolationMode
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    if hasattr(cam_info.image, 'shape'):
        orig_h, orig_w = cam_info.image.shape[:2]
    else:
        orig_h, orig_w = cam_info.image.size
    
    if args.resolution in [1, 2, 4, 8]:
        scale = resolution_scale * args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = global_down * resolution_scale
    resolution = (int(orig_h / scale), int(orig_w / scale))
    
    if isinstance(cam_info.image, np.ndarray):
        image = torch.from_numpy(cam_info.image).float().permute(2, 0, 1)
        if scale == 1:
            resized_image_rgb = image
        else:
            resized_image_rgb = torchvision.transforms.Resize(resolution, antialias=True)(image)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        gt_image = resized_image_rgb[:3, ...]

    if len(cam_info.image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        resized_image_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb #* resized_image_mask
    else:
        resized_image_mask = None

    # change the fx and fy
    scale_cx = cam_info.cx
    scale_cy = cam_info.cy
    scale_fx = cam_info.fx
    scale_fy = cam_info.fy
    if cam_info.cx is not None and cam_info.cy is not None:
        scale_cx /= scale
        scale_cy /= scale
        scale_fx /= scale
        scale_fy /= scale

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, fx=scale_fx, fy=scale_fy, cx=scale_cx, cy=scale_cy,
                  image=gt_image, image_mask=resized_image_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]

    if camera.cx is None:
        camera_entry = {
            'id': id,
            'img_name': camera.image_name,
            'width': camera.width,
            'height': camera.height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'FoVx': camera.FovX,
            'FoVy': camera.FovY,
        }
    else:
        camera_entry = {
            'id': id,
            'img_name': camera.image_name,
            'width': camera.width,
            'height': camera.height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'fx': camera.fx,
            'fy': camera.fy,
            'cx': camera.cx,
            'cy': camera.cy,
        }
    return camera_entry
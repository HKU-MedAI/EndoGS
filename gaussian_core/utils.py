import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torchvision

import imageio
import struct

from plyfile import PlyData, PlyElement
from time import time
from tqdm import tqdm

from gaussian_renderer import render
from utils.image_utils import psnr, img_tv_loss
from utils.graphics_utils import BasicPointCloud
from gaussian_core.cameras import CamerasWrapper
from gaussian_core.cameras import convert_gs_to_pytorch3d
from pytorch3d.transforms import quaternion_apply, quaternion_invert

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def training(opt, dataloader, gaussians):

    spatial_lr_scale = 5

    data = next(iter(dataloader))
    ply_path = os.path.join(data['sparse_path'], "points3D.ply")
    bin_path = os.path.join(data['sparse_path'], "points3D.bin")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")

        xyz, rgb, _ = read_points3D_binary(bin_path)

        storePly(ply_path, xyz, rgb)
    
    pcd = fetchPly(ply_path)
    gaussians.create_from_pcd(pcd, spatial_lr_scale)

    gaussians.training_setup()

    if opt.coarse_iters > 0:
        recon(opt, dataloader, gaussians, "coarse", opt.coarse_iters)
    if opt.fine_iters > 0:
        recon(opt, dataloader, gaussians, "fine", opt.fine_iters)


def recon(opt, dataloader, gaussians, stage, num_iter):

    first_iter = 0
    
    white_background = 1
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    densify_from_iter = 500
    densify_until_iter = 45000
    densification_interval = 100
    densify_grad_threshold_coarse = 0.0002
    densify_grad_threshold_fine_init = 0.0002
    densify_grad_threshold_after = 0.0002

    opacity_reset_interval = 6000
    opacity_threshold_coarse = 0.005
    opacity_threshold_fine_init = 0.005
    opacity_threshold_fine_after = 0.005

    pruning_from_iter = 500
    pruning_interval = 7000

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    start_entropy_regular = 4000
    end_entropy_regular = 7000
    entropy_regularization_factor = 0.1

    close_gaussian_threshold = 2
    regularity_knn = 16
    reset_neighbors_every = 500
    regular_from = 4000
    density_factor = 1. / 16.
    density_threshold = 1.
    sdf_estimation_factor = 0.2
    sdf_normal_factor = 0.2
    sample_number = 4000 

    final_iter = num_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    iteration = first_iter
    while True:
        for data in dataloader:
        
            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            viewpoint_cams = [data['camera']]
            wrap_gs_cams = CamerasWrapper(viewpoint_cams)

            images = []
            gt_images = []
            pred_depth = []
            radii_list = []
            visibility_filter_list = []
            viewspace_point_tensor_list = []
            entropy_opacities_loss_list = []
            sdf_loss_list = []
            normal_loss_list = []
            
            for viewpoint_cam in viewpoint_cams:
                gs_cam = viewpoint_cam
                fov_camera = convert_gs_to_pytorch3d([gs_cam])
                
                render_pkg = render(gs_cam, gaussians, data['time'], background, stage=stage)
                image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
                opacities = render_pkg["opacities"]
                images.append(image.unsqueeze(0))
                depth = depth / (depth.max() + 1e-5)
                pred_depth.append(depth.unsqueeze(0))
                gt_image = viewpoint_cam.original_image.cuda()

                gt_images.append(gt_image.unsqueeze(0))
                radii_list.append(radii.unsqueeze(0))
                visibility_filter_list.append(visibility_filter.unsqueeze(0))
                viewspace_point_tensor_list.append(viewspace_point_tensor)

                if iteration > start_entropy_regular and iteration < end_entropy_regular:
                    if visibility_filter is not None:
                        vis_opacities = opacities[visibility_filter]
                    else:
                        vis_opacities = opacities

                    entropy_opacities_loss_list.append(entropy_regularization_factor * (
                        - vis_opacities * torch.log(vis_opacities + 1e-10)
                        - (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)
                        ).mean())

                if iteration > regular_from:
                    if iteration == regular_from + 1 or iteration % reset_neighbors_every == 0:
                        gaussians.reset_neighbors(regularity_knn)
                        
                    sampling_mask = visibility_filter
                    
                    with torch.no_grad():
                        gaussian_to_camera = torch.nn.functional.normalize(fov_camera.get_camera_center() - gaussians.get_xyz, dim=-1)
                        gaussian_centers_in_camera_space = fov_camera.get_world_to_view_transform().transform_points(gaussians.get_xyz)

                        gaussian_centers_z = gaussian_centers_in_camera_space[..., 2] + 0.
                        gaussian_centers_map_z = gaussians.get_points_depth_in_depth_map(fov_camera, depth, gaussian_centers_in_camera_space, gs_cam.image_height, gs_cam.image_width)
                        
                        gaussian_standard_deviations = (
                            gaussians.get_scaling * quaternion_apply(quaternion_invert(gaussians.get_rotation), gaussian_to_camera)
                            ).norm(dim=-1)
                    
                        gaussians_close_to_surface = (gaussian_centers_map_z - gaussian_centers_z).abs() < close_gaussian_threshold * gaussian_standard_deviations
                        sampling_mask = sampling_mask * gaussians_close_to_surface

                    n_gaussians_in_sampling = sampling_mask.sum()

                    if n_gaussians_in_sampling > 0:
                        sdf_samples, sdf_gaussian_idx = gaussians.sample_points_in_gaussians(
                            num_samples=sample_number, 
                            sampling_scale_factor=1.5,
                            mask=sampling_mask
                            )
                            
                        fields = gaussians.get_field_values(
                            sdf_samples, sdf_gaussian_idx, 
                            return_sdf=True, 
                            density_threshold=density_threshold, density_factor=density_factor, 
                            return_sdf_grad=False, sdf_grad_max_value=10.,
                            return_closest_gaussian_opacities=True,
                            return_beta=True,
                            opacity=opacities
                            )
                        
                        # Compute the depth of the points in the gaussians
                        sdf_samples_in_camera_space = fov_camera.get_world_to_view_transform().transform_points(sdf_samples)
                        sdf_samples_z = sdf_samples_in_camera_space[..., 2] + 0.
                        proj_mask = sdf_samples_z > fov_camera.znear
                        sdf_samples_map_z = gaussians.get_points_depth_in_depth_map(fov_camera, depth, sdf_samples_in_camera_space[proj_mask], gs_cam.image_height, gs_cam.image_width)
                        sdf_estimation = sdf_samples_map_z - sdf_samples_z[proj_mask]
                        
                        with torch.no_grad():
                            sdf_sample_std = gaussian_standard_deviations[sdf_gaussian_idx][proj_mask]

                        # sdf loss
                        sdf_values = fields['sdf'][proj_mask]
                        sdf_estimation_loss = (sdf_values - sdf_estimation.abs()).abs() / sdf_sample_std
                        sdf_loss = sdf_estimation_factor * sdf_estimation_loss.clamp(max=10.*gaussians.get_cameras_spatial_extent(wrap_gs_cams)).mean()

                        # normal loss
                        closest_gaussians_idx = gaussians.knn_idx[sdf_gaussian_idx]
                        # Compute minimum scaling
                        closest_min_scaling = gaussians.get_scaling.min(dim=-1)[0][closest_gaussians_idx].detach().view(len(sdf_samples), -1)
                        
                        # Compute normals and flip their sign if needed
                        closest_gaussian_normals = gaussians.get_normals()[closest_gaussians_idx]
                        samples_gaussian_normals = gaussians.get_normals()[sdf_gaussian_idx]
                        closest_gaussian_normals = closest_gaussian_normals * torch.sign(
                            (closest_gaussian_normals * samples_gaussian_normals[:, None]).sum(dim=-1, keepdim=True)
                            ).detach()
                        
                        # Compute weights for normal regularization, based on the gradient of the sdf
                        closest_gaussian_opacities = fields['closest_gaussian_opacities'].detach()
                        normal_weights = ((sdf_samples[:, None] - gaussians.get_xyz[closest_gaussians_idx]) * closest_gaussian_normals).sum(dim=-1).abs()
                        
                        # sdf_better_normal_gradient_through_normal_only
                        normal_weights = normal_weights.detach()
                        normal_weights =  closest_gaussian_opacities * normal_weights / closest_min_scaling.clamp(min=1e-6)**2
                        
                        # The weights should have a sum of 1 because of the eikonal constraint
                        normal_weights_sum = normal_weights.sum(dim=-1).detach()
                        normal_weights = normal_weights / normal_weights_sum.unsqueeze(-1).clamp(min=1e-6)
                        
                        # Compute regularization loss
                        sdf_better_normal_loss = (samples_gaussian_normals - (normal_weights[..., None] * closest_gaussian_normals).sum(dim=-2)
                                                    ).pow(2).sum(dim=-1)

                        normal_loss = sdf_normal_factor * sdf_better_normal_loss.mean()

                        sdf_loss_list.append(sdf_loss)
                        normal_loss_list.append(normal_loss)

            radii = torch.cat(radii_list,0).max(dim=0).values
            visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
            image_tensor = torch.cat(images,0)
            pred_depth_tensor = torch.cat(pred_depth,0)
            gt_image_tensor = torch.cat(gt_images,0)

            mask = data['mask'].unsqueeze(0).unsqueeze(0).to(image_tensor.device)
            weight = data['spatialweight'].unsqueeze(0).unsqueeze(0).to(image_tensor.device)
            gt_depth = (data['depth']/(data['depth'].max()+1e-5)).unsqueeze(0).unsqueeze(0).to(image_tensor.device)

            # Loss
            Ll1 = (torch.abs((image_tensor*mask - gt_image_tensor*mask))*weight).mean()
            psnr_ = psnr(image_tensor*mask, gt_image_tensor*mask).mean().double()

            depth_loss = F.huber_loss(pred_depth_tensor*mask, gt_depth*mask, delta=0.2)
            img_tvloss = img_tv_loss(image_tensor*(1-mask))

            loss = Ll1 + 0.5*depth_loss + 0.01*img_tvloss
            
            if iteration > start_entropy_regular and iteration < end_entropy_regular:
                opacities_loss_tensor = torch.tensor(entropy_opacities_loss_list).float()
                loss += torch.mean(opacities_loss_tensor)

            if iteration > regular_from and n_gaussians_in_sampling > 0:
                sdf_loss_tensor = torch.tensor(sdf_loss_list).float()
                normal_loss_tensor = torch.tensor(normal_loss_list).float()
                loss += torch.mean(sdf_loss_tensor) + torch.mean(normal_loss_tensor)

            if stage == "fine":
                tv_loss = gaussians.compute_regulation(1e-3, 2e-2, 1e-3)
                loss += tv_loss

            loss.backward()
            viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
            for idx in range(0, len(viewspace_point_tensor_list)):
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
                total_point = gaussians._xyz.shape[0]

                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                            "psnr": f"{psnr_:.{2}f}",
                                            "point":f"{total_point}"
                                            })
                    progress_bar.update(10)

                if iteration > final_iter and stage == 'fine':
                    progress_bar.close()

                # Log and save
                if iteration % 20000 == 0 or iteration == final_iter:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))

                    gaussians.save(opt.workspace, iteration, stage)
                
                # Densification
                if iteration < densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                    if stage == "coarse":
                        opacity_threshold = opacity_threshold_coarse
                        densify_threshold = densify_grad_threshold_coarse
                    else:    
                        opacity_threshold = opacity_threshold_fine_init - iteration*(opacity_threshold_fine_init - opacity_threshold_fine_after)/(densify_until_iter)  
                        densify_threshold = densify_grad_threshold_fine_init - iteration*(densify_grad_threshold_fine_init - densify_grad_threshold_after)/(densify_until_iter)  

                    if iteration > densify_from_iter and iteration % densification_interval == 0 :
                        size_threshold = 20 if iteration > opacity_reset_interval else None

                        # because we use the same camera infos for the video frames,
                        # there is no need to compute the radius from getNerfppNorm to get the extent
                        # if use that, an error would occur in pruning due to all1s mask
                        # here free to treat the radius and the spatial lr scale as hyperparamters
                        gaussians.densify(densify_threshold, opacity_threshold, 5, size_threshold)

                        if iteration > regular_from:
                            gaussians.reset_neighbors()
                    
                    if iteration > pruning_from_iter and iteration % pruning_interval == 0:
                        size_threshold = 20 if iteration > opacity_reset_interval else None
                        gaussians.prune(densify_threshold, opacity_threshold, 5, size_threshold)

                        if iteration > regular_from:
                            gaussians.reset_neighbors()
                        
                    if iteration % opacity_reset_interval == 0 or (white_background and iteration == densify_from_iter):
                        print("reset opacity")
                        gaussians.reset_opacity()

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                

            iteration += 1
            if iteration > final_iter:
                break
            
        if iteration > final_iter:
            break


def testing(opt, dataloader, gaussians, save_gt=True):

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians.load_ply(os.path.join(opt.model_path, "point_cloud.ply"))
    gaussians.load_model(os.path.join(opt.model_path))

    render_path = os.path.join(opt.model_path, "render")
    os.makedirs(render_path, exist_ok=True)
    render_images = []
    render_list = []
    
    if save_gt:
        gts_path = os.path.join(opt.model_path, "gt")
        os.makedirs(gts_path, exist_ok=True)
        gt_images = []
        gt_list = []

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader, desc="Rendering progress")):
            if idx == 0:time1 = time()

            viewpoint_cams = [data['camera']]
            for viewpoint_cam in viewpoint_cams:
                rendering = render(data['camera'], gaussians, data['time'], background)["render"]
            
                render_images.append(to8b(rendering).transpose(1,2,0))
                render_list.append(rendering)

                if save_gt:
                    gt = viewpoint_cam.original_image[0:3, :, :]
                    gt_images.append(to8b(gt).transpose(1,2,0))
                    gt_list.append(gt)

    time2=time()
    print("FPS:",(len(dataloader)-1)/(time2-time1))

    count = 0
    print("writing rendering images.")
    if len(render_list) != 0:
        for image in tqdm(render_list):
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            count +=1

    imageio.mimwrite(os.path.join(opt.model_path, 'video_render.mp4'), render_images, fps=10, quality=8)

    if save_gt:
        count = 0
        print("writing training images.")
        if len(gt_list) != 0:
            for image in tqdm(gt_list):
                torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
                count+=1
        imageio.mimwrite(os.path.join(opt.model_path, 'video_gt.mp4'), gt_images, fps=10, quality=8)

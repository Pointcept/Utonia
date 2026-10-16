# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This demo code is modified from VGGT huggingface demo(https://huggingface.co/spaces/facebook/vggt/blob/main/app.py)
import argparse
import os
import cv2
import torch
import shutil
from datetime import datetime
import glob
import gc
import numpy as np
import open3d as o3d
import utonia
from scipy.spatial.transform import Rotation as R
import trimesh
import time
from typing import List, Tuple
from pathlib import Path
from natsort import natsorted
from safetensors.torch import load_file
from einops import rearrange
from tqdm import tqdm
import camtools as ct
from PIL import Image
from torchvision import transforms as TF

try:
    import flash_attn
except ImportError:
    flash_attn = None
device = "cuda" if torch.cuda.is_available() else "cpu"


from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # if not torch.cuda.is_available():
    #     raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = (
                predictions[key].cpu().numpy().squeeze(0)
            )  # remove batch dimension

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"]
    )
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions


def handle_uploads(input_video, conf_thres, frame_interval, prediction_mode, if_TSDF):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(os.path.dirname(script_path))
    target_dir = os.path.join(script_dir, "video_demo_output")
    target_dir_images = os.path.join(target_dir, "images")
    target_dir_pcds = os.path.join(target_dir, "pcds")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)
    os.makedirs(target_dir_pcds)
    # --- Handle video ---
    if input_video is not None:
        print("processing video")
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * frame_interval)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        image_paths = []
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(
                    target_dir_images, f"{video_frame_num:06}.png"
                )
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1
        # Sort final images for gallery
        image_paths = sorted(image_paths)
        original_points, original_colors, original_normals = parse_frames(
            target_dir, conf_thres, prediction_mode, if_TSDF
        )
    scene_3d = trimesh.Scene()
    point_cloud_data = trimesh.PointCloud(
        vertices=original_points,
        colors=original_colors,
        vertex_normals=original_normals,
    )
    scene_3d.add_geometry(point_cloud_data)
    np.save(os.path.join(target_dir_pcds, f"points.npy"), original_points)
    np.save(os.path.join(target_dir_pcds, f"colors.npy"), original_colors)
    np.save(os.path.join(target_dir_pcds, f"normals.npy"), original_normals)
    end_time = time.time()
    print(f"Files copied to {target_dir}; took {end_time - start_time:.3f} seconds")
    return (
        target_dir,
        image_paths,
        original_points,
        original_colors,
        original_normals,
        end_time - start_time,
    )


def parse_frames(
    target_dir,
    conf_thres=3.0,
    prediction_mode="Pointmap Regression",
    if_TSDF=True,
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    target_dir_pcds = os.path.join(target_dir, "pcds")
    all_files = (
        sorted(os.listdir(target_dir_images))
        if os.path.isdir(target_dir_images)
        else []
    )
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, VGGT_model)

    # Convert pose encoding to extrinsic and intrinsic matrices
    images = predictions["images"]
    Ts, Ks = predictions["extrinsic"], predictions["intrinsic"]
    Ts = ct.convert.pad_0001(Ts)
    Ts_inv = np.linalg.inv(Ts)
    Cs = np.array([ct.convert.T_to_C(T) for T in Ts])  # (n, 3)

    # [1, 8, 294, 518, 3]
    world_points = predictions["world_points"]

    # Compute view direction for each pixel
    # (b n h w c) - (n, 3)
    view_dirs = world_points - rearrange(Cs, "n c -> n 1 1 c")
    view_dirs = rearrange(view_dirs, "n h w c -> (n h w) c")
    view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=-1, keepdims=True)

    # Extract points and colors
    # [1, 8, 3, 294, 518]
    img_num = world_points.shape[1]
    images = predictions["images"]
    points = rearrange(world_points, "n h w c -> (n h w) c")
    colors = rearrange(images, "n c h w -> (n h w) c")

    if prediction_mode == "Pointmap Branch":
        world_points_conf = predictions["world_points_conf"]
        conf = world_points_conf.reshape(-1)
        if conf_thres == 0.0:
            conf_threshold = 0.0
        else:
            conf_threshold = np.percentile(conf, conf_thres)
        conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
        points = points[conf_mask]
        colors = colors[conf_mask]
        points, Ts_inv, _ = Coord2zup(points, Ts_inv)
        scale = 3 / (points[:, 2].max() - points[:, 2].min())
        points *= scale
        Ts_inv[:, :3, 3] *= scale

        # Create a point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.estimate_normals()
        # o3d.io.write_point_cloud("pcd.ply", pcd)
        try:
            pcd, inliers, rotation_matrix, offset = extract_and_align_ground_plane(pcd)
            T_pcd = np.eye(4)
            T_pcd[:3, :3] = rotation_matrix
            T_pcd[2, 3] = -offset
            Ts_inv = T_pcd @ Ts_inv
        except Exception as e:
            print(f"cannot find ground, err:{e}")
        # Filp normals such that normals always point to camera
        # Compute the dot product between the normal and the view direction
        # If the dot product is less than 0, flip the normal
        normals = np.asarray(pcd.normals)
        view_dirs = np.asarray(view_dirs)
        dot_product = np.sum(normals * view_dirs, axis=-1)
        flip_mask = dot_product > 0
        normals[flip_mask] = -normals[flip_mask]

        # Normalize normals a nd m
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
        pcd.normals = o3d.utility.Vector3dVector(normals)
    elif prediction_mode == "Depthmap and Camera Branch":
        # Integrate RGBD images into a TSDF volume and extract a mesh
        # (n, h, w, 3)
        im_colors = rearrange(images, "n c h w -> (n) h w c")
        # (b, n, h, w, 3)
        im_dists = world_points - rearrange(Cs, "n c -> n 1 1 c")
        im_dists = np.linalg.norm(im_dists, axis=-1, keepdims=False)

        # Convert distance to depth
        im_depths = []  # (n, h, w, c)
        for im_dist, K in zip(im_dists, Ks):
            im_depth = ct.convert.im_distance_to_im_depth(im_dist, K)
            im_depths.append(im_depth)
        im_depths = np.stack(im_depths, axis=0)
        if if_TSDF:
            mesh = integrate_rgbd_to_mesh(
                Ks=Ks,
                Ts=Ts,
                im_depths=im_depths,
                im_colors=im_colors,
                voxel_size=1 / 512,
            )
            rotation_angle = -np.pi / 2
            rotation_axis = np.array([1, 0, 0])  # X 轴
            mesh.rotate(
                o3d.geometry.get_rotation_matrix_from_axis_angle(
                    rotation_axis * rotation_angle
                ),
                center=(0, 0, 0),
            )
            vertices = np.asarray(mesh.vertices)
            scale_factor = 3.0 / (np.max(vertices[:, 2]) - np.min(vertices[:, 2]))
            mesh.scale(scale_factor, center=(0, 0, 0))
            vertices = np.asarray(mesh.vertices)
            colors = (
                np.asarray(mesh.vertex_colors)
                if mesh.has_vertex_colors()
                else np.zeros_like(vertices)
            )
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals)
            Ts_inv = rotx(Ts_inv, theta=-90)
            Ts_inv[:, :3, 3] *= scale_factor
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd.normals = o3d.utility.Vector3dVector(normals)
        else:
            points = []
            for K, T, im_depth in zip(Ks, Ts, im_depths):
                point = ct.project.im_depth_to_point_cloud(
                    im_depth=im_depth,
                    K=K,
                    T=T,
                    to_image=False,
                    ignore_invalid=False,
                )
                points.append(point)
            points = np.vstack(points)
            colors = im_colors.reshape(-1, 3)
            world_points_conf = predictions["depth_conf"]
            conf = world_points_conf.reshape(-1)
            if conf_thres == 0.0:
                conf_threshold = 0.0
            else:
                conf_threshold = np.percentile(conf, conf_thres)
            conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
            points = points[conf_mask]
            colors = colors[conf_mask]
            points, Ts_inv, _ = Coord2zup(points, Ts_inv)
            scale_factor = 3.0 / (np.max(points[:, 2]) - np.min(points[:, 2]))
            points *= scale_factor
            Ts_inv[:, :3, 3] *= scale_factor
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd.estimate_normals()
        try:
            pcd, inliers, rotation_matrix, offset = extract_and_align_ground_plane(pcd)
            T_pcd = np.eye(4)
            T_pcd[:3, :3] = rotation_matrix
            T_pcd[2, 3] = -offset
            Ts_inv = T_pcd @ Ts_inv
        except Exception as e:
            print(f"cannot find ground, err:{e}")
    original_points = np.asarray(pcd.points)
    original_colors = np.asarray(pcd.colors)
    original_normals = np.asarray(pcd.normals)

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
    return original_points, original_colors, original_normals


def preprocess_images(frames, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        frames (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(frames) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for img in frames:

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = (
                    round(height * (new_width / width) / 14) * 14
                )  # Make divisible by 14
            else:
                new_height = target_size
                new_width = (
                    round(width * (new_height / height) / 14) * 14
                )  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(frames) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def extract_and_align_ground_plane(
    pcd,
    height_percentile=20,
    ransac_distance_threshold=0.01,
    ransac_n=3,
    ransac_iterations=1000,
    max_angle_degree=40,
    max_trials=6,
):
    points = np.asarray(pcd.points)
    z_vals = points[:, 2]
    z_thresh = np.percentile(z_vals, height_percentile)
    low_indices = np.where(z_vals <= z_thresh)[0]

    remaining_indices = low_indices.copy()

    for trial in range(max_trials):
        if len(remaining_indices) < ransac_n:
            raise ValueError("Not enough points left to fit a plane.")

        low_pcd = pcd.select_by_index(remaining_indices)

        plane_model, inliers = low_pcd.segment_plane(
            distance_threshold=ransac_distance_threshold,
            ransac_n=ransac_n,
            num_iterations=ransac_iterations,
        )
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)

        # current_plane_pcd = pcd.select_by_index(remaining_indices[inliers])
        # o3d.io.write_point_cloud("plane.ply",current_plane_pcd)
        # exit()

        angle = np.arccos(np.clip(np.dot(normal, [0, 0, 1]), -1.0, 1.0)) * 180 / np.pi
        if angle <= max_angle_degree:
            inliers_global = remaining_indices[inliers]

            target = np.array([0, 0, 1])
            axis = np.cross(normal, target)
            axis_norm = np.linalg.norm(axis)

            if axis_norm < 1e-6:
                rotation_matrix = np.eye(3)
            else:
                axis /= axis_norm
                rot_angle = np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0))
                rotation = R.from_rotvec(axis * rot_angle)
                rotation_matrix = rotation.as_matrix()

            rotated_points = points @ rotation_matrix.T
            ground_points_z = rotated_points[inliers_global, 2]
            offset = np.mean(ground_points_z)
            rotated_points[:, 2] -= offset

            aligned_pcd = o3d.geometry.PointCloud()
            aligned_pcd.points = o3d.utility.Vector3dVector(rotated_points)
            if pcd.has_colors():
                aligned_pcd.colors = pcd.colors
            if pcd.has_normals():
                rotated_normals = np.asarray(pcd.normals) @ rotation_matrix.T
                aligned_pcd.normals = o3d.utility.Vector3dVector(rotated_normals)

            return aligned_pcd, inliers_global, rotation_matrix, offset

        else:
            rejected_indices = remaining_indices[inliers]
            remaining_indices = np.setdiff1d(remaining_indices, rejected_indices)

    raise ValueError("Failed to find a valid ground plane within max trials.")


def rotx(x, theta=90):
    """
    Rotate x by theta degrees around the x-axis
    """
    theta = np.deg2rad(theta)
    rot_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )
    return rot_matrix @ x


def Coord2zup(points, extrinsics, normals=None):
    """
    Convert the dust3r coordinate system to the z-up coordinate system
    """
    points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1).T
    points = rotx(points, -90)[:3].T
    if normals is not None:
        normals = np.concatenate([normals, np.ones([normals.shape[0], 1])], axis=1).T
        normals = rotx(normals, -90)[:3].T
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    t = np.min(points, axis=0)
    points -= t
    extrinsics = rotx(extrinsics, -90)
    extrinsics[:, :3, 3] -= t.T
    return points, extrinsics, normals


def integrate_rgbd_to_mesh(
    Ks,
    Ts,
    im_depths,
    im_colors,
    voxel_size,
    bbox=None,
):
    """
    Integrate RGBD images into a TSDF volume and extract a mesh.

    Args:
        Ks: (N, 3, 3) camera intrinsics.
        Ts: (N, 4, 4) camera extrinsics.
        im_depths: (N, H, W) depth images, already in world scale.
        im_colors: (N, H, W, 3) color images, float range in [0, 1].
        voxel_size: TSDF voxel size, in meters, e.g. 3 / 512.
        bbox: Open3D axis-aligned bounding box, for cropping.

    Per Open3D convention, invalid depth values shall be set to 0.
    """
    num_images = len(Ks)
    if (
        len(Ts) != num_images
        or len(im_depths) != num_images
        or len(im_colors) != num_images
    ):
        raise ValueError("Ks, Ts, im_depths, im_colors must have the same length.")

    # Constants.
    trunc_voxel_multiplier = 8.0
    sdf_trunc = trunc_voxel_multiplier * voxel_size

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for K, T, im_depth, im_color in tqdm(
        zip(Ks, Ts, im_depths, im_colors),
        total=len(Ks),
        desc="Integrating RGBD frames",
    ):
        # Set invalid depth values to 0, based on bounding box.
        if bbox is not None:
            points = ct.project.im_depth_to_point_cloud(
                im_depth=im_depth,
                K=K,
                T=T,
                to_image=False,
                ignore_invalid=False,
            )
            assert len(points) == im_depth.shape[0] * im_depth.shape[1]
            point_indices_inside_bbox = bbox.get_point_indices_within_bounding_box(
                o3d.utility.Vector3dVector(points)
            )
            point_indices_outside_bbox = np.setdiff1d(
                np.arange(len(points)), point_indices_inside_bbox
            )
            im_depth.ravel()[point_indices_outside_bbox] = 0

        im_color_uint8 = np.ascontiguousarray((im_color * 255).astype(np.uint8))
        im_depth_uint16 = np.ascontiguousarray((im_depth * 1000).astype(np.uint16))
        im_color_o3d = o3d.geometry.Image(im_color_uint8)
        im_depth_o3d = o3d.geometry.Image(im_depth_uint16)
        im_rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im_color_o3d,
            im_depth_o3d,
            depth_scale=1000.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False,
        )
        o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=im_depth.shape[1],
            height=im_depth.shape[0],
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
        )
        o3d_extrinsic = T
        volume.integrate(
            im_rgbd_o3d,
            o3d_intrinsic,
            o3d_extrinsic,
        )

    mesh = volume.extract_triangle_mesh()
    return mesh


def get_pca_color(feat, start=0, brightness=1.25, center=True):
    u, s, v = torch.pca_lowrank(feat, center=center, q=3 * (start + 1), niter=5)
    projection = feat @ v
    projection = (
        projection[:, 3 * start : 3 * (start + 1)] * 0.6
        + projection[:, 3 * start : 3 * (start + 1)] * 0.4
    )
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Utonia pipeline on input video.")

    parser.add_argument(
        "--input_video", type=str, required=True, help="Path to input video"
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=10.0,
        help="Confidence threshold percentile (e.g., 5.0)",
    )
    parser.add_argument(
        "--frame_interval",
        type=float,
        default=1.0,
        help="Frame interval in seconds, 1 frame/ N sec",
    )
    parser.add_argument(
        "--prediction_mode",
        type=str,
        choices=["Pointmap Branch", "Depthmap and Camera Branch"],
        default="Depthmap and Camera Branch",
        help="Prediction mode",
    )
    parser.add_argument(
        "--if_TSDF",
        action="store_true",
        help="Whether to use TSDF integration (only for 'Depthmap and Camera Branch')",
    )
    parser.add_argument("--pca_start", type=int, default=1, help="pca start dimension")
    parser.add_argument(
        "--pca_brightness", type=float, default=1.2, help="pca brightness"
    )

    parser.add_argument(
        "--wo_color", dest="wo_color", action="store_true", help="disable the color."
    )
    parser.add_argument(
        "--wo_normal", dest="wo_normal", action="store_true", help="disable the normal."
    )

    parser.add_argument(
        "--model",
        choices=["utonia", "utonia_small", "utonia_tiny"],
        default="utonia",
        help="pre-trained Utonia encoder to use (default: utonia).",
    )
    args = parser.parse_args()

    VGGT_model = VGGT().to(device)
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    VGGT_model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    (
        target_dir,
        image_paths,
        original_points,
        original_colors,
        original_normals,
        frames_time,
    ) = handle_uploads(
        args.input_video,
        args.conf_thres,
        args.frame_interval,
        args.prediction_mode,
        args.if_TSDF,
    )

    # set random seed
    utonia.utils.set_seed(53124)

    if flash_attn is not None:
        model = utonia.load(args.model, repo_id="Pointcept/Utonia").to(device)
    else:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],  # reduce patch size if necessary
            enable_flash=False,
        )
        model = utonia.load(
            args.model, repo_id="Pointcept/Utonia", custom_config=custom_config
        ).to(device)
    model.eval()
    # The choice of scale depends on the desired level of granularity. You may need to fine-tune this parameter to find the optimal value for your specific tasks. Generally, the larger the scale, the more fine-grained the results will be.
    transform = utonia.transform.default(scale=1.0)

    coord = np.asarray(original_points)
    color = np.asarray(original_colors)
    normal = np.asarray(original_normals)

    point = {"coord": coord, "color": color, "normal": normal}

    if args.wo_color:
        point["color"] = np.zeros_like(point["coord"])
    if args.wo_normal:
        point["normal"] = np.zeros_like(point["coord"])

    original_coord = point["coord"].copy()
    original_color = point["color"].copy()
    point = transform(point)

    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor) and device == "cuda":
                point[key] = point[key].cuda(non_blocking=True)
        point = model(point)

        # upcast point feature
        # Point is a structure contains all the information during forward
        # Use range(4) to upcast features from all levels for quantitative evaluation
        for _ in range(2):
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        while "pooling_parent" in point:
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent

        pca_color = get_pca_color(
            point.feat,
            start=args.pca_start,
            brightness=args.pca_brightness,
            center=True,
        )

    original_pca_color = pca_color[point.inverse]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_coord)
    pcd.colors = o3d.utility.Vector3dVector(original_pca_color.cpu().detach().numpy())
    # o3d.visualization.draw_geometries([pcd])
    point_cloud = trimesh.PointCloud(vertices=original_coord, colors=original_color)
    point_cloud.export(os.path.join(target_dir, "pcd.ply"))

    point_cloud = trimesh.PointCloud(
        vertices=original_coord, colors=original_pca_color.cpu().numpy()
    )
    point_cloud.export(os.path.join(target_dir, "pca.ply"))

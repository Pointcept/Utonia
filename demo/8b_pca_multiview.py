# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Multi-view variant of 8_pca_video.py: lift a few view images (e.g. 4 cameras
# around a scene) to a point cloud with VGGT, then visualize Utonia PCA features.
import argparse
import gc
import os

import camtools as ct
import numpy as np
import open3d as o3d
import torch
import trimesh
import utonia
from einops import rearrange
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

try:
    import flash_attn
except ImportError:
    flash_attn = None
device = "cuda" if torch.cuda.is_available() else "cpu"


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

VGGT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def run_vggt(image_paths, model) -> dict:
    """Run VGGT on the given images and return numpy predictions."""
    images = load_and_preprocess_images(image_paths).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    print("Running VGGT inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = (
                predictions[key].cpu().numpy().squeeze(0)
            )  # remove batch dimension
    return predictions


def predictions_to_point_cloud(predictions, conf_thres, prediction_mode, if_TSDF):
    """Turn VGGT predictions into a z-up, ground-aligned point cloud."""
    images = predictions["images"]
    Ts, Ks = predictions["extrinsic"], predictions["intrinsic"]
    Ts = ct.convert.pad_0001(Ts)
    Ts_inv = np.linalg.inv(Ts)
    Cs = np.array([ct.convert.T_to_C(T) for T in Ts])  # (n, 3)

    world_points = predictions["world_points"]  # (n, h, w, 3)

    # Compute view direction for each pixel
    view_dirs = world_points - rearrange(Cs, "n c -> n 1 1 c")
    view_dirs = rearrange(view_dirs, "n h w c -> (n h w) c")
    view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=-1, keepdims=True)

    points = rearrange(world_points, "n h w c -> (n h w) c")
    colors = rearrange(images, "n c h w -> (n h w) c")

    if prediction_mode == "Pointmap Branch":
        conf = predictions["world_points_conf"].reshape(-1)
        if conf_thres == 0.0:
            conf_threshold = 0.0
        else:
            conf_threshold = np.percentile(conf, conf_thres)
        conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
        points = points[conf_mask]
        colors = colors[conf_mask]
        view_dirs = view_dirs[conf_mask]

        # Estimate normals and flip them towards the cameras while points and
        # view directions are still in the same (VGGT) frame.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)
        flip_mask = np.sum(normals * view_dirs, axis=-1) > 0
        normals[flip_mask] = -normals[flip_mask]
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

        points, _, normals = Coord2zup(points, Ts_inv, normals)
        points *= 3 / (points[:, 2].max() - points[:, 2].min())

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        try:
            pcd = extract_and_align_ground_plane(pcd)[0]
        except Exception as e:
            print(f"cannot find ground, err:{e}")
    elif prediction_mode == "Depthmap and Camera Branch":
        # (n, h, w, 3)
        im_colors = rearrange(images, "n c h w -> n h w c")
        im_dists = world_points - rearrange(Cs, "n c -> n 1 1 c")
        im_dists = np.linalg.norm(im_dists, axis=-1, keepdims=False)

        # Convert distance to depth
        im_depths = []  # (n, h, w)
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
            mesh.rotate(
                o3d.geometry.get_rotation_matrix_from_axis_angle(
                    np.array([1, 0, 0]) * -np.pi / 2
                ),
                center=(0, 0, 0),
            )
            vertices = np.asarray(mesh.vertices)
            if len(vertices) == 0:
                raise RuntimeError(
                    "TSDF fusion produced an empty mesh; try without --if_TSDF."
                )
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
            conf = predictions["depth_conf"].reshape(-1)
            if conf_thres == 0.0:
                conf_threshold = 0.0
            else:
                conf_threshold = np.percentile(conf, conf_thres)
            conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
            points = points[conf_mask]
            colors = colors[conf_mask]
            points, _, _ = Coord2zup(points, Ts_inv)
            points *= 3.0 / (np.max(points[:, 2]) - np.min(points[:, 2]))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd.estimate_normals()
        try:
            pcd = extract_and_align_ground_plane(pcd)[0]
        except Exception as e:
            print(f"cannot find ground, err:{e}")
    else:
        raise ValueError(f"Unknown prediction mode: {prediction_mode}")

    return np.asarray(pcd.points), np.asarray(pcd.colors), np.asarray(pcd.normals)


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


def integrate_rgbd_to_mesh(Ks, Ts, im_depths, im_colors, voxel_size):
    """
    Integrate RGBD images into a TSDF volume and extract a mesh.

    Args:
        Ks: (N, 3, 3) camera intrinsics.
        Ts: (N, 4, 4) camera extrinsics.
        im_depths: (N, H, W) depth images, already in world scale.
        im_colors: (N, H, W, 3) color images, float range in [0, 1].
        voxel_size: TSDF voxel size, in meters, e.g. 3 / 512.
    """
    trunc_voxel_multiplier = 8.0
    depth_scale = 1000.0
    depth_max = 10.0

    # Tensor-API TSDF: the legacy ScalableTSDFVolume yields empty meshes in
    # Open3D 0.20.0 (https://github.com/isl-org/Open3D/issues/7592).
    # block_count is only the initial capacity; the grid grows as needed.
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=(1, 1, 3),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=10000,
        device=o3d.core.Device("CPU:0"),
    )

    for K, T, im_depth, im_color in tqdm(
        zip(Ks, Ts, im_depths, im_colors),
        total=len(Ks),
        desc="Integrating RGBD frames",
    ):
        im_color_uint8 = np.ascontiguousarray((im_color * 255).astype(np.uint8))
        im_depth_uint16 = np.ascontiguousarray((im_depth * 1000).astype(np.uint16))
        im_color_o3d = o3d.t.geometry.Image(im_color_uint8)
        im_depth_o3d = o3d.t.geometry.Image(im_depth_uint16)
        o3d_intrinsic = o3d.core.Tensor(K, dtype=o3d.core.float64)
        o3d_extrinsic = o3d.core.Tensor(T, dtype=o3d.core.float64)
        block_coords = volume.compute_unique_block_coordinates(
            im_depth_o3d,
            o3d_intrinsic,
            o3d_extrinsic,
            depth_scale,
            depth_max,
            trunc_voxel_multiplier,
        )
        volume.integrate(
            block_coords,
            im_depth_o3d,
            im_color_o3d,
            o3d_intrinsic,
            o3d_extrinsic,
            depth_scale,
            depth_max,
            trunc_voxel_multiplier,
        )

    # weight_threshold=0 keeps every observed voxel, like the legacy volume.
    return volume.extract_triangle_mesh(weight_threshold=0.0).to_legacy()


def get_pca_color(feat, start=0, brightness=1.25, center=True):
    u, s, v = torch.pca_lowrank(feat, center=center, q=3 * (start + 1), niter=5)
    projection = feat @ v
    projection = projection[:, 3 * start : 3 * (start + 1)]
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Utonia pipeline on a few multi-view images."
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="View images, e.g. --images views/*.png. The first image is the reference view.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write pcd.ply and pca.ply (default: <repo>/multiview_demo_output).",
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=50.0,
        help="Confidence percentile threshold (VGGT demo default 50); 0 keeps all points. Not used with --if_TSDF.",
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

    for path in args.images:
        if not os.path.isfile(path):
            parser.error(f"image not found: {path}")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "multiview_demo_output",
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Using {len(args.images)} views: {args.images}")

    # Lift the views to a point cloud with VGGT, then free its GPU memory.
    VGGT_model = VGGT().to(device)
    VGGT_model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_URL))
    VGGT_model.eval()
    predictions = run_vggt(args.images, VGGT_model)
    del VGGT_model
    gc.collect()
    torch.cuda.empty_cache()

    coord, color, normal = predictions_to_point_cloud(
        predictions, args.conf_thres, args.prediction_mode, args.if_TSDF
    )
    print(f"Point cloud: {len(coord)} points")

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

    # VGGT colors are in [0, 1]; the transform's NormalizeColor expects [0, 255].
    point = {"coord": coord, "color": color * 255.0, "normal": normal}

    if args.wo_color:
        point["color"] = np.zeros_like(point["coord"])
    if args.wo_normal:
        point["normal"] = np.zeros_like(point["coord"])

    original_coord = coord.copy()
    original_color = color.copy()
    point = transform(point)

    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor) and device == "cuda":
                point[key] = point[key].cuda(non_blocking=True)
        point = model(point)

        # upcast point feature
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

    original_pca_color = pca_color[point.inverse].cpu().numpy()
    trimesh.PointCloud(vertices=original_coord, colors=original_color).export(
        os.path.join(output_dir, "pcd.ply")
    )
    trimesh.PointCloud(vertices=original_coord, colors=original_pca_color).export(
        os.path.join(output_dir, "pca.ply")
    )
    print(f"Saved pcd.ply (input colors) and pca.ply (Utonia PCA) to {output_dir}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(original_coord)
    pcd.colors = o3d.utility.Vector3dVector(original_pca_color)
    o3d.visualization.draw_geometries([pcd])

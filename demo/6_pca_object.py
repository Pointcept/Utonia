# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import utonia
import torch
import open3d as o3d
import numpy as np
import argparse

try:
    import flash_attn
except ImportError:
    flash_attn = None

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_pca_color(feat, brightness=1.25, center=True):
    u, s, v = torch.pca_lowrank(feat, center=center, niter=5, q=9)
    projection = feat @ v
    projection = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    utonia.utils.set_seed(73)

    # Load Model using utonia API
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

    # Load data
    point = utonia.data.load("sample3_object")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point["coord"])
    pcd.colors = o3d.utility.Vector3dVector(point["color"])
    pcd.estimate_normals()
    point["normal"] = np.asarray(pcd.normals)

    point_rotated = {}
    point_rotated["coord"] = point["coord"][:, [0, 2, 1]]  # Specific shuffle
    point_rotated["color"] = point["color"]

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(point_rotated["coord"])
    pcd2.colors = o3d.utility.Vector3dVector(point_rotated["color"])
    pcd2.estimate_normals()
    point_rotated["normal"] = np.asarray(pcd.normals)

    if args.wo_color:
        point["color"] = np.zeros_like(point["coord"])
        point_rotated["color"] = np.zeros_like(point_rotated["coord"])
    if args.wo_normal:
        point["normal"] = np.zeros_like(point["coord"])
        point_rotated["normal"] = np.zeros_like(point_rotated["coord"])

    bias = np.array([0, 0, 1])
    point_rotated["coord"] = point_rotated["coord"] + bias  # Apply bias for positioning

    transform = utonia.transform.default()

    point = transform(point)
    point_rotated = transform(point_rotated)

    point = utonia.data.collate_fn([point, point_rotated])

    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor) and device == "cuda":
                point[key] = point[key].cuda(non_blocking=True)
        # model forward:
        point = model(point)
        # upcast point feature
        # Point is a structure contains all the information during forward
        # Use range(4) to upcast features from all levels for quantitative evaluation
        for _ in range(2):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
    batched_coord = point.coord.clone()
    batched_coord[:, 2] += point.batch * bias[2]
    batched_color = point.color.clone()
    pca_color = get_pca_color(point.feat, brightness=1.2, center=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(batched_coord.cpu().detach().numpy())
    pcd.colors = o3d.utility.Vector3dVector(pca_color.cpu().detach().numpy())
    o3d.visualization.draw_geometries([pcd])
    # or
    # o3d.visualization.draw_plotly([pcd])

    # # Export PCA
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(batched_coord.cpu().detach().numpy())
    # pcd.colors = o3d.utility.Vector3dVector(batched_color.cpu().detach().numpy())
    # o3d.io.write_point_cloud("pc.ply", pcd)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(batched_coord.cpu().detach().numpy())
    # pcd.colors = o3d.utility.Vector3dVector(pca_color.cpu().detach().numpy())
    # o3d.io.write_point_cloud("pca.ply", pcd)

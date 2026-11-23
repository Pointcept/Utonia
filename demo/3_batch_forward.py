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


import argparse
import copy
import numpy as np
import utonia
import torch
import open3d as o3d

from pcd_util import load_point_cloud

try:
    import flash_attn
except ImportError:
    flash_attn = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_pca_color(feat, brightness=1.25, center=True):
    u, s, v = torch.pca_lowrank(feat, center=center, q=6, niter=5)
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
        "--model",
        choices=["utonia", "utonia_small", "utonia_tiny"],
        default="utonia",
        help="pre-trained Utonia encoder to use (default: utonia).",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="your own point clouds (.pcd, .ply, .xyz, ...) in meters, or any units "
        "with --object; default: two copies of the sample scene.",
    )
    parser.add_argument(
        "--object",
        action="store_true",
        help="inputs are single objects: rescale each like demo/6_pca_object.py.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="transform scale, larger = finer (default 0.5 for scenes, 1.0 with --object; "
        "the other demos use 0.2 outdoor, 4.0 tabletop).",
    )
    parser.add_argument(
        "--up",
        choices=["z", "-z", "y", "-y"],
        default="z",
        help="axis pointing up in your files; they are rotated so it becomes +z, "
        "which Utonia expects (write negative axes as --up=-z).",
    )
    args = parser.parse_args()
    # set random seed
    # (random seed affect pca color, yet change random seed need manual adjustment kmeans)
    # (the pca prevent in paper is with another version of cuda and pytorch environment)
    utonia.utils.set_seed(53124)
    # Load model
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
    # Load default data transform pipeline
    if args.scale is None:
        args.scale = 1.0 if args.object else 0.5
    transform = utonia.transform.default(args.scale)
    # Load data
    if args.input is None:
        point1 = utonia.data.load("sample1")
        point1.pop("segment200")
        segment = point1.pop("segment20")
        point1["segment"] = segment  # two kinds of segment exist in ScanNet, only use one
        point2 = copy.deepcopy(point1)
        points = [point1, point2]
    else:
        points = [
            load_point_cloud(path, normalize=args.object, up=args.up)
            for path in args.input
        ]
        print("Left to right:", ", ".join(args.input))
    points = [transform(point) for point in points]

    # Without FlashAttention, the attention patch size of a batch is capped by its
    # smallest cloud, which degrades the features of the larger clouds. In that
    # case forward each cloud on its own.
    groups = [points] if flash_attn is not None else [[point] for point in points]
    feats, coords, batch_ids, start = [], [], [], 0
    with torch.inference_mode():
        for group in groups:
            point = utonia.data.collate_fn(group)
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
            feats.append(point.feat)
            coords.append(point.coord)
            batch_ids.append(point.batch + start)
            start += len(group)

        # PCA over all clouds together, so similar features get similar colors
        pca_color = get_pca_color(torch.cat(feats), brightness=1.2, center=True)

    # place the clouds side by side along x
    batched_coord = torch.cat(coords).cpu().numpy()
    batch = torch.cat(batch_ids).cpu().numpy()
    spacing = 1.2 * max(np.ptp(batched_coord[batch == b, 0]) for b in range(len(points)))
    batched_coord[:, 0] += batch * spacing

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(batched_coord)
    pcd.colors = o3d.utility.Vector3dVector(pca_color.cpu().detach().numpy())
    o3d.visualization.draw_geometries([pcd])

    # o3d.io.write_point_cloud("batch.ply", pcd)

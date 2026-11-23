import numpy as np
import open3d as o3d

# Rotations that turn each "up" axis into +z, the up axis Utonia expects.
UP_TO_Z = {
    "z": np.eye(3),
    "-z": np.diag([1.0, -1.0, -1.0]),
    "y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "-y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
}


def load_point_cloud(path, normalize=True, up="z"):
    """Load a point cloud file (.pcd, .ply, .xyz, ...) into a Utonia data dict.

    normalize=True centers the cloud and scales its longest side to 1, like the
    sample object (use for objects or data in arbitrary units). Otherwise the
    coordinates are kept as they are and should be in meters. `up` is the axis
    that points up in the file; the cloud is rotated so that it becomes +z.
    """
    pcd = o3d.io.read_point_cloud(
        path, remove_nan_points=True, remove_infinite_points=True
    )
    if not pcd.has_points():
        raise ValueError(f"No points loaded from {path}")
    rotation = UP_TO_Z[up]
    coord = np.asarray(pcd.points) @ rotation.T
    if normalize:
        bbox_min, bbox_max = coord.min(0), coord.max(0)
        coord = (coord - (bbox_min + bbox_max) / 2) / (bbox_max - bbox_min).max()
    point = {"coord": coord.astype(np.float32)}
    # Open3D colors are in [0, 1]; the transform's NormalizeColor expects [0, 255].
    if pcd.has_colors():
        point["color"] = (np.asarray(pcd.colors) * 255).astype(np.float32)
    else:
        print(f"No colors in {path}; using zeros.")
        point["color"] = np.zeros_like(point["coord"])
    if not pcd.has_normals():
        pcd.estimate_normals()
    point["normal"] = (np.asarray(pcd.normals) @ rotation.T).astype(np.float32)
    return point

# Utonia
**TL;DR:** This repo provide cross-domain pre-trained [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3) for 3D point cloud downstream tasks, modified from [Sonata](https://github.com/facebookresearch/sonata.git) and [Concerto](https://pointcept.github.io/Concerto/).

This repo is the official project repository of the paper **_Utonia: Toward One Encoder for All Point Clouds_** (ICML 2026) and is mainly used for providing pre-trained models, inference code and visualization demo. For reproduce pre-training process of Utonia, please refer to our [Pointcept](https://github.com/Pointcept/Pointcept) codebase.

[ **Pretrain** ] [ **Utonia** ] - [ [Homepage](https://pointcept.github.io/Utonia/) ] [ [Paper](https://arxiv.org/abs/2603.03283) ] [ [Weight](https://huggingface.co/Pointcept/Utonia) ] [ [Pretrain](https://github.com/Pointcept/Pointcept) ] [ [Bib](#citation) ]


<div align='left'>
<img src="https://raw.githubusercontent.com/pointcept/assets/main/utonia/teaser.png" alt="teaser" width="800" />
</div>

## Highlights
- *October, 2026* : We have uploaded the **Utonia Tiny** and **Utonia Small** pre-trained model weights to [Hugging Face](https://huggingface.co/Pointcept/Utonia), with inference and visualization support in this repo. See **[Quick Start](#quick-start)** for usage.
- *February, 2026* : We provide an easy-to-use inference demo and visualization with our pre-trained model weight in this repo. We highly recommend user begin with is repo for **[quick start](#quick-start)**.

## Overview
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Private Studies](#private-studies)
- [Citation](#citation)

## Installation
This repo is a [uv](https://docs.astral.sh/uv/) project. Every CUDA extension (`spconv`, `torch-scatter`, FlashAttention) installs from a prebuilt wheel, so no local CUDA toolkit or compiler is needed, only an NVIDIA driver supporting CUDA 12.8 or newer.
The environment is Python 3.10 + PyTorch 2.8.0 (CUDA 12.8) on Linux x86_64, which covers GPUs from Volta up to Blackwell (e.g. RTX 50-series). Docker images are also available thanks to [HubertLongshanks](https://github.com/HubertLongshanks): [utonia-cuda128](https://hub.docker.com/r/hughbertlong/utonia-cuda-128-rtx-blackwell) and [utonia-cuda128-flashattn](https://hub.docker.com/r/hughbertlong/utonia-cu128-x86-ubuntu24.04-flashattn), as in [#28](https://github.com/Pointcept/Utonia/issues/28).
- The **standalone mode** is recommended for users who want to use the code for quick inference and visualization:
  ```bash
  # Core dependencies + visualization demos (demo/0-7)
  uv sync

  # Optional add-ons, combine as needed:
  #   --extra flash   FlashAttention (Ampere or newer GPUs); without it the demos fall back to PyTorch SDPA
  #   --group video   video demo (demo/8_pca_video.py), including VGGT
  uv sync --extra flash --group video
  ```

- The **package mode** is recommended for users who want to inject our model into their own codebase. Add it to your uv project (Python 3.10); the prebuilt `torch-scatter` wheel source is picked up from this repo's `pyproject.toml`:
  ```bash
  uv add "utonia @ git+https://github.com/Krevera/pointcept-utonia-demo"
  # or with FlashAttention
  uv add "utonia[flash] @ git+https://github.com/Krevera/pointcept-utonia-demo"
  ```

## Quick Start
***Let's first begin with some simple visualization demo with Utonia, our pre-trained PTv3 model:***
- **Visualization.** We provide the same visualization demos as Sonata and Concerto: the segmentation result, similarity heatmap, and PCA visualization demo in the `demo` folder, including the visualization without color/normal input and the demo for videos lifted by [VGGT](https://github.com/facebookresearch/vggt.git). You can run the following command to visualize the result of `utonia`:
  ```bash
  # For visualization demo without color and normal, use uv run python demo/xxx.py --wo_color --wo_normal
  uv run python demo/0_pca_indoor.py
  uv run python demo/1_similarity.py
  uv run python demo/2_sem_seg.py  # linear probed head on ScanNet
  uv run python demo/3_batch_forward.py
  uv run python demo/4_pca_hk.py
  uv run python demo/5_pca_manipulation.py
  uv run python demo/6_pca_object.py
  uv run python demo/7_pca_outdoor.py
  ```

  Select a smaller encoder with `--model utonia_small` or `--model utonia_tiny`:
  ```bash
  uv run python demo/0_pca_indoor.py --model utonia_small
  uv run python demo/3_batch_forward.py --model utonia_tiny
  ```
  All PCA, similarity, batch, and video demos support `--model`; the default remains `utonia`.
  `demo/2_sem_seg.py` uses the ScanNet linear probing head trained for `utonia` and does not support the small/tiny encoders.

  To run the object demo on your own point cloud (`.pcd`, `.ply`, `.xyz`, or any format Open3D reads), pass `--input`. The object is rescaled to the sample's size (centered, longest side 1), so its original units don't matter:
  ```bash
  uv run python demo/6_pca_object.py --input your_object.pcd
  ```

  To run several of your own point clouds together, pass them all to `demo/3_batch_forward.py`. They are shown side by side, and PCA colors are computed jointly, so similar colors mean similar features across clouds. Inputs are assumed to be in meters; add `--object` for single objects in any units. Use `--scale` to set the granularity (default 0.5 for rooms; the other demos use 0.2 for outdoor and 4.0 for tabletop scenes):
  ```bash
  uv run python demo/3_batch_forward.py --input scan_a.pcd scan_b.pcd scan_c.pcd
  uv run python demo/3_batch_forward.py --object --input mug.pcd bowl.pcd
  ```
  Without FlashAttention, a batch's attention patch size is capped by its smallest cloud, which degrades the larger clouds' features, so the demo then runs each cloud separately. With `uv sync --extra flash` the clouds are forwarded as a single batch.
  Utonia expects +z to point up. If your files use another up axis, pass it with `--up` (`--up=y`, `--up=-y` or `--up=-z`), and the clouds are rotated so that axis becomes +z.

  <div align='left'>
  <img src="https://raw.githubusercontent.com/pointcept/assets/main/utonia/demo.png" alt="teaser" width="800" />
  </div>

  For video demo, we need VGGT to lift video to point cloud. The `video` dependency group installs VGGT and the other video demo dependencies:
  ```bash
  uv sync --group video
  # Then you can run the demo
  uv run python demo/8_pca_video.py \
  --input_video ${YOUR_VIDEO_PATH} \
  --conf_thres 0 \
  --prediction_mode "Depthmap and Camera Branch" \
  --if_TSDF \
  --pca_start 1 \
  --pca_brightness 1.2 # the usage of these inpputs can be found in 'help'.
  ```

  To use a few multi-view images (e.g. 4 cameras around a scene) instead of a video, use `demo/8b_pca_multiview.py`. The first image is the reference view. It writes `pcd.ply` and `pca.ply` to `multiview_demo_output/` and opens the PCA result in an Open3D window:
  ```bash
  uv run python demo/8b_pca_multiview.py --images view_0.png view_1.png view_2.png view_3.png
  # --conf_thres (default 50) drops low-confidence VGGT points; lower it to keep more points
  ```

***Then, here are the instruction to run inference on custom data with our Utonia:***

- **Data.** Organize your data in a dictionary with the following format:
  ```python
  # single point cloud
  point = {
    "coord": numpy.array,  # (N, 3)
    "color": numpy.array,  # (N, 3)
    "normal": numpy.array,  # (N, 3)
    "segment": numpy.array,  # (N,) optional
  }

  # batched point clouds

  # check the data structure of batched point clouds from here:
  # https://github.com/Pointcept/Pointcept#offset
  point = {
    "coord": numpy.array,  # (N, 3)
    "color": numpy.array,  # (N, 3)
    "normal": numpy.array,  # (N, 3)
    "batch": numpy.array,  # (N,) optional
    "segment": numpy.array,  # (N,) optional
  }
  ```
  One example of the data can be loaded by running the following command:
  ```python
  point = utonia.data.load("sample1")
  ```
- **Transform.** The data transform pipeline is different from previous Sonata and Concerto. You can use the following code to construct the transform pipeline:
  ```python
  # The choice of scale depends on the desired level of granularity. You may need to fine-tune this parameter to find the optimal value for your specific tasks. Generally, the larger the scale, the more fine-grained the results will be.
  config = [
      # dict(type="NormalizeCoord"), # uncomment when applying to objects
      dict(type="RandomScale", scale=[scale, scale]),
      dict(type="CenterShift", apply_z=True), # remove this for outdoor LiDAR and ensure the ego-vehicle is at the origin (0, 0, 0), with the road plane aligned with the xy-plane
      dict(
          type="GridSample",
          grid_size=0.01,
          hash_type="fnv",
          mode="train",
          return_grid_coord=True,
          return_inverse=True,
      ),
      dict(type="NormalizeColor"),
      dict(type="ToTensor"),
      dict(
          type="Collect",
          keys=("coord", "grid_coord", "color", "inverse"),
          feat_keys=("coord", "color", "normal"),
      ),
  ]
  transform = utonia.transform.Compose(config)
  ```
  The above default inference augmentation pipeline can also be acquired by running the following command:
  ```python
  # apply_z_positive = False when applying to outdoor data to ensure the road is on xy-plane, normalize_coord = True when applying to the single object
  transform = utonia.transform.default(scale = scale, apply_z_positive = True, normalize_coord = False)
  ```
- **Model.** Load the pre-trained model by running the following command:
  ```python
  # Load the pre-trained model from Huggingface
  # When color or normal is absent, please apply default zeros to the missing modality
  # ckpt is cached in ~/.cache/utonia/ckpt, and the path can be customized by setting 'download_root'
  model = utonia.model.load("utonia", repo_id="Pointcept/Utonia").cuda()

  # or
  from utonia.model import PointTransformerV3
  model = PointTransformerV3.from_pretrained("Pointcept/Utonia").cuda()

  # Load the small or tiny encoder
  model = utonia.model.load("utonia_small").cuda()
  model = utonia.model.load("utonia_tiny").cuda()

  # Without FlashAttention, override the attention backend
  model = utonia.model.load(
      "utonia_tiny", custom_config=dict(enable_flash=False)
  ).cuda()

  # Load the pre-trained model from local path
  # assume the ckpt file is stored in the 'ckpt' folder
  model = utonia.model.load("ckpt/utonia.pth").cuda()
  model = utonia.model.load("ckpt/utonia_small.pth").cuda()
  model = utonia.model.load("ckpt/utonia_tiny.pth").cuda()

  # the ckpt file store the config and state_dict of pretrained model
  ```

  Model architecture and feature widths are read from each checkpoint automatically.
  Downstream heads must match the selected encoder's feature width; the existing
  `utonia_linear_prob_head_sc` checkpoint is only compatible with `utonia`.

- **Inference.** Run the inference by running the following command:
  ```python
  point = transform(point)
  for key in point.keys():
      if isinstance(point[key], torch.Tensor):
          point[key] = point[key].cuda(non_blocking=True)
  point = model(point)
  ```
  As Utonia is a pre-trained **encoder-only** PTv3, the default output of the model is point cloud after hierarchical encoding. The encoded point feature can be mapping back to original scale with the following code:
  ```python
  # use range(4) to concatenate the features from all the upcast layers, here we use range 2 for visualization
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
  ```
  Yet during data transformation, we operate `GridSampling` which makes the number of points feed into the network mismatch with the original point cloud. Using the following code to further map the feature back to the original point cloud:
  ```python
  feat = point.feat[point.inverse]
  ```

## Private Studies
`study/` holds analyses on internal data, so it is git-ignored in this public repo and stored in S3 at `s3://thindev/pointcept-utonia-demo/study`. With access to the bucket (for SSO, run `aws sso login` first):
```bash
scripts/study_sync.sh pull            # download or update study/ from S3
scripts/study_sync.sh push --dryrun   # preview your changes
scripts/study_sync.sh push            # upload your changes
```
The script wraps `aws s3 sync` and never transfers `__pycache__`. Extra arguments go to `aws s3 sync`; for example, add `--delete` to also remove files that no longer exist on the source side.

## Citation
If you find _Utonia_ useful to your research, please consider citing our works as an acknowledgment. (੭ˊ꒳​ˋ)੭✧
```bib
@inproceedings{zhang2026utonia,
    title={Utonia: Toward One Encoder for All Point Clouds}, 
    author={Yujia Zhang and Xiaoyang Wu and Yunhan Yang and Xianzhe Fan and Han Li and Yuechen Zhang and Zehao Huang and Naiyan Wang and Hengshuang Zhao},
    booktitle={ICML},
    year={2026},
}
```

```bib
@inproceedings{zhang2025concerto,
    title={Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations},
    author={Zhang, Yujia and Wu, Xiaoyang and Lao, Yixing and Wang, Chengyao and Tian, Zhuotao and Wang, Naiyan and Zhao, Hengshuang},
    booktitle={NeurIPS},
    year={2025}
}
```

```bib
@inproceedings{wu2025sonata,
    title={Sonata: Self-Supervised Learning of Reliable Point Representations},
    author={Wu, Xiaoyang and DeTone, Daniel and Frost, Duncan and Shen, Tianwei and Xie, Chris and Yang, Nan and Engel, Jakob and Newcombe, Richard and Zhao, Hengshuang and Straub, Julian},
    booktitle={CVPR},
    year={2025}
}
```

```bib
@inproceedings{wu2024ptv3,
    title={Point Transformer V3: Simpler, Faster, Stronger},
    author={Wu, Xiaoyang and Jiang, Li and Wang, Peng-Shuai and Liu, Zhijian and Liu, Xihui and Qiao, Yu and Ouyang, Wanli and He, Tong and Zhao, Hengshuang},
    booktitle={CVPR},
    year={2024}
}
```
```bib
@inproceedings{wu2024ppt,
    title={Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt Training},
    author={Wu, Xiaoyang and Tian, Zhuotao and Wen, Xin and Peng, Bohao and Liu, Xihui and Yu, Kaicheng and Zhao, Hengshuang},
    booktitle={CVPR},
    year={2024}
}
```
```bib
@inproceedings{wu2023masked,
  title={Masked Scene Contrast: A Scalable Framework for Unsupervised 3D Representation Learning},
  author={Wu, Xiaoyang and Wen, Xin and Liu, Xihui and Zhao, Hengshuang},
  journal={CVPR},
  year={2023}
}
```
```bib
@inproceedings{wu2022ptv2,
    title={Point transformer V2: Grouped Vector Attention and Partition-based Pooling},
    author={Wu, Xiaoyang and Lao, Yixing and Jiang, Li and Liu, Xihui and Zhao, Hengshuang},
    booktitle={NeurIPS},
    year={2022}
}
```
```bib
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished={\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```

## License

- Utonia code is based on Sonata code, which is released by Meta under the [Apache 2.0 license](LICENSE);
- Utonia weight is released under the [CC-BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en) (restricted by NC of datasets like HM3D, ArkitScenes).

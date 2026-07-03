# Uni3D DataSkill

Executable helpers for adapting multi-source 3D datasets into Pi3X-compatible
PyTorch dataloaders.

Chinese documentation: [README.ch-ZN.md](README.ch-ZN.md)

## Status

Current workflow:

```text
raw dataset root
  -> Pi3X-compatible Dataset implementation
  -> PyTorch DataLoader
  -> sequence summary
```

Implemented loaders include KITTI-360, KITTI raw with official aligned depth,
KITTI odometry, nuScenes table layouts, WayveScenes-style transforms, Waymo
KITTI-style converted layouts, SAGE-10k sampled route layouts, and BlendedMVG.

## Requirements

- Python 3.10 or newer.
- A local conda or Python environment with PyTorch-compatible dependencies.
- Pi3 training branch checked out at `thirdparty/Pi3`.

## Installation

Clone this repository and create or activate your Python environment:

```bash
git clone <this-repository-url>
cd Uni3D-DataSkill
conda activate <env-name>
```

Install this package in editable mode:

```bash
python -m pip install -e .
```

Clone Pi3 into the required third-party path:

```bash
git clone https://github.com/yyfz/Pi3.git thirdparty/Pi3
cd thirdparty/Pi3
git checkout training
```

Install Pi3 dependencies:

```bash
python -m pip install -r thirdparty/Pi3/requirements.txt
```

## Dataset Config

Use a local JSON config to map dataset labels to dataset roots. The example
file contains one entry for each supported loader. Copy it, keep the entries
you need, and replace the paths with mounted local paths:

```bash
cp dataset_config.example.json dataset_config.local.json
```

Example:

```json
{
  "datasets": [
    {
      "label": "kitti360_train",
      "dataset": "kitti360",
      "root": "/mnt/datasets/KITTI-360",
      "roots": {
        "calibration": "/mnt/datasets/KITTI-360/calibration",
        "images": "/mnt/datasets/KITTI-360/data_2d_raw",
        "poses": "/mnt/datasets/KITTI-360/data_poses"
      },
      "optional_roots": {
        "lidar": "/mnt/datasets/KITTI-360/data_3d_raw",
        "semantics_2d": null,
        "semantics_3d": null
      },
      "sequences": ["2013_05_28_drive_0000_sync"],
      "cameras": ["image_00"]
    }
  ]
}
```

`root` should be a local path or an already mounted path that the current
Python process can read. If the dataset follows the loader's default official
layout under `root`, `roots` can be omitted. Use `roots` when required
components are mounted separately, and use `optional_roots` for additional
modalities that may be unavailable.

Supported dataset keys:

| Dataset key | Loader |
| --- | --- |
| `ase`, `aria-synthetic-environments`, `aria_synthetic_environments` | Aria Synthetic Environments RGB-D layout |
| `arkitscenes`, `arkit-scenes`, `arkit` | ARKitScenes raw RGB-D layout |
| `blendedmvs`, `blendedmvg` | BlendedMVG layout |
| `hypersim`, `hyper-sim` | Hypersim scene/camera HDF5 layout |
| `kitti360`, `kitti-360` | KITTI-360 raw perspective layout |
| `kitti`, `kitti-odometry` | KITTI odometry-style layout |
| `kitti-raw`, `kitti_raw`, `kitti-depth`, `kitti-depth-completion` | KITTI raw layout with official depth completion ground truth |
| `nuscenes` | nuScenes JSON table layout |
| `sage`, `sage-10k`, `sage10k` | SAGE-10k sampled route layout |
| `uco3d`, `uco3d-depth` | uCO3D official package wrapper |
| `wayve`, `wayvescenes`, `wayvescenes101` | WayveScenes/Nerfstudio-style transforms |
| `waymo-kitti`, `waymo_kitti`, `waymo-converted-kitti` | Waymo converted to KITTI-style layout |

## Usage

Validate a dataset entry from a config file:

```bash
python -m unidata_skill sample-dataset \
  --config dataset_config.local.json \
  --label kitti360_train
```

Omit `--label` to iterate over every dataset entry in the config.

For datasets that use a generated NumPy index, rebuild the index first:

```bash
python -m unidata_skill reindex-dataset \
  --config dataset_config.local.json \
  --label sage_sample
```

`reindex-dataset` writes to `index_file`; `sample-dataset` reads from the same
field.

The sampling command prints sequence-level information and one sampled batch
summary:

```text
label: kitti360_train
dataset: kitti360
root: /mnt/datasets/KITTI-360
num sequences: 1234
first sequences: [...]
num_imgs:
  2013_05_28_drive_0000_sync: 11518
sample index: 0
sample views: 8
```

## Correspondence Dataset Builder

The current annotation-tool work is implemented as a standalone correspondence
builder. It constructs the configured Pi3X dataloaders from
`src/unidata_skill/datasets`, reads the views returned by those loaders, then
builds positives from two GT-supervised sources by default: dense geometry
correspondences from depth back-projection, and source-image feature points
projected into the target frame with GT depth, intrinsics, and pose. Both paths
are filtered by depth range and target-depth consistency. The builder writes
correspondence arrays plus an optional visualization for every successful pair.
By default, pair views stay in the dataset's native image resolution so feature
extraction, dense geometry projection, saved coordinates, and visualizations all
use original-image coordinates. Add `--resize-views` only when you intentionally
want the Pi3 crop/resize path controlled by `--width` and `--height`.

```bash
python tools/build_correspondence_dataset.py \
  --config dataset_config.local.json \
  --n-corres 8192 \
  --nneg 0.5 \
  --frame-gap 1
```

Use `--positive-source geometry`, `--positive-source features`, or the default
`--positive-source mixed` to choose the positive source. The feature path uses
`--feature-method sift` by default and supports the same extractor names as the
single-pair feature projection demo. Tune `--depth-consistency-thresh` for
projection/depth filtering. Dense geometry uses CPU by default; pass
`--geometry-device cuda` to run the projection/filtering tensors on GPU, and
use `--geometry-stride <N>` to project every N-th source pixel before filtering
when full-resolution dense projection is too slow. In `mixed` mode, geometry and feature positives are unioned;
duplicate pairs are marked as `both`, and sampling keeps feature-related
positives from being drowned out by dense geometry positives. By default, the
builder writes the requested sampled correspondence count. Use
`--save-stride <N>` to stride saved positives and negatives when a large
`--n-corres` request would write too many points. Add `--no-visualization` when
building arrays in an environment without Matplotlib or when image previews are
not needed. Pairs are generated in sequence order as `(frame_i, frame_i +
--frame-gap)`, so `--frame-gap 1` builds adjacent-frame pairs and larger values
build fixed-gap pairs. The builder processes every selected sequence in full;
use `--width` and `--height` to control dataloader resolution.

Each saved `.npz` contains VGGT-style `tracks`, `track_vis_mask`, and
`track_positive_mask` fields with shape `S=2`, plus compatibility fields
`corres1`, `corres2`, `valid_corres`, and `distance_m` after save-stride
subsampling. It also stores `positive_source_code`, `feature_score`,
`target_depth_error_m`, `requested_n_corres`, and `save_stride` for diagnosing
whether sampled positives came from the geometry path, feature path, or both,
and how many were written.
`manifest.jsonl` records the `.npz` path, visualization path, source/target
frame metadata, and match counts. The older
`tools/kitti_npy_match_cropping_demo.py` remains available as a single-pair
debug demo. The builder processes every selected dataset entry in the config
through the same loader registry used by `sample-dataset`. Per-dataset outputs
are written below a label-named subdirectory under `--output-dir`.

For visual feature experiments, `tools/kitti_npy_feature_match_demo.py` extracts
features from the source image only, uses GT depth, intrinsics, and pose to
project those feature points into the target image, then filters the projected
points with target depth consistency. Projected feature correspondences are
rendered as short cross markers:

```bash
python tools/kitti_npy_feature_match_demo.py \
  --index-file /path/to/index.npy \
  --feature-method sift
```

`--feature-method` supports `sift`, `aliked`, `superpoint`, `sp`, and
`lightglue_sift`. OpenCV SIFT is used for `sift`; ALIKED, SuperPoint, and
LightGlue SIFT require the `lightglue` package. The demo does not run image
descriptor matching between the two frames. It rejects source features with
invalid or out-of-range depth, target projections outside the target image, and
target projections whose projected depth disagrees with the target depth by more
than `--depth-consistency-thresh`; tune the valid depth range with
`--min-depth` and `--max-depth`.

## Dataset Layouts

Each dataloader reads its own official or converted raw layout directly.
KITTI-360 example:

```text
KITTI-360/
  calibration/
    perspective.txt
    calib_cam_to_pose.txt
  data_2d_raw/
    2013_05_28_drive_0000_sync/
      image_00/
        data_rect/
          0000000000.png
          ...
  data_poses/
    2013_05_28_drive_0000_sync/
      cam0_to_world.txt
```

## Development

Run tests with:

```bash
python -m unittest discover -s tests -v
```

## Known Limitations

- No index, cache, or intermediate target schema is generated for direct
  dataloaders.
- Dense depth, point cloud projection, semantic labels, instance labels, and 3D
  boxes are not loaded for the current autonomous-driving direct loaders.
- nuScenes and Waymo native point cloud/depth projection are not implemented in
  the current direct loaders.
- Waymo support currently targets KITTI-style converted layouts, not native
  TFRecord/protobuf tables.

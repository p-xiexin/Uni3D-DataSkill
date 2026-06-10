# Uni3D DataSkill

Executable helpers for adapting multi-source 3D datasets into Pi3X-compatible
PyTorch dataloaders. The current implementation focuses on direct dataloader
validation for raw dataset layouts that already provide images, calibration,
and poses.

## Status

This repository is in an early executable MVP stage. The active path is:

```text
raw dataset root
  -> Pi3X-compatible Dataset implementation
  -> PyTorch DataLoader
  -> validation report
```

The repository contains initial direct dataloaders for KITTI-360 perspective
rectified imagery, KITTI odometry, nuScenes table layouts, WayveScenes-style
transforms, Waymo KITTI-style converted layouts, and BlendedMVG.

## Requirements

- Python 3.10 or newer.
- A local conda or Python environment with PyTorch-compatible dependencies.
- Pi3 training branch checked out at `thirdparty/Pi3`.

Pi3 is required because the dataloaders inherit from Pi3 training's
`datasets.base.base_dataset.BaseDataset`. The project intentionally resolves
Pi3 only from `thirdparty/Pi3`.

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

Pi3 does not currently provide Python packaging metadata for
`python -m pip install -e .`. Install its Python dependencies instead:

```bash
python -m pip install -r thirdparty/Pi3/requirements.txt
```

After this, Uni3D DataSkill will add `thirdparty/Pi3` to `sys.path` at runtime
and import Pi3's dataset code directly.

## Dataset Config

Use a local JSON config to map dataset labels to dataset roots. Start from the
example file:

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
      "root": "/path/to/KITTI-360",
      "sequences": ["2013_05_28_drive_0000_sync"],
      "cameras": ["image_00"],
      "frame_num": 8,
      "stride": 5,
      "resolution": "512x384",
      "max_samples": 4,
      "batch_size": 1
    }
  ]
}
```

Supported dataset keys currently include:

| Dataset key | Loader |
| --- | --- |
| `kitti360`, `kitti-360` | KITTI-360 raw perspective layout |
| `kitti`, `kitti-odometry` | KITTI odometry-style layout |
| `nuscenes` | nuScenes JSON table layout |
| `wayve`, `wayvescenes`, `wayvescenes101` | WayveScenes/Nerfstudio-style transforms |
| `waymo-kitti`, `waymo_kitti`, `waymo-converted-kitti` | Waymo converted to KITTI-style layout |
| `blendedmvs`, `blendedmvg` | BlendedMVG layout |

## Usage

Validate a dataset entry from a config file:

```bash
python -m unidata_skill validate-config \
  --config dataset_config.local.json \
  --label kitti360_train
```

The validator prints a JSON report:

```json
{
  "status": "ok",
  "dataset_len": 1234,
  "checked_samples": 4,
  "checked_batches": 1,
  "errors": [],
  "warnings": []
}
```

## Dataset Layouts

Each dataloader reads its own official or converted raw layout directly. For
example, the KITTI-360 loader expects at least:

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

Dense depth is not loaded by the current autonomous-driving direct loaders.
Returned `depthmap` values may be placeholders used to satisfy the Pi3X view
contract and must not be treated as ground-truth depth unless the loader
explicitly documents a native or derived depth source.

## Development

Run tests with:

```bash
python -m unittest discover -s tests -v
```

Tests use the real Pi3 checkout under `thirdparty/Pi3`. If Pi3 dependencies are
not installed yet, imports may fail with the missing dependency reported by Pi3.

## Known Limitations

- No index, cache, or intermediate target schema is generated for direct
  dataloaders.
- Dense depth, point cloud projection, semantic labels, instance labels, and 3D
  boxes are not loaded yet for the current autonomous-driving direct loaders.
- nuScenes and Waymo native point cloud/depth projection are not implemented in
  the current direct loaders.
- Waymo support currently targets KITTI-style converted layouts, not native
  TFRecord/protobuf tables.

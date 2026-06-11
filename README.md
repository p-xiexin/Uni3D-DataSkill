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
  -> validation report
```

Implemented loaders include KITTI-360, KITTI odometry, nuScenes table layouts,
WayveScenes-style transforms, Waymo KITTI-style converted layouts, and
BlendedMVG.

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

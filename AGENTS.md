# Repository Agent Guide

This repository is an executable dataset-loading toolkit for multi-source 3D
datasets. The current implementation is centered on direct Pi3X-compatible
PyTorch dataloaders.

## Current Architecture

The active flow is:

```text
raw or converted dataset root
  -> dataset-specific Pi3X dataloader
  -> Pi3 BaseDataset behavior
  -> PyTorch DataLoader
  -> sequence summary
```

The implementation lives mainly in:

```text
src/unidata_skill/
  cli.py
  config.py
  datasets/
    __init__.py
    blendedmvg_dataset.py
    kitti360_dataset.py
    kitti_odometry_dataset.py
    nuscenes_dataset.py
    pi3x_validator.py
    waymo_kitti_dataset.py
    wayve_dataset.py
tests/
```

Use the actual package layout, README files, tests, and current dataloader
patterns as the operating reference.

## Required Local Environment

Use the local conda environment named `huawei` for development, validation, and
test commands.

Confirmed local paths:

```text
Conda root: D:\miniconda3
Conda executable: D:\miniconda3\Scripts\conda.exe
Environment: D:\miniconda3\envs\huawei
Python: D:\miniconda3\envs\huawei\python.exe
Python version: 3.11.15
```

In interactive shells:

```powershell
conda activate huawei
```

In non-interactive PowerShell sessions, call the environment Python directly:

```powershell
$env:PYTHONPATH='src'
D:\miniconda3\envs\huawei\python.exe -m unittest discover -s tests -v
```

## Pi3 Dependency

Pi3 is a required third-party checkout at:

```text
thirdparty/Pi3
```

It should be `https://github.com/yyfz/Pi3.git` on the `training` branch.

Setup:

```powershell
git clone https://github.com/yyfz/Pi3.git thirdparty/Pi3
cd thirdparty/Pi3
git checkout training
python -m pip install -r requirements.txt
```

Pi3 does not currently install as an editable Python package in this workflow.
This repository adds `thirdparty/Pi3` to `sys.path` once in
`src/unidata_skill/datasets/__init__.py`.

Do not add `--pi3-root`, `PI3_ROOT` environment variables, fake Pi3 packages, or
alternate Pi3 discovery logic unless the user explicitly asks for it.

## Dataset Loader Pattern

Each dataset gets its own concrete dataloader file. Keep implementations easy
to debug before introducing shared abstractions.

Use this import and inheritance style in every dataset file:

```python
from datasets.base.base_dataset import BaseDataset


class XxxPi3XDataset(BaseDataset):
    ...
```

Do not add:

- helper files such as `pi3x.py` or `pi3_base.py`
- dynamic class factories for normal dataset classes
- per-file Pi3 path insertion blocks
- fallback fake `BaseDataset` implementations
- local resize/crop compatibility branches when Pi3
  `_crop_resize_if_necessary()` is available

Keep dataset-specific logic local to each dataloader:

- raw layout discovery
- image/table parsing
- calibration loading
- pose loading
- dataset-specific view sampling inside `_get_views`
- placeholder fields for unavailable geometry

KITTI-360 is not special. Treat it the same as KITTI odometry, nuScenes, Wayve,
Waymo KITTI-style, BlendedMVG, and future direct loaders.

## Pi3 Dataset Style Reference

When adding or adapting loaders, follow the style used by Pi3 examples such as
`thirdparty/Pi3/datasets/tartanair_dataset.py`,
`thirdparty/Pi3/datasets/scannet_dataset.py`, and
`thirdparty/Pi3/datasets/co3dv2_dataset.py`.

These loaders assume `data_root` already points at a readable official or
preprocessed dataset layout. They do not download raw data, unpack archives,
copy files into a working tree, or normalize every dataset into a shared
directory schema at runtime.

Use fixed, dataset-native directory expectations inside each concrete loader:

- TartanAir-style loaders can directly scan scene folders and read files such
  as `image_left`, `depth_left`, and `pose_left.txt`.
- ScanNet-style loaders can directly expect per-scene folders such as `color`,
  `depth`, `pose`, and `intrinsic`.
- CO3Dv2-style loaders can directly read official annotation files such as
  `*_train.jgz` and `*_test.jgz`, then resolve referenced `images`, `depths`,
  and optional `masks`.

If a loader writes under `data/dataset_cache`, treat that cache as lightweight
index metadata only, such as sequence lists or image counts. Do not use it as a
place to store extracted raw datasets, converted images, generated geometry, or
alternate dataset layouts.

## CLI Pattern

The public CLI has two separate responsibilities:

```powershell
python -m unidata_skill reindex-dataset --config <config.json> --label <label>
python -m unidata_skill sample-dataset --config <config.json> --label <label>
```

Omit `--label` to process every dataset entry in the config.

`reindex-dataset` rebuilds lightweight JSON indexes from raw dataset roots. It
uses `index_file` as the output path.

`sample-dataset` constructs the dataloader and runs one sampling probe. It uses
the same `index_file` as the read path for datasets that support precomputed
indexes.

Dataset construction is registered in `DATASET_LOADERS` in
`src/unidata_skill/cli.py` using direct class references, not module/class
strings.

Each loader registry entry should define:

- aliases
- class
- constructor defaults for data-source choices such as default cameras or mode
- default `frame_num`
- default `resolution`

Do not add dataset-specific CLI commands unless the user explicitly asks for a
temporary debug command.

## Config And Paths

Dataset configs are JSON files with entries like:

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
      "sequences": ["2013_05_28_drive_0000_sync"]
    }
  ]
}
```

Use a local path or an already mounted path that the current Python process can
read. If an official dataset is laid out directly under `root`, component
`roots` can be omitted. If components are stored separately, set `roots` for
required inputs and `optional_roots` for modalities that may be unavailable.

## Sampling Expectations

Keep `sample-dataset` focused on dataloader probing:

- referenced image paths exist
- returned views have required Pi3X fields
- camera intrinsics are finite and plausible
- pose matrices are valid and invertible
- depth placeholders or depth maps have expected shapes

Do not silently invent missing geometry. If depth, pose, calibration, or labels
are unavailable or placeholders, expose that clearly in fields or warnings.

## Tests

Use tiny synthetic fixtures in tests. Do not restore `tests/fake_pi3` or local
fake `BaseDataset` utilities.

Preferred command:

```powershell
$env:PYTHONPATH='src'
D:\miniconda3\envs\huawei\python.exe -m unittest discover -s tests -v
```

If tests fail while Pi3 dependencies are still being installed, report the
specific missing dependency. Pi3 currently imports `omegaconf` from its code
even though it may not appear in its requirements file.

## Documentation

Keep `README.md` as the English README and `README.ch-ZN.md` as the Chinese
README. Keep both aligned for user-facing setup, config, and validation
instructions.

When editing Markdown:

- keep examples syntactically valid
- use fenced code blocks for commands, JSON, and directory trees
- avoid long clarification sections
- avoid reintroducing dataset-specific CLI commands in docs

## Git And Workspace Rules

- Always run `git status --short` before committing.
- Do not commit unrelated files.
- Do not overwrite or revert user changes in unrelated documents or code.
- Use short imperative commit messages, for example
  `Unify Pi3 dataset loading` or `Document dataset config paths`.
- `data/`, `outputs/`, and local dataset configs are local/generated assets and
  should not be committed unless the user explicitly asks.

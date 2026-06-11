# Dataset Path Configuration Plan

## Problem

Current dataloaders tend to assume one fixed on-disk directory layout. This is
fragile for professional dataset use because official datasets are often
downloaded as separate packages, split by scene, sensor, annotation type,
benchmark task, or data source. After extraction, the usable parts may live in
different mounted locations.

Hard-coded layout discovery inside dataloaders makes failures harder to debug:
the loader has to both discover data and read samples, and it may silently pick
the wrong copy when multiple versions or converted layouts exist on a server.

## Direction

Use explicit, user-provided paths for each required dataset component.

The tool is intended for professional users, so the configuration should favor
auditable, reproducible paths over automatic guessing. Automatic inspection can
help generate a draft config, but training and validation should rely on the
explicit config.

## Proposed Config Shape

Use `roots` for required components and `optional_roots` for components that can
be absent without making the dataset unusable.

```json
{
  "dataset": "kitti360",
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
```

Guidelines:

- `roots` contains the minimum paths needed to construct valid samples.
- `optional_roots` contains additional modalities such as LiDAR, depth,
  semantics, masks, meshes, or annotations.
- Missing required roots should fail early with a clear config error.
- Missing optional roots should be represented explicitly as unavailable data,
  not silently invented placeholders.
- Dataset files should not scan unrelated parent directories during normal
  loading.

## Loader Structure

Keep one dataloader per dataset for debugging, but separate path resolution from
sample reading.

```text
config roots
  -> dataset-specific path resolver / index builder
  -> explicit sample index or manifest
  -> Pi3-compatible dataloader
```

The dataloader should read from resolved paths or a generated in-memory index.
It should not guess where official packages were extracted.

## Benefits

- Better fit for server-mounted datasets and separated official downloads.
- Fewer false positives from old versions, converted copies, caches, or symlinks.
- Clearer error messages such as `roots.poses does not exist`.
- Better reproducibility because the config fully documents the data source.
- Easier debugging because path issues are separated from tensor/sample issues.

## Implementation Notes

- Introduce the explicit `roots` / `optional_roots` schema dataset by dataset.
- Keep backward compatibility only if it does not add hidden path guessing.
- Prefer early validation in config parsing before constructing PyTorch datasets.
- Do not add a universal scanner as the primary mechanism.
- Treat dataset-specific official layouts as validation rules, not as implicit
  filesystem search logic.

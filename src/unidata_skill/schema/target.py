"""Target schema constants for the first executable workflow."""

TARGET_ARTIFACTS = (
    "dataset_meta.json",
    "frames.jsonl",
    "cameras.json",
    "poses.jsonl",
    "pairs.jsonl",
    "annotations/",
    "depth/",
    "rgb/",
    "point_clouds/",
    "meshes/",
    "qa_report.json",
)

MODALITY_SOURCES = (
    "native",
    "rendered",
    "sampled",
    "derived",
    "estimated",
    "missing",
)

from pathlib import Path
import numpy as np

from src.unidata_skill.datasets.sage_dataset import SagePi3XDataset


data_root = "/mnt/nas165/open_source/SAGE-10k-0522"

ds = SagePi3XDataset(
    data_root=data_root,
    domains=["blend"],
    layouts=["layout_21027b7b"],
    settings=["yaw_amplitude_0522"],
    route_ids=None,   # 先可以不填；如果还慢，就填具体 route
    frame_num=8,
    stride=1,
    resolution=(512, 384),
    verbose=True,
)

print("num sequences:", len(ds))
print("first sequences:", ds.sequences[:5])

if len(ds) == 0:
    raise RuntimeError("No valid routes found.")

rng = np.random.default_rng(0)

views = ds._get_views(
    index=0,
    resolution=[512, 384],
    rng=rng,
    is_test=True,
)

print("num views:", len(views))

for i, v in enumerate(views):
    print("=" * 80)
    print("view:", i)
    print("label:", v["label"])
    print("instance:", v["instance"])
    print("image_path:", v["image_path"])
    print("depth_path:", v["depth_path"])

    img = v["img"]
    if hasattr(img, "shape"):
        print("img:", type(img), img.shape, img.dtype)
    else:
        print("img:", type(img), "size:", img.size, "mode:", img.mode)

    print(
        "depthmap:",
        type(v["depthmap"]),
        v["depthmap"].shape,
        v["depthmap"].dtype,
        float(v["depthmap"].min()),
        float(v["depthmap"].max()),
    )

    print("camera_pose:", v["camera_pose"].shape, v["camera_pose"].dtype)
    print("camera_intrinsics:", v["camera_intrinsics"].shape, v["camera_intrinsics"].dtype)
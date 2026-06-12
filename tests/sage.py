from pathlib import Path

from unidata_skill.datasets.sage_dataset import SagePi3XDataset, generate_sage_index


data_root = Path("/mnt/nas165/open_source/SAGE-10k-0522")
index_file = Path("tests/.cache/sage_index.npy")

domains = ["blend"]
layouts = ["layout_21027b7b"]
settings = ["yaw_amplitude_0522"]
route_ids = None

generate_sage_index(
    data_root=data_root,
    output_path=index_file,
    domains=domains,
    layouts=layouts,
    settings=settings,
    route_ids=route_ids,
)

ds = SagePi3XDataset(
    data_root=data_root,
    index_file=index_file,
    domains=domains,
    layouts=layouts,
    settings=settings,
    route_ids=route_ids,
    frame_num=8,
    resolution=[[512, 384]],
    verbose=True,
)

print("num sequences:", len(ds))
print("first sequences:", ds.sequences[:5])

if len(ds) == 0:
    raise RuntimeError("No valid routes found.")

views = ds[0]

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
        print("img:", type(img), "size:", img.size if hasattr(img, "size") else None)

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

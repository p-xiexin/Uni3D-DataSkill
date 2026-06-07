# SKILL.md — 开源三维/多视图数据集自动适配与目标格式构建

版本：v0.1  
适用方向：三维重建、前馈几何模型训练、SLAM/NVS/3DGS 数据准备、具身智能仿真数据抽取  
核心目标：让 Agent 能够识别不同开源数据集的原始组织方式，自动抽取 RGB、深度、相机内参、相机位姿、语义/实例标注、点云、mesh、3DGS、文本描述等信息，并统一转换为目标数据集格式。

---

## 0. 使用原则

当用户提出“适配某个数据集”“把某数据集转换成统一训练格式”“为前馈三维模型构建训练样本”“渲染 mesh/USD/GLB 数据为 RGB-D/pose 数据”等需求时，启用本 Skill。

本 Skill 不假定数据集一定已经下载完整。Agent 必须先做三件事：

1. 检查数据集根目录，判断数据集类型与版本。
2. 读取少量样本文件，确认真实目录、单位、坐标系、pose 方向、depth 编码、mask 编码。
3. 生成 `inspection_report.json` 后再执行批量转换。

禁止直接凭数据集名称硬编码转换逻辑。官方说明、论文、仓库 README 只能作为初始假设；真正转换前必须用样本文件验证。

---

## 1. 目标数据集格式设计

### 1.1 设计动机

前馈三维模型通常需要多视图 RGB、相机内参、相机位姿、可选深度/点云/语义/文本约束。DUSt3R、MASt3R、VGGT、Pi3 等方法的共同点是从单帧、多帧或图像对中学习几何关系。MASt3R-Fusion 这类系统进一步表明，前馈模型输出的 pointmap、descriptor、dense matching 与 Sim(3)/SE(3) 位姿约束可以结合到 SLAM/优化框架中。因此，本目标格式不仅保存普通 RGB-D 数据，还显式保存 camera、pose、pair/multiview relation、scale、pointmap-ready 信息。

### 1.2 标准目录

```text
target_dataset/
├── dataset_card.json
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── scenes/
│   └── <dataset_name>/<scene_id>/
│       ├── scene_meta.json
│       ├── frames.jsonl
│       ├── cameras.json
│       ├── poses.jsonl
│       ├── pairs.jsonl
│       ├── rgb/<camera_id>/<frame_id>.jpg
│       ├── depth/<camera_id>/<frame_id>.png|.exr|.npy
│       ├── normal/<camera_id>/<frame_id>.exr|.npy
│       ├── semantic/<camera_id>/<frame_id>.png
│       ├── instance/<camera_id>/<frame_id>.png
│       ├── valid_mask/<camera_id>/<frame_id>.png
│       ├── pointmap/<camera_id>/<frame_id>.npy
│       ├── point_clouds/*.ply|*.pcd|*.npz
│       ├── meshes/*.ply|*.obj|*.glb|*.usd|*.usda|*.usdz
│       ├── gs/*.ply|*.splat|*.ksplat|*.usdz
│       ├── annotations/
│       │   ├── objects.json
│       │   ├── bboxes_2d.jsonl
│       │   ├── bboxes_3d.jsonl
│       │   ├── captions.jsonl
│       │   └── scene_graph.json
│       ├── provenance.json
│       └── quality_report.json
└── registry/
    ├── source_datasets.json
    ├── class_mapping.json
    └── adapter_versions.json
```

### 1.3 `frames.jsonl` 单帧记录

```json
{
  "dataset": "nuScenes",
  "scene_id": "scene-0001",
  "frame_id": "000123",
  "timestamp_ns": 1532402927647951,
  "split": "train",
  "camera_id": "CAM_FRONT",
  "rgb_path": "rgb/CAM_FRONT/000123.jpg",
  "width": 1600,
  "height": 900,
  "camera_ref": "CAM_FRONT_v0",
  "pose_ref": "000123_CAM_FRONT",
  "depth_path": null,
  "depth_unit": null,
  "semantic_path": null,
  "instance_path": null,
  "valid_mask_path": "valid_mask/CAM_FRONT/000123.png",
  "text": null,
  "source": {
    "raw_path": "samples/CAM_FRONT/n008-2018-...jpg",
    "license": "source_dataset_terms",
    "conversion_adapter": "nuscenes@0.1"
  }
}
```

### 1.4 `cameras.json` 相机记录

统一使用 OpenCV pinhole/fisheye 描述。原始数据是 OpenGL/Blender/Nerfstudio 坐标时，必须在 `provenance.json` 中记录转换矩阵。

```json
{
  "CAM_FRONT_v0": {
    "model": "OPENCV",
    "width": 1600,
    "height": 900,
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "distortion": [k1, k2, p1, p2, k3],
    "raw_model": "PINHOLE|OPENCV|OPENCV_FISHEYE|FTHETA|EQUIRECTANGULAR",
    "raw_intrinsics": {},
    "undistorted": false
  }
}
```

### 1.5 `poses.jsonl` 位姿记录

统一保存 camera-to-world：

```json
{
  "pose_id": "000123_CAM_FRONT",
  "frame_id": "000123",
  "camera_id": "CAM_FRONT",
  "T_c2w": [[...], [...], [...], [0,0,0,1]],
  "T_w2c": [[...], [...], [...], [0,0,0,1]],
  "coord_system": "opencv_camera_to_metric_world",
  "scale": 1.0,
  "pose_source": "gt|slam|colmap|arkit|simulator|rendered",
  "confidence": 1.0
}
```

约定：OpenCV camera frame 为 `x right, y down, z forward`；world frame 尽量保留数据集原始 metric world，再在 `provenance.json` 写明 `T_raw_to_target_world`。若原始 pose 是 world-to-camera，必须显式求逆并记录。

### 1.6 `pairs.jsonl` 图像对/多视图关系

用于 MASt3R/DUSt3R 风格 pair training，也用于 VGGT/Pi3 风格 multiview chunk training。

```json
{
  "pair_id": "000123_CAM_FRONT__000124_CAM_FRONT",
  "target_frame": "000123_CAM_FRONT",
  "source_frames": ["000124_CAM_FRONT"],
  "relative_pose": {
    "T_target_to_source": [[...]],
    "baseline_m": 0.37,
    "view_angle_deg": 8.2
  },
  "overlap": {
    "method": "depth_projection|pointcloud_visibility|pose_heuristic|colmap_tracks",
    "score": 0.68
  },
  "pair_type": "temporal|stereo|cross_time_loop|rendered_multiview|object_turntable",
  "valid": true
}
```

### 1.7 深度、点云、语义和 GS 规范

深度统一为米。若原始是毫米 PNG、inverse depth、ray distance、z-buffer 或 disparity，必须转换并记录：

```json
{
  "depth_encoding": "z_depth_m|ray_distance_m|inverse_depth|disparity|uint16_mm",
  "raw_unit": "mm|m|unknown",
  "invalid_values": [0, -1, 65535],
  "converted_to": "z_depth_m"
}
```

点云统一为 metric world 坐标；点云字段可包含 `xyz, rgb, normal, semantic_id, instance_id`。3DGS 原始 `.ply/.splat/.usdz` 不强制重写，只需要复制到 `gs/` 并记录 reader/exporter。语义类别统一映射到 `class_mapping.json`，同时保留原始 label id。

---

## 2. Agent 总体架构

### 2.1 Agent 角色

```text
DatasetPlannerAgent
  └─ 解析用户目标、数据集名称、目标任务、优先级、是否需要渲染

SourceResearchAgent
  └─ 查官方说明、论文、README、license，生成 source_profile

DatasetInspectorAgent
  └─ 扫描本地目录，识别文件类型、目录结构、样本数量、pose/depth/mask 编码

AdapterSelectorAgent
  └─ 根据 signature 选择已有 adapter；若无则调用 AdapterWriterAgent

AdapterWriterAgent
  └─ 生成或修改 Python DatasetAdapter，包含 detect/inspect/iter/convert/validate

ConversionAgent
  └─ 执行 dry-run、抽样转换、全量转换、断点续跑

GeometryValidationAgent
  └─ 检查 K、T、depth、点云、mesh、mask、pair overlap、坐标系一致性

RenderAgent
  └─ 对 mesh/USD/GLB/3DGS-only 数据集执行相机采样、渲染 RGB-D/normal/segmentation

ReportAgent
  └─ 生成 dataset_card、inspection_report、quality_report、转换日志
```

### 2.2 Adapter 抽象接口

```python
class DatasetAdapter:
    dataset_id: str
    dataset_name: str
    version: str
    priority: str

    def detect(self, root: Path) -> tuple[float, dict]:
        """返回匹配置信度与证据，如关键文件、目录名、metadata 表。"""

    def inspect(self, root: Path, sample_limit: int = 5) -> DatasetProfile:
        """读取样本，确认真实格式、单位、坐标系、数量、可用模态。"""

    def iter_scenes(self, root: Path, split: str | None = None):
        """逐场景枚举。"""

    def iter_frames(self, scene):
        """逐帧枚举，返回 RGB、depth、pose、camera、mask 等 lazy record。"""

    def build_pairs(self, scene, policy: PairPolicy):
        """构建 temporal/stereo/loop/random multiview pairs。"""

    def convert(self, root: Path, out: Path, config: ConvertConfig):
        """执行转换。必须支持 dry_run、resume、sample_only。"""

    def validate(self, out_scene: Path) -> QualityReport:
        """检查输出完整性与几何一致性。"""
```

### 2.3 识别 signature

每个 adapter 需要定义：

```yaml
signature:
  required_any:
    - "metadata.sqlite"
    - "scene*/scene*.sens"
    - "samples/CAM_FRONT"
    - "transforms.json"
  required_all:
    - "cameras.json|cameras.txt|camera_intrinsics.txt"
  forbidden:
    - ".git"
  sample_files:
    rgb: ["*.jpg", "*.png", "*.mp4", "*.mkv"]
    depth: ["*depth*.png", "*.pfm", "*.h5", "*.npy", "*.exr"]
    pose: ["*.traj", "images.txt", "transforms.json", "ego_pose.json", "pose_intrinsic_imu.json"]
```

---

## 3. 数据集格式调研与适配策略

下面按用户提供的数据清单逐项说明。字段含义：

- 原始内容：数据集直接提供的模态。
- 格式/目录：公开说明中可确认的组织方式。
- 目标转换：如何进入本 Skill 的目标格式。
- 风险点：需要下载样本或二次确认的问题。

Metadata 中的模态来源层级统一按以下含义使用：

- 原生：数据集已经直接提供该模态或字段，adapter 读取后即可使用，但仍需做格式、单位、坐标系归一。
- 渲染生成：数据集未直接提供该模态，但提供了 mesh、CAD、USD、GLB、URDF、3DGS 或场景配置等资产，可通过渲染器生成 RGB、depth、semantic、mask 等观测。
- 采样生成：数据集没有固定观测序列或相机轨迹，需要由适配流程定义采样策略，例如相机位姿、视角数量、导航轨迹、frame pair 或物体观察角度；采样结果通常再用于渲染。
- 派生：数据集未直接给出目标模态，但可由已有真实传感器或标注计算得到，例如 LiDAR 投影生成稀疏 depth、RGB-D 融合生成点云、mesh 采样生成点云。
- 估计/pseudo：数据集没有对应真值，只能通过 SfM、SLAM、单目深度、分割模型或其他算法估计；导出时必须标记为伪标签，不能当作 ground truth。
- 无：数据集不提供该模态，也没有足够可靠的资产或传感器信息生成它。

### 113. 3D-FRONT

官方/主要参考：
- https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset
- https://arxiv.org/abs/2011.09127
- https://dlr-rm.github.io/BlenderProc/examples/datasets/front_3d/README.html

原始内容：合成室内家居场景。核心是房屋/房间布局 JSON，家具资产来自 3D-FUTURE，纹理来自 3D-FRONT-texture。论文说明该数据集包含大规模 furnished rooms、布局语义和高质量带纹理家具模型。

Metadata：profile=rendered_indoor_scene；domain=室内合成场景；modalities=RGB:渲染生成, depth:渲染生成, sem2d:渲染生成, pose:采样生成, pointcloud:mesh采样, pc_sem:实例映射, text:无, gs:无；geometry=CAD/mesh原生；convention=Blender/asset坐标需确认；access=public/需确认；risk=JSON到3D-FUTURE资产映射、单位和材质缺失。

格式/目录：常见下载包包括 `3D-FRONT/`、`3D-FUTURE-model/`、`3D-FRONT-texture/`。`3D-FRONT` 内每个 JSON 表示一个 house/flat；渲染时需要同时传入 JSON、3D-FUTURE 模型路径和 texture 路径。

目标转换：这是 mesh/layout-first 数据集，不直接提供 RGB/depth/pose。需要 RenderAgent 在 BlenderProc/Blender 中加载 house JSON，采样相机，渲染 RGB、depth、normal、semantic/instance mask，再写入目标格式。家具 CAD、房间 polygon、object transform 进入 `meshes/` 与 `annotations/objects.json`。

风险点：需要确认每个 JSON 的坐标单位、家具 id 到 3D-FUTURE 模型的映射、墙/地/天花板纹理缺失情况。直接用于前馈数据集时优先构建合成 RGB-D + pose，而不是直接训练 mesh。

建议：构建前馈数据集：是。优先级：中高。

### 114. Aria Synthetic Environments

官方/主要参考：
- https://www.projectaria.com/datasets/ase/
- https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_data_format

原始内容：合成室内场景，提供 RGB、depth、instance segmentation、trajectory、scene language、semidense points/observations、object-to-class mapping。

Metadata：profile=rgbd_sequence；domain=室内合成场景；modalities=RGB:原生, depth:ray_distance_mm原生, sem2d:instance原生, pose:trajectory原生, pointcloud:semidense原生, pc_sem:弱映射, text:scene_language原生, gs:无；geometry=无；convention=fisheye + ray depth；access=public；risk=不能把fisheye/ray depth当作pinhole/z-depth。

格式/目录：每个 scene 目录通常包含：

```text
<scene>/
├── rgb/*.jpg
├── depth/*.png
├── instances/*.png
├── ase_scene_language.txt
├── trajectory.txt
├── semidense_points.csv.gz
├── semidense_observations.csv.gz
└── object_instances_to_classes.json
```

RGB 为 fisheye JPEG，通常 10 FPS；depth 为 16-bit PNG，单位为毫米，且定义为沿相机 ray 的距离；instance mask 为 16-bit PNG，像素值表示 object instance id。`trajectory.txt` 保存 10 FPS 相机轨迹。

目标转换：RGB 复制到 `rgb/`；depth 需从 ray distance 转换或至少标记为 `ray_distance_m`；instance mask 写入 `instance/`；`object_instances_to_classes.json` 写入 class mapping；`trajectory.txt` 转为 `poses.jsonl`；semidense CSV 写入 `point_clouds/semidense_points.ply/npz`。

风险点：fisheye 相机模型必须保留，不能未经处理当作 pinhole。若训练模型只支持 pinhole，需要先去畸变并同步重采样 depth/mask。

建议：构建前馈数据集：是。优先级：中高。

### 115. ARKitScenes

官方/主要参考：
- https://github.com/apple/ARKitScenes
- https://github.com/apple/ARKitScenes/blob/main/DATA.md

原始内容：真实室内 RGB-D 扫描。包括低/高分辨率 RGB、低/高分辨率 depth、confidence、ARKit pose、mesh/point cloud、3D object annotations 等，具体取决于 raw、3dod、upsampling 子集。

Metadata：profile=indoor_rgbd_scan；domain=真实室内扫描；modalities=RGB:原生, depth:原生, sem2d:有限/子集, pose:原生, pointcloud:原生, pc_sem:3D标注子集, text:无, gs:无；geometry=mesh原生；convention=ARKit pose/单位需样本确认；access=public；risk=子集字段差异、pose缺失、confidence过滤。

格式/目录：官方 `DATA.md` 说明常见文件包括 `.png` RGB/depth/confidence、`.pincam` 相机内参、`.json` 3D annotation、`.traj` timestamp + axis-angle + translation、`.ply` mesh/point cloud、`.mov` 视频、`_pose.txt` 逐帧 pose。

目标转换：选择 `raw` 或 `upsampling` 子集时，抽取 RGB/depth/confidence/pose/intrinsics；选择 `3dod` 子集时，额外抽取 3D bounding boxes 和 scene mesh。`.traj` 或 `_pose.txt` 转换为 `T_c2w` 前必须确认方向。depth 通常为 metric depth，但需用样本确认单位与尺度。

风险点：ARKitScenes 存在多个下载资产组，字段不完全一致；部分帧 pose 可能缺失或无效。必须做 frame-level join 和 confidence mask 过滤。

建议：构建前馈数据集：是。优先级：中高。

### 116. BDD100K

官方/主要参考：
- https://bair.berkeley.edu/blog/2018/05/30/bdd/
- https://github.com/bdd100k/bdd100k

原始内容：真实自动驾驶视频/图像数据。主要提供 RGB、检测框、车道线、可行驶区域、语义/实例/panoptic 分割等 2D 标注。通常不提供深度、相机位姿和点云。

Metadata：profile=driving_2d_perception；domain=真实道路驾驶；modalities=RGB:原生, depth:无, sem2d:原生, pose:无, pointcloud:无, pc_sem:无, text:无, gs:无；geometry=无；convention=2D image task schema；access=public/需注册确认；risk=不能伪装几何真值，视频和标注同步需确认。

格式/目录：常见组织为：

```text
bdd100k/
├── videos/
├── images/
└── labels/
```

labels 多为 JSON；`images/100k` 通常是从视频抽取的关键帧；不同任务有不同 label schema，例如 detection、lane、drivable map、segmentation。

目标转换：只可直接构建 RGB + 2D semantics/instances 的前馈弱监督数据。若目标必须包含 pose/depth，需要额外用 SfM/SLAM/单目深度估计生成伪标签。BDD100K 更适合做开放道路场景的外观/语义预训练，而非几何强监督。

风险点：视频帧与标注帧的同步关系需确认。无 GT pose/depth，不能伪装为几何真值。

建议：构建前馈数据集：是，但属于 RGB/2D 标注型。优先级：高。

### 117. BlendedMVS / BlendMVS

官方/主要参考：
- https://github.com/YoYo000/BlendedMVS

原始内容：MVS/NVS 训练数据，包含多视图 RGB、渲染/融合深度图、相机参数、pair 信息，场景涵盖建筑、雕塑、小物体等。

Metadata：profile=multiview_mvs；domain=多视图重建场景；modalities=RGB:原生, depth:pfm原生, sem2d:无, pose:相机参数原生, pointcloud:MVS可重建, pc_sem:无, text:无, gs:无；geometry=相机+深度/pair；convention=MVSNet world-to-camera常见；access=public；risk=外参方向和depth range解析。

格式/目录：常见结构：

```text
BlendedMVS/
├── training_list.txt
├── validation_list.txt
└── <scene_id>/
    ├── blended_images/*.jpg
    ├── cams/pair.txt
    ├── cams/*_cam.txt
    └── rendered_depth_maps/*.pfm
```

`pair.txt` 给出参考视图与源视图候选；`*_cam.txt` 保存外参、内参、depth range；depth map 常为 `.pfm`。

目标转换：这是最适合直接转换成前馈多视图几何格式的数据集之一。读取 `pair.txt` 生成 `pairs.jsonl`；读取 `*_cam.txt` 生成 `cameras.json` 与 `poses.jsonl`；`.pfm` depth 转为 `.npy/.exr` 或保留并标注；RGB 转到 `rgb/`。

风险点：需确认 `*_cam.txt` 中外参是 world-to-camera 还是 camera-to-world，MVSNet 系列通常使用 extrinsic world-to-camera。转换时要统一求逆。

建议：构建前馈数据集：是。优先级：高。

### 118. CO3D v1/v2

官方/主要参考：
- https://github.com/facebookresearch/co3d/tree/v1
- https://github.com/facebookresearch/co3d

原始内容：真实 object-centric 多视图数据，提供 RGB、foreground masks、depth/depth masks、camera poses、point cloud、category/sequence annotations。v2 相比 v1 有更多 sequences/frames 和更好的 mask。

Metadata：profile=object_multiview_sequence；domain=真实物体中心序列；modalities=RGB:原生, depth:原生, sem2d:foreground mask, pose:原生, pointcloud:原生, pc_sem:类别级弱语义, text:类别元数据, gs:无；geometry=point cloud原生；convention=PyTorch3D/CO3D camera需归一；access=public；risk=v1/v2结构差异和mask/depth有效性。

格式/目录：典型 v2 结构：

```text
<category>/
└── <sequence>/
    ├── images/
    ├── masks/
    ├── depths/
    ├── depth_masks/
    ├── pointcloud.ply
    ├── frame_annotations.jgz
    └── sequence_annotations.jgz
set_lists/*.json
```

`frame_annotations.jgz` 与 `sequence_annotations.jgz` 是 gzip JSON，保存 dataclass 风格的帧/序列标注，包括 camera、image、depth、mask、point cloud 等索引。

目标转换：按 category/sequence 映射为 scene；frame annotations 转 `frames.jsonl/cameras.json/poses.jsonl`；mask 写 `instance/valid_mask`；depth 写 `depth/`；pointcloud 写 `point_clouds/pointcloud.ply`。object-centric 数据默认以物体为场景中心，适合 pair/multiview 训练。

风险点：CO3D 坐标系、NDC/pytorch3d camera convention 与 OpenCV 不同，需要专门转换。不要直接把 PyTorch3D camera matrix 当作 OpenCV K/R/T。

建议：构建前馈数据集：是。优先级：中高。

### 119. DL3DV-10K

官方/主要参考：
- https://dl3dv-10k.github.io/DL3DV-10K/
- https://huggingface.co/datasets/DL3DV/DL3DV-10K

原始内容：大规模真实视频数据，面向 NVS/3D reconstruction。提供 4K 视频、COLMAP 标定结果、稀疏点云、downsampled images、`transforms.json` 等。

Metadata：profile=scene_multiview_recon；domain=真实多视图场景；modalities=RGB:原生, depth:重建派生/非GT, sem2d:scene label非像素级, pose:原生, pointcloud:COLMAP可导出, pc_sem:无, text:human scene labels, gs:原生/benchmark；geometry=Nerfstudio/3DGS/COLMAP；convention=Nerfstudio/OpenGL与COLMAP需区分；access=public/需确认；risk=视频抽帧、相机约定和重建产物对齐。

格式/目录：常见样例结构：

```text
<scene>/
├── video.mp4
├── images/
├── images_2/
├── images_4/
├── images_8/
├── transforms.json
└── colmap/
    ├── database.db
    ├── features.h5
    ├── matches.h5
    ├── pairs-netvlad.txt
    ├── global-feats-netvlad.h5
    └── sparse/0/{cameras.bin, images.bin, points3D.bin}
```

目标转换：优先使用 `transforms.json` 或 COLMAP `cameras/images/points3D` 生成 camera/pose/point cloud；`pairs-netvlad.txt` 可转为候选 pair；RGB 使用 downsampled images 控制训练成本。若 `transforms.json` 是 Nerfstudio/OpenGL convention，需要转换到 OpenCV camera。

风险点：不同发布分片可能结构略有差异；视频抽帧与 COLMAP images 必须按文件名对齐。

建议：构建前馈数据集：是。优先级：高。

### 120. HM3D

官方/主要参考：
- https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d
- https://matterport.com/habitat-matterport-3d-research-dataset

原始内容：真实室内/建筑级 textured 3D mesh，常用于 Habitat。包含 `.glb/.obj` mesh、textures、语义版本/语义映射（取决于下载版本）。不直接提供 RGB-D 轨迹帧。

Metadata：profile=rendered_indoor_mesh；domain=真实室内mesh资产；modalities=RGB:渲染生成, depth:渲染生成, sem2d:语义版可渲染, pose:采样生成, pointcloud:mesh采样, pc_sem:语义版映射, text:无, gs:无；geometry=mesh原生；convention=Habitat坐标/语义版本需确认；access=agreement_required/token；risk=下载门槛、语义版差异和相机采样有效性。

格式/目录：通常以 scene 为单位存储 mesh asset，例如 `.glb` 或 `.obj + .mtl + textures`；HM3D Semantics 版本提供带实例/类别颜色或语义映射的 mesh。

目标转换：作为 mesh-first 数据集，需要 RenderAgent 在 Habitat-Sim 中采样导航轨迹或相机位姿，渲染 RGB/depth/semantic，生成 `frames.jsonl` 和 `pairs.jsonl`。原始 mesh 存入 `meshes/`。

风险点：下载权限、版本差异、语义版本和非语义版本字段不同。相机采样应避免穿墙、过近、视野无效。

建议：构建前馈数据集：是，但需渲染。优先级：中。

### 121. Hypersim

官方/主要参考：
- https://github.com/apple/ml-hypersim

原始内容：大规模 photorealistic synthetic indoor 数据，提供 RGB/HDR、depth、surface normal、3D position、semantic、semantic instance、mesh/object metadata、camera trajectory、lighting/material 信息。

Metadata：profile=indoor_synthetic_fullmodal；domain=合成室内场景；modalities=RGB:原生, depth:meters原生, sem2d:原生, pose:原生, pointcloud:position/depth生成, pc_sem:语义对齐, text:无, gs:无；geometry=mesh/object metadata；convention=asset units到meters + camera convention；access=public；risk=HDF5体量、官方过滤列表和单位换算。

格式/目录：典型 scene 包含 `_detail/` 和 `images/`。`_detail` 中有 `metadata_scene.csv`、camera orientation/position hdf5、mesh/object metadata、bounding boxes 等；`images` 中包含 `scene_cam_##_final_hdf5`、`geometry_hdf5`、`semantic_hdf5` 等，字段如 `depth_meters.hdf5`、`normal_cam.hdf5`、`position.hdf5`、`semantic.hdf5`、`semantic_instance.hdf5`。

目标转换：直接抽取 RGB、depth_meters、normal、semantic、semantic_instance、camera trajectory。由于 Hypersim 的 depth/camera convention 有明确说明，转换时必须保留原始 convention 并统一到 OpenCV target。object metadata/3D bbox 写入 annotations。

风险点：HDF5 文件体量大，读取时必须 lazy loading；某些场景/图像被官方排除，需要使用官方 split/metadata 过滤。

建议：构建前馈数据集：是。优先级：中。

### 122. Matterport3D

官方/主要参考：
- https://niessner.github.io/Matterport/
- https://github.com/niessner/Matterport

原始内容：真实建筑级 RGB-D panorama/mesh 数据，包含大量 RGB-D 图像、camera poses、surface reconstructions、2D/3D semantic annotations、region/object annotations。

Metadata：profile=indoor_rgbd_mesh；domain=真实室内建筑；modalities=RGB:原生, depth:原生, sem2d:原生/可导出, pose:原生, pointcloud:RGB-D/mesh生成, pc_sem:region/object语义, text:无, gs:无；geometry=mesh原生；convention=panorama/local view/mesh坐标需统一；access=agreement_required；risk=许可、全景相机模型和多坐标系对齐。

格式/目录：常见数据包括 color/depth images、intrinsics、poses、textured meshes、floor plans、house_segmentations、region/object semantics。图像命名通常包含 scan/region/panorama/view 等字段；mesh `.ply` 可包含 material/segment/category 等属性；`.fsegs.json` 和 `.semseg.json` 存储面片分割和语义聚合。

目标转换：可直接从 RGB-D + pose 构建前馈训练数据，也可从 mesh 渲染补充视角。全景图可拆分成 perspective views，也可保留 equirectangular camera model；mesh/region/object semantics 写入 annotations 与 scene_graph。

风险点：数据下载需要 license；全景坐标、局部视图相机和 mesh 坐标之间的转换复杂。不要把 panorama 当普通 pinhole 图像。

建议：构建前馈数据集：是。优先级：中。

### 123. MegaDepth

官方/主要参考：
- https://www.cs.cornell.edu/projects/megadepth/

原始内容：从互联网照片集合通过 SfM/MVS 生成的大规模单目深度数据。主要用于单目深度和几何学习，场景多为室外地标/建筑。

Metadata：profile=internet_photo_depth_sfm；domain=室外地标/建筑；modalities=RGB:原生, depth:预处理原生, sem2d:无, pose:SfM/预处理, pointcloud:SfM稀疏点, pc_sem:无, text:无, gs:无；geometry=COLMAP/SfM；convention=COLMAP/OpenCV常见但来源需记录；access=public/镜像差异；risk=版本分散、pose/depth是否官方GT不稳定。

格式/目录：官方页面说明其由 Internet photo collections 经 SfM/MVS 生成。实际使用中常见预处理包会包含 undistorted images、depth maps、camera intrinsics/extrinsics 或 COLMAP/SfM 输出，但不同镜像/二次处理版本差异较大。

目标转换：若有官方/预处理 pose 与 depth，可转换为 RGB + depth + camera + pose；若只有图像和 depth，需标注 pose unavailable；若使用 COLMAP 输出，按照 COLMAP adapter 处理。

风险点：版本分散，文件结构不统一。必须在 `inspection_report.json` 中记录具体来源。若 pose 来自第三方预处理，不应标为官方 GT。

建议：构建前馈数据集：是，但需样本确认。优先级：中。

### 124. MVImgNet

官方/主要参考：
- https://gaplab.cuhk.edu.cn/projects/MVImgNet/
- https://github.com/GAP-LAB-CUHK-SZ/MVImgNet

原始内容：object-centric 多视图真实视频/图像数据，覆盖大量物体类别。提供 RGB、object masks、camera parameters、point clouds 等标注。

Metadata：profile=object_multiview_colmap；domain=真实物体多视图；modalities=RGB:原生, depth:无/可估计, sem2d:mask子集, pose:COLMAP原生, pointcloud:稀疏点原生, pc_sem:类别/实例级, text:类别标签, gs:无；geometry=COLMAP稀疏模型；convention=COLMAP/OpenCV；access=gated/form；risk=下载门槛、子集不一致和mask可用性。

格式/目录：官方仓库说明 MVImgNet 包含数百万帧、数十万视频和数百类别；完整数据通常按 category 分包下载。具体目录按 category/object/video 组织，标注包含 masks、camera parameters、point clouds。

目标转换：按 object instance 作为 scene；RGB 与 masks 写入 frame；camera/pose 写入 `cameras/poses`；point cloud 写入 `point_clouds`。适合构建 object-level multiview 前馈数据。

风险点：大规模压缩包，解压和索引耗时。mask/camera/point cloud 的具体文件名需样本确认。

建议：构建前馈数据集：是。优先级：中高。

### 125. MVImgNet 2.0

官方/主要参考：
- https://luyues.github.io/mvimgnet2/

原始内容：MVImgNet 的扩展版本，包含更多 object instances/categories，并提供 segment masks、SfM poses、dense point clouds 等。

Metadata：profile=object_multiview_colmap_dense；domain=真实物体多视图；modalities=RGB:原生, depth:无/可估计, sem2d:mask改进, pose:SfM原生, pointcloud:dense原生, pc_sem:类别/实例级, text:类别标签, gs:无；geometry=SfM/点云；convention=SfM/OpenCV需确认；access=gated/需确认；risk=与MVImgNet版本差异和dense点云尺度。

格式/目录：以 object/category 为主组织，提供 360-degree views、masks、camera parameters/SfM poses、dense point clouds。具体目录需下载样本确认。

目标转换：同 MVImgNet，但优先使用更高质量的 masks/poses/dense point clouds。可直接生成 object-level multiview chunks。

风险点：官方页面与数据下载可能分批发布，具体文件命名和可下载字段需要验证。

建议：构建前馈数据集：是。优先级：中高。

### 126. nuScenes

官方/主要参考：
- https://www.nuscenes.org/nuscenes
- https://github.com/nutonomy/nuscenes-devkit

原始内容：真实自动驾驶多传感器数据。提供 6 cameras、LiDAR、radar、ego pose、calibrated sensor、3D boxes、map、lidarseg/panoptic 等。

Metadata：profile=driving_multisensor_sequence；domain=真实城市驾驶；modalities=RGB:原生, depth:LiDAR投影派生, sem2d:2D框/分割可用, pose:ego/sensor原生, pointcloud:LiDAR/Radar原生, pc_sem:lidarseg/panoptic可选, text:scene metadata, gs:无；geometry=无；convention=global/ego/sensor链路；access=public/需注册；risk=token关系表、时间同步和坐标链路。

格式/目录：nuScenes 是 token-based relational schema。核心表包括 `scene`、`sample`、`sample_data`、`sample_annotation`、`sensor`、`calibrated_sensor`、`ego_pose`、`map`、`lidarseg` 等。camera image 和 lidar/radar point cloud 通过 `sample_data.filename` 指向文件；`calibrated_sensor` 保存传感器外参和 camera intrinsic；`ego_pose` 保存 ego-to-global pose。

目标转换：对每个 keyframe sample，组合：

```text
T_camera_to_global = T_ego_to_global @ T_camera_to_ego
```

写入 `poses.jsonl`。RGB 来自 camera sample_data；LiDAR 点云投到 global/world 后写 `point_clouds`；3D boxes 写 `bboxes_3d.jsonl`；map mask/semantic 写 `annotations/map.json`；lidarseg 写点云语义。

风险点：nuScenes 标注 keyframe 为 2Hz，但原始 sample_data 可能更高频；多 camera 同步和非 keyframe处理要明确。相机 pose 是通过 ego pose 与 calibrated_sensor 间接得到。

建议：构建前馈数据集：是。优先级：高。

### 127. Objaverse-XL

官方/主要参考：
- https://objaverse.allenai.org/
- https://github.com/allenai/objaverse-xl

原始内容：超大规模 3D object asset 数据集，包含 10M+ 3D objects。对象来自多源，许可证逐对象不同。通常通过 Python API 下载 metadata 和 object assets。

Metadata：profile=asset_bank；domain=物体资产；modalities=RGB:渲染生成, depth:渲染生成, sem2d:asset id渲染, pose:采样生成, pointcloud:mesh采样, pc_sem:metadata弱映射, text:metadata/tags, gs:无；geometry=mesh多格式原生；convention=asset坐标/尺度需归一；access=mixed_license；risk=逐对象license、可渲染性和格式清洗。

格式/目录：通过 `objaverse` API 或 Hugging Face 访问。对象文件可能为 glb/gltf/obj/fbx/usd 等多种格式，metadata 提供 uid、source、license、tags/captions 等。仓库提供 rendering scripts。

目标转换：这是 asset-first 数据集。需要先筛选可渲染、可归一化、license 允许的对象，再用 Blender/Trimesh 渲染多视角 RGB-D/normal/mask，构建 object-level 前馈数据。原始 asset 复制到 `meshes/`，metadata 写入 captions/tags。

风险点：质量参差、尺度/朝向/材质/动画复杂，逐对象 license 不一致。必须执行 mesh repair、scale normalization、empty render filtering。

建议：构建前馈数据集：是，但需渲染与质量过滤。优先级：中。

### 128. OpenVid-1M

官方/主要参考：
- https://huggingface.co/datasets/nkp37/OpenVid-1M

原始内容：大规模 text-video 数据，提供视频文件及 CSV/JSON 描述，包含 caption/metadata。无官方深度、相机位姿、点云。

Metadata：profile=video_text_pretrain；domain=开放域视频；modalities=RGB:视频帧原生, depth:无, sem2d:无, pose:无, pointcloud:无, pc_sem:无, text:caption原生, gs:无；geometry=无；convention=video fps/resolution schema；access=public/需确认；risk=仅能作为弱监督/预训练，几何标签需标注为pseudo。

格式/目录：Hugging Face 数据集包含 `OpenVid-1M.csv`、`OpenVidHD.csv`、`OpenVidHD.json` 与多个视频 zip 分片。CSV 可用 pandas 读取，视频分辨率至少 512×512，OpenVidHD 包含较多 1080p 视频。

目标转换：只能直接构建 RGB/video + text 数据。若用于前馈三维，需要 VideoFrameExtractor 抽帧，再调用 SfM/SLAM/DepthEstimator 生成伪 pose/depth，并标注为 pseudo labels。

风险点：视频版权/许可、镜头运动不足、动态物体、文本与帧级内容未必严格对应。不能作为几何真值数据。

建议：构建前馈数据集：可选，主要做文本/视频预训练。优先级：低到中。

### 129. Replica

官方/主要参考：
- https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#replica-dataset
- https://github.com/facebookresearch/Replica-Dataset

原始内容：高质量室内空间重建，提供 clean dense geometry、HDR textures、semantic class/instance、planar segmentation、Habitat export。

Metadata：profile=rendered_semantic_mesh；domain=真实室内重建；modalities=RGB:渲染生成, depth:渲染生成, sem2d:语义mesh渲染, pose:采样生成, pointcloud:mesh采样, pc_sem:原生语义映射, text:无, gs:无；geometry=mesh原生；convention=Habitat/mesh坐标；access=public；risk=渲染采样、语义映射和Habitat导出版本。

格式/目录：每个场景常见资产包括 `mesh.ply`、`habitat/mesh_semantic.ply`、`info_semantic.json`、`semantic.bin/json`、textures、navmesh 等。Habitat 目录可直接用于模拟渲染。

目标转换：作为 mesh-first/semantic mesh 数据集，使用 Habitat-Sim 采样相机位姿并渲染 RGB/depth/semantic/instance。mesh 与 semantic metadata 写入 `meshes/` 与 annotations。

风险点：原始 Replica 不一定提供现成 RGB-D trajectory；需要渲染。语义 id 与类别映射应从 `info_semantic.json` 读取。

建议：构建前馈数据集：是。优先级：中高。

### 130. ReplicaCAD

官方/主要参考：
- https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#replicacad

原始内容：基于 Replica FRL apartment 的可交互 CAD/仿真室内场景，包含 static background、object assets、URDF/physical properties、receptacle metadata、scene configs、navmesh。

Metadata：profile=interactive_sim_scene；domain=室内交互仿真；modalities=RGB:渲染生成, depth:渲染生成, sem2d:渲染生成, pose:采样生成, pointcloud:mesh采样, pc_sem:object metadata映射, text:配置元数据, gs:无；geometry=GLB/URDF/navmesh原生；convention=Habitat scene dataset config；access=public；risk=刚体/关节体/navmesh/receptacle元数据不能丢。

格式/目录：常见组织为 Habitat scene dataset config + stage/object asset configs + navmesh + articulated/rigid object assets。目标是 embodied AI/rearrangement，不是原始 RGB-D 扫描。

目标转换：使用 Habitat-Sim/Isaac/Habitat renderer 从 scene config 渲染 RGB-D/semantic，并可保留物体物理属性、可交互关系到 `scene_graph.json`。原始 CAD/URDF/scene config 存入 `meshes/` 或 `annotations/sim_config`。

风险点：交互对象、可动关节、receptacle 与静态语义需要分开编码。训练前馈几何时可先只做静态渲染，具身任务再保留物理字段。

建议：构建前馈数据集：是。优先级：中高。

### 131. ScanNet v2

官方/主要参考：
- https://www.scan-net.org/
- https://github.com/ScanNet/ScanNet

原始内容：真实 RGB-D video scans，包含 2.5M views、1500+ scans、3D camera poses、surface reconstruction、instance-level semantic segmentations。

Metadata：profile=indoor_rgbd_semantic_scan；domain=真实室内扫描；modalities=RGB:原生, depth:原生, sem2d:原生导出, pose:原生, pointcloud:mesh/点云原生, pc_sem:原生, text:无, gs:无；geometry=mesh原生；convention=ScanNet sensor/mesh坐标；access=agreement_required；risk=.sens解包、2D/3D标注对齐和许可。

格式/目录：每个 scan 通常为 `scene%04d_%02d`，包含 `.sens` 传感器流、mesh、pose、color/depth 导出、`*.aggregation.json`、`*.segs.json`、`*_vh_clean*.ply`、`*_vh_clean_2.labels.ply`、2D label/instance 等。`.sens` 内含 color、depth、pose、intrinsic/extrinsic。

目标转换：先用官方脚本或 adapter 解码 `.sens` 为 RGB/depth/pose；mesh/semantic ply 转 `meshes`；aggregation/segs 转 instance/semantic；按时间邻近生成 pairs。

风险点：`.sens` 解码耗时；pose 可能存在 invalid frames；depth 多为 uint16 mm，需要转米；ScanNet 相机坐标与 mesh/world 坐标需要验证。

建议：构建前馈数据集：是。优先级：中。

### 132. StaticThings3D

官方/主要参考：
- https://github.com/lmb-freiburg/robustmvd/blob/master/rmvd/data/README.md#staticthings3d

原始内容：合成/静态物体多视图数据，常用于 MVD/MVS。RobustMVD 转换格式提供 images、poses、intrinsics、depth、invdepth、depth_range 等。

Metadata：profile=synthetic_mvd_regression；domain=合成多视图场景；modalities=RGB:原生/转换后, depth:原生/转换后, sem2d:无, pose:原生/转换后, pointcloud:depth生成, pc_sem:无, text:无, gs:无；geometry=无；convention=RobustMVD schema；access=public/需下载源确认；risk=转换目录结构和无效depth处理。

格式/目录：RobustMVD 统一格式中 sample record 包含：`images` list、`poses` 4×4、`intrinsics`、`keyview_idx`、`depth`、`invdepth`、`depth_range`。depth 单位为米，无效值通常为 0。

目标转换：直接从 RobustMVD format 转为 scene/frame/pair。keyview 与 source views 可对应到 `pairs.jsonl`。depth/invdepth 保留其中一种，优先 depth_m。

风险点：用户表格将其标为 Real，但 StaticThings3D 更常见于 synthetic rendered 数据，需要按具体来源修正 `Real/Synth` 字段。

建议：构建前馈数据集：是。优先级：中。

### 133. uCO3D

官方/主要参考：
- https://github.com/facebookresearch/uco3d

原始内容：object-centric 真实 turntable 视频数据。提供 RGB video、mask video、depth maps、camera poses、point clouds、segmented point clouds、3D Gaussian splats、LVIS 类别、短/长文本描述。

Metadata：profile=object_video_fullmodal；domain=真实物体中心序列；modalities=RGB:视频原生, depth:H5原生, sem2d:mask原生, pose:原生, pointcloud:多级点云原生, pc_sem:segmented point cloud, text:caption原生, gs:原生；geometry=point cloud/GS；convention=uCO3D sqlite/video时间轴；access=public；risk=sqlite索引、视频帧同步和多模态路径解析。

格式/目录：典型结构：

```text
metadata.sqlite
set_lists/*.sqlite
<super_category>/<category>/<sequence>/
├── rgb_video.mp4
├── mask_video.mkv
├── depth_maps.h5
├── point_cloud.ply
├── sparse_point_cloud.ply
├── segmented_point_cloud.ply
└── gaussian_splats/
```

frame-level metadata、camera poses、paths 等存储在 `metadata.sqlite`；split 存储在 `set_lists/*.sqlite`。

目标转换：读取 sqlite 作为主索引；视频解码成 RGB/mask frame；HDF5 读取 depth；pose 转 `T_c2w`；point cloud 与 GS 复制；LVIS/category/caption 写 annotations。适合 object-level multiview/3DGS 训练。

风险点：sqlite schema 需固定版本解析；视频帧、depth_maps.h5 与 metadata frame index 必须严格对齐。数据量大，建议流式转换。

建议：构建前馈数据集：是。优先级：高。

### 134. YCB Benchmark / YCB Object and Model Set

官方/主要参考：
- https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#ycb-benchmarks---object-and-model-set
- https://www.ycbbenchmarks.com/

原始内容：常见物体 object set，提供每个物体多视角 RGB-D/RGB 图像、segmentation masks、camera calibration、texture-mapped 3D mesh。面向机器人抓取/操作。

Metadata：profile=object_asset_rgbd；domain=物体资产；modalities=RGB:扫描原生/渲染生成, depth:扫描原生/渲染生成, sem2d:渲染生成, pose:采样/标定生成, pointcloud:扫描/mesh采样, pc_sem:object id映射, text:object metadata, gs:无；geometry=mesh/GLB原生；convention=object frame/renderer frame需统一；access=public；risk=原始YCB与Habitat版字段不同。

格式/目录：YCB 视频/模型包通常以 object 为单位，包含 RGB/RGB-D capture、calibration、mask、mesh。部分版本每个物体有多个相机/turntable 视角。

目标转换：按 object 构建 scene；RGB-D/camera/mask 直接转 frame；mesh 转 `meshes/`；可从 turntable 顺序生成 pairs。若只下载 object model set，则需自行渲染。

风险点：YCB 有多个子集/镜像，文件结构和命名不同。需确认是否包含 pose，若没有物体/相机位姿则只能渲染或用标定轨迹推导。

建议：构建前馈数据集：是。优先级：中。

### 177. 3D-GloBFP / gloBFPr

官方/主要参考：
- https://github.com/billbillbilly/gloBFPr
- https://zenodo.org/records/10570660

原始内容：全球建筑物 footprint + height 数据。面向遥感/建筑高度提取，不是图像多视图数据。数据通常以 GIS vector/tile 形式提供，包含 building footprint polygon 和 height 属性。

Metadata：profile=geo_polygon_height；domain=遥感地理建筑；modalities=RGB:无, depth:无, sem2d:polygon/raster原生, pose:地理坐标非相机, pointcloud:无, pc_sem:无, text:属性表, gs:无；geometry=3D footprint polygons；convention=CRS/地理坐标；access=public；risk=CRS、tile范围和height单位。

格式/目录：`gloBFPr` 工具用于搜索、下载和处理全球建筑物 footprint tiles with height；常见格式为 shapefile/GeoPackage/GeoJSON 等 GIS 数据，具体取决于下载接口。

目标转换：不直接构建前馈三维重建图像数据。可转换为 `annotations/geospatial_buildings.geojson`，或与卫星影像/DEM 结合生成遥感 2.5D 训练数据。

风险点：坐标系通常是 WGS84/投影坐标，不能直接与相机 SE(3) 混用。高度可能是模型估计值，不是激光/测量真值。

建议：构建前馈数据集：否，除非目标是遥感高度/地图任务。优先级：低。

### 188. ScanNet++

官方/主要参考：
- https://scannetpp.mlsg.cit.tum.de/scannetpp/
- https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation
- https://github.com/scannetpp/scannetpp

原始内容：高保真真实室内场景，提供 laser scans、DSLR images、iPhone RGB-D、mesh、semantic/instance annotations、point clouds、panocam 等。

Metadata：profile=indoor_multisensor_recon；domain=真实室内扫描；modalities=RGB:原生, depth:原生, sem2d:原生/可导出, pose:多约定原生, pointcloud:原生, pc_sem:原生, text:无, gs:无；geometry=mesh/COLMAP/Nerfstudio；convention=COLMAP/OpenCV + Nerfstudio/OpenGL + iPhone轨迹；access=agreement_required/需确认；risk=多相机约定并存、scene graph导出和对齐。

格式/目录：当前官方文档说明数据集约 1006 scenes；目录包括 `split/`、`metadata/`、`data/<scene_id>/scans`、`dslr`、`iphone`、`panocam`。`scans` 下有 `pc_aligned.ply`、`mesh_aligned_0.05_semantic.ply`、`segments.json`、`segments_anno.json`；`dslr/colmap` 有 `cameras.txt/images.txt/points3D.txt`；`dslr/nerfstudio` 有 `transforms.json`；`iphone` 有 `rgb.mkv`、`depth.bin`/depth PNG、`pose_intrinsic_imu.json`。

目标转换：优先使用 DSLR undistorted + COLMAP metric poses 构建高质量 RGB+pose；用 mesh 渲染 high-res depth；iPhone RGB-D 可作为低分辨率 depth source；semantic mesh 可投影到 2D 生成 semantic/instance masks。场景语义图可由 segments/object annotations 构建。

风险点：DSLR fisheye/OpenCV_FISHEYE、Nerfstudio OpenGL convention、iPhone ARKit right-handed +Z forward 三种坐标需分开处理。NVS test split 不含 3D 信息，不能用于深度/语义真值。

建议：构建前馈数据集：是。优先级：高。

### 244. InteriorAgent

官方/主要参考：
- https://huggingface.co/datasets/spatialverse/InteriorAgent

原始内容：高质量 USD/USDa 室内场景资产，面向 NVIDIA Isaac Sim。提供 materials、meshes、lighting、floorplan、room metadata，适用于导航、操作、布局理解。

Metadata：profile=usd_scene_asset；domain=合成室内仿真资产；modalities=RGB:渲染生成, depth:渲染生成, sem2d:metadata渲染, pose:采样生成, pointcloud:资产采样, pc_sem:metadata映射, text:scene description原生, gs:无；geometry=USD原生；convention=USD/Isaac Sim坐标；access=public/需确认；risk=USD材质/物理属性和仿真版本兼容。

格式/目录：每个 scene 目录结构示例：

```text
kujiale_xxxx/
├── Materials/
│   ├── Textures/
│   └── *.mdl
├── Meshes/
├── kujiale_xxxx.usda
├── limpopo_golf_course_4k.hdr
└── rooms.json
```

`rooms.json` 保存 room_type 和 polygon，坐标在 Isaac Sim world frame 中，X forward、Y right、Z up。

目标转换：使用 Isaac Sim/Omniverse RenderAgent 加载 `.usda`，采样 camera，渲染 RGB-D/semantic；`rooms.json` 写 `scene_graph.json` 或 `floorplan.json`；USD 原文件复制到 `meshes/` 或 `sim_assets/`。

风险点：不是现成 RGB-D dataset；必须渲染。Isaac 坐标系与 OpenCV camera/world 需要转换。

建议：构建前馈数据集：是。优先级：中高。

### 245. InteriorGS / SAGE-3D InteriorGS USDZ

官方/主要参考：
- https://github.com/manycore-research/InteriorGS
- https://huggingface.co/datasets/spatialverse/SAGE-3D_InteriorGS_usdz

原始内容：室内 3D Gaussian Splatting 场景，带语义标注和空间占用信息。用户给出的 HF 数据是将 InteriorGS compressed PLY 转换为 USDZ 的版本，面向 Isaac Sim/Omniverse。

Metadata：profile=gaussian_indoor_scene；domain=室内Gaussian场景；modalities=RGB:预览/可渲染, depth:渲染生成, sem2d:labels渲染, pose:采样生成, pointcloud:GS转换可选, pc_sem:object labels映射, text:无, gs:原生；geometry=USDZ/structure metadata；convention=3DGS/USdz坐标需确认；access=gated/需接受条件；risk=GS渲染器、occupancy/labels对齐和访问条件。

格式/目录：HF USDZ 版本结构非常简单：

```text
InteriorGS_usdz/
├── 839873.usdz
├── 839874.usdz
└── ...  # 约 1000 scenes
```

转换管线为 `InteriorGS compressed PLY -> decompressed PLY -> USDZ`，USDZ 使用面向 3D Gaussian rendering 的 USD schema 扩展。

目标转换：若目标支持 GS，直接复制 `.usdz` 到 `gs/` 并记录 metadata；若目标为 RGB-D/pose，则通过 Isaac Sim/3DGRUT renderer 渲染多视角 RGB、depth、semantic/occupancy。GS 本身不等同 mesh，物理碰撞需配合 collision mesh。

风险点：3DGS 渲染与真实 mesh depth/normal 不完全等价；semantic/occupancy 字段是否随 USDZ 一起保留需样本确认。

建议：构建前馈数据集：是。优先级：中高。

### 246. Tabletop_Scenes / TabletopGen-Assets

官方/主要参考：
- https://huggingface.co/datasets/xinjue1/TabletopGen-Assets/tree/main/scene_gallery

原始内容：TabletopGen 生成的预制 3D 桌面场景和机器人 manipulation demo assets，面向文本/单图到可交互 3D tabletop scene。

Metadata：profile=tabletop_asset_scene；domain=桌面操作场景；modalities=RGB:渲染生成, depth:渲染生成, sem2d:渲染生成, pose:采样生成, pointcloud:mesh采样, pc_sem:asset id映射, text:任务/配置元数据, gs:无；geometry=GLB原生；convention=Isaac Sim/GLB坐标；access=public；risk=场景资产与manipulation demo代码分离。

格式/目录：HF 数据集说明：

```text
scene_gallery/        # generated 3D tabletop scenes, .glb
manipulation_demo/    # Isaac Sim pick-and-place demo code/assets
```

目标转换：作为 GLB scene-first 数据集，需要 Blender/Isaac RenderAgent 渲染多视图 RGB-D/normal/mask，并保留 GLB scene 到 `meshes/`。适合构建桌面物体密集布局的前馈数据。

风险点：当前公开资产行数较少，规模有限；语义标注、物理属性、pose 是否内嵌在 GLB 中需样本确认。

建议：构建前馈数据集：是。优先级：中。

### 247. Maya

官方/主要参考：用户表格未提供 URL。

原始内容：按用户描述，是室内外 mesh 场景，包含奇特场景，如宫殿、货船等。

Metadata：profile=custom_dcc_scene；domain=自定义三维场景；modalities=RGB:渲染生成, depth:渲染生成, sem2d:对象ID/材质渲染, pose:采样生成, pointcloud:mesh采样, pc_sem:对象层级映射, text:需人工补充, gs:无/需转换；geometry=DCC/mesh原生；convention=Maya坐标/单位需人工确认；access=unknown；risk=缺官方数据集身份和schema。

格式/目录：无法从官方来源确认。可能是 Maya/Autodesk 格式资产、`.ma/.mb`、或某个内部数据集的名称。

目标转换：如果是 `.ma/.mb`，需要 Maya/Blender/Assimp 可读转换链；如果是 `.fbx/.obj/.glb`，可走通用 MeshAdapter。转换为前馈数据集必须渲染 RGB-D/pose。

风险点：缺少 URL 和样本，不能确定 license、目录结构、材质贴图路径、坐标单位、是否含语义。

建议：构建前馈数据集：是，但需要用户提供来源或样本。优先级：中高（按用户表格）。

### 283. OpenSatMap

官方/主要参考：
- https://opensatmap.github.io/
- https://huggingface.co/datasets/z-hb/OpenSatMap

原始内容：高分辨率卫星图像 + 细粒度 instance-level road structure annotations。覆盖多国多城市，并与 nuScenes/Argoverse2 等自动驾驶区域有对齐关系。

Metadata：profile=geo_vector_map；domain=遥感道路地图；modalities=RGB:卫星图原生, depth:无, sem2d:polyline/attribute原生, pose:地理配准非相机, pointcloud:无, pc_sem:无, text:属性表, gs:无；geometry=vector polylines；convention=tile level/CRS/像素地理映射；access=public；risk=矢量属性、mask栅格化和地图坐标对齐。

格式/目录：官方说明包含 OpenSatMap19 与 OpenSatMap20：level 19 约 0.3 m/pixel，level 20 约 0.15 m/pixel。标注对象包括 lane line、curb、virtual line，并提供八类属性，如颜色、线型、线数、特殊功能、边界、遮挡、清晰度等。标注以 vectorized polylines 表示，同时可能提供 mask。

目标转换：不属于普通相机多视图 3D 数据，但可转为遥感地图任务格式。RGB satellite image 写入 `rgb/overhead`；polyline 写入 `annotations/map_polylines.geojson/json`；mask 写入 `semantic/`。若与 nuScenes/Argoverse 对齐，可建立 global map prior。

风险点：坐标基准、tile origin、像素坐标到地理坐标映射需确认。不能直接生成 camera pose/depth。

建议：构建前馈数据集：否，除非目标是地图/遥感前馈模型。优先级：中。

### 284. SEED-MAP / SatelliteLaneDataset2024

官方/主要参考：
- https://github.com/rilab314/SatelliteLaneDataset2024

原始内容：韩国首尔/仁川卫星道路标注数据，包含 image-label pairs，也提供 COCO form 和 ADE20K form。用户表格说明包含大量车道线和路面符号标注。

Metadata：profile=geo_road_segmentation；domain=遥感道路地图；modalities=RGB:image原生, depth:无, sem2d:COCO/ADE20K原生, pose:地理配准非相机, pointcloud:无, pc_sem:无, text:属性/类别表, gs:无；geometry=shapefile/vector源；convention=NGII shapefile到image alignment；access=public；risk=矢量到栅格转换链路和类别映射。

格式/目录：仓库给出的目录：

```text
datasets/
├── satellite_dataset_250206/
│   ├── image/
│   └── label/
├── satellite_coco_250206/
│   ├── annotations/
│   ├── test2017/
│   ├── train2017/
│   └── val2017/
└── satellite_ade20k_250206/
    ├── annotations/{training,validation}/
    └── images/{training,validation}/
```

目标转换：优先读取 COCO/ADE20K 格式，因为 schema 标准。卫星图写 `rgb/overhead`，label/mask 写 `semantic/`，COCO annotations 写 `annotations/coco.json`。若需要 polyline，需要从原始 label 或 NGII HD map 数据回溯。

风险点：COCO/ADE20K 转换版本可能丢失 lane instance/polyline 几何细节。必须确认原始 label 的精度。

建议：构建前馈数据集：否，除非目标是地图/遥感语义。优先级：中。

### 293. Articraft-10K

官方/主要参考：
- https://articraft3d.github.io/
- https://github.com/mattzh72/articraft
- https://arxiv.org/html/2605.15187v1

原始内容：Articraft 是 agentic articulated 3D asset generation 系统；Articraft-10K 包含 10K+ articulated 3D assets，覆盖日常物体类别。每个资产由程序生成，输出 URDF、3D meshes、semantic parts、articulated joints、joint axes 与 motion ranges。

Metadata：profile=articulated_asset_bank；domain=可动3D物体；modalities=RGB:渲染生成, depth:渲染生成, sem2d:渲染生成, pose:采样生成, pointcloud:URDF/mesh采样, pc_sem:link/joint映射, text:类别/生成元数据, gs:无；geometry=URDF原生；convention=URDF joint/link frame；access=public/需确认；risk=关节限制、collision mesh和材质层级。

格式/目录：仓库采用 code-first records，`data/records/**` 通过 Git LFS 按需 hydrate。每条记录可能包含 `model.py`、metadata、生成结果和可视化资产。执行 record 需要运行 Python 代码。

目标转换：若目标是仿真/具身任务，应保留 URDF、关节、part semantics 到 `annotations/articulation.json`。若目标是前馈三维训练，需要按关节状态采样多个 articulation configurations，并渲染多视图 RGB-D/mask/part segmentation。

风险点：安全性重要：不要执行不可信 `model.py`。必须在 sandbox/container 中运行，禁用网络和危险系统调用。关节状态采样会影响同一 object 的几何一致性。

建议：构建前馈数据集：是。优先级：中高。

### 294. SAGE-10k

官方/主要参考：
- https://huggingface.co/datasets/nvidia/SAGE-10k
- https://github.com/NVlabs/sage
- https://research.nvidia.com/labs/dir/sage/

原始内容：大规模交互式室内场景数据，包含 10,000 diverse scenes、50 room types/styles、565K generated 3D objects。面向 Isaac Sim、embodied AI、physics-based simulation。

Metadata：profile=interactive_indoor_asset；domain=交互式室内场景；modalities=RGB:preview/渲染生成, depth:渲染生成, sem2d:asset id渲染, pose:采样生成, pointcloud:mesh采样, pc_sem:object metadata映射, text:layout metadata, gs:无；geometry=objects/materials/layout原生；convention=Isaac Sim/scene layout坐标；access=public/需确认；risk=assets/materials/layout路径闭环和仿真版本。

格式/目录：官方仓库包含 `client/`、`server/`、`IsaacLab/`、`M2T2/`、`assets/`、`robomimic/` 等；SAGE-10k 数据集位于 Hugging Face。scene 通常是 simulation-ready scene assets/configs。

目标转换：和 InteriorAgent 类似，属于 simulation scene-first 数据集。用 Isaac Sim RenderAgent 渲染 RGB-D/semantic/instance/normal/pose；保留 scene config、object metadata、room/task 信息。若包含机器人动作数据，可写入 `annotations/embodied_tasks.jsonl`。

风险点：数据格式可能依赖 Isaac Sim 版本；大规模渲染成本高。需要明确相机采样策略和物理碰撞过滤。

建议：构建前馈数据集：是。优先级：中高。

### 10146. KITTI

官方/主要参考：
- https://www.cvlibs.net/datasets/kitti/
- https://registry.opendata.aws/kitti/

原始内容：真实自动驾驶数据，包含 stereo cameras、Velodyne LiDAR、GPS/IMU localization。任务覆盖 stereo、optical flow、visual odometry、3D object detection/tracking、road/semantic 等。

Metadata：profile=driving_stereo_lidar_sequence；domain=真实道路驾驶；modalities=RGB:原生, depth:LiDAR投影派生, sem2d:检测/分割任务可用, pose:GPS/IMU原生, pointcloud:Velodyne原生, pc_sem:3D boxes/任务标注, text:无, gs:无；geometry=无；convention=calib/oxts/camera/Velodyne坐标链；access=public；risk=benchmark版本差异、calib解析和LiDAR投影稀疏性。

格式/目录：不同 benchmark 目录不同。Raw/odometry 常见内容包括：

```text
image_00/ image_01/ image_02/ image_03/
velodyne_points/velodyne/
oxts/data/
calib_cam_to_cam.txt
calib_velo_to_cam.txt
calib_imu_to_velo.txt
poses/<seq>.txt   # odometry benchmark
```

目标转换：优先选择 odometry/raw 子集。相机内参从 calib 读取；pose 从 odometry poses 或 OXTS GPS/IMU 推导；LiDAR 转 point cloud/depth projection；stereo pairs 直接生成 `pair_type=stereo`，temporal pairs 按帧邻近生成。

风险点：KITTI 不同任务文件结构差异很大；raw data 的 OXTS pose 需地理坐标转换到局部 ENU/metric world。

建议：构建前馈数据集：是。优先级：高。

### 10147. KITTI-360

官方/主要参考：
- https://www.cvlibs.net/datasets/kitti-360/

原始内容：大规模自动驾驶数据，包含 320K+ images、100K laser scans、73.7 km driving distance、准确地理定位、2D/3D dense semantic & instance annotations。

Metadata：profile=driving_multicamera_mapping；domain=真实道路建图；modalities=RGB:原生, depth:LiDAR/SICK派生, sem2d:原生, pose:原生, pointcloud:原生, pc_sem:3D语义/标注可用, text:无, gs:无；geometry=无；convention=perspective/fisheye/Velodyne/SICK多坐标；access=public/需注册确认；risk=多相机模型、跨帧实例ID和语义格式。

格式/目录：常见目录包括 perspective/fisheye cameras、Velodyne scans、SICK scans、calibration、poses、3D bounding boxes、semantic/instance labels 等。官方说明所有帧准确 geolocalized，语义定义与 Cityscapes 一致，实例 ID 跨帧一致。

目标转换：适合高优先级构建多相机/长序列前馈数据。读取 calibration 和 poses，生成每个 camera 的 `T_c2w`；LiDAR/semantic point cloud 写入 `point_clouds`；2D/3D semantics 写 annotations；temporal、stereo、loop pairs 可全部构建。

风险点：perspective 与 fisheye 相机模型不同；2D/3D 标注跨目录关联复杂。长序列转换必须支持分块和断点续跑。

建议：构建前馈数据集：是。优先级：高。

### 10148. Wayve / WayveScenes101

官方/主要参考：
- https://wayve.ai/science/wayvescenes101/
- https://github.com/wayveai/wayve_scenes

原始内容：真实自动驾驶 NVS/scene reconstruction 数据集，包含 101 scenes，每个 scene 20 秒，5 个 time-synchronised cameras，10 FPS，共约 101,000 images，并提供 camera poses、held-out evaluation camera、scene-level metadata。

Metadata：profile=driving_nvs_sequence；domain=真实道路驾驶；modalities=RGB:多相机原生, depth:无/可估计, sem2d:无, pose:原生, pointcloud:无, pc_sem:无, text:无, gs:可训练生成/非原生；geometry=无；convention=Nerfstudio/NVS camera schema需样本确认；access=public/需确认；risk=无GT几何、仅适合NVS/重建评测。

格式/目录：官方仓库面向 NerfStudio/NVS 使用，包含 high-resolution images、camera poses、metadata、evaluation split。具体下载后的目录需样本确认。

目标转换：直接转为 driving multiview NVS/front-feed 数据。5 相机同步帧可构建 cross-camera pairs；10 FPS temporal frames 可构建 temporal pairs；held-out camera 应进入 test/novel-view split，不参与训练。

风险点：用户表格写 “wayve” 不够明确，此处按 WayveScenes101 处理；若用户实际指 Wayve 其他内部/公开数据，需要重查。pose convention 可能与 Nerfstudio/OpenGL 相关，需转换。

建议：构建前馈数据集：是。优先级：高。

### 10149. Waymo Open Dataset

官方/主要参考：
- https://waymo.com/open/
- https://github.com/waymo-research/waymo-open-dataset

原始内容：真实自动驾驶大规模多传感器数据。官方仓库说明包含 Perception dataset、Motion dataset、End-To-End Driving dataset；Perception 提供高分辨率传感器数据和多任务 labels，Motion 提供 103,354 scenes 的 object trajectories 和 3D maps。

Metadata：profile=driving_multisensor_sequence；domain=真实道路驾驶；modalities=RGB:原生, depth:LiDAR投影派生, sem2d:2D框/panoptic可用, pose:vehicle/sensor原生, pointcloud:LiDAR原生, pc_sem:3D框/分割任务可用, text:无, gs:无；geometry=无；convention=vehicle/global/sensor + TFRecord/Proto；access=public/需注册；risk=Proto读取、组件版本和KITTI-like预转换信息损失。

格式/目录：Waymo 经典版本使用 sharded TFRecord + protocol buffer；v2 系列还有组件化表/列式数据。常见 converter 会先把 Waymo 转成 KITTI-style 或自定义中间格式。

目标转换：Perception 数据可抽取 camera images、LiDAR range image/point cloud、camera/lidar calibration、vehicle pose、3D boxes、segmentation labels。Motion 数据可抽取 agent trajectories 与 maps。构建前馈几何时优先选择 camera + pose + LiDAR projected depth。

风险点：TFRecord/protobuf 解析依赖官方包版本；v1/v2 数据组织差异大。必须锁定 dataset version 和 waymo-open-dataset package version。

建议：构建前馈数据集：是。优先级：高。

---

## 4. 数据集类型归类

### 4.1 可直接构建前馈几何数据的优先组

这些数据集已经提供 RGB + pose，通常还提供 depth/point cloud，可优先实现 adapter：

```text
BlendedMVS, DL3DV-10K, CO3D, uCO3D, ScanNet v2, ScanNet++, ARKitScenes,
ASE, Matterport3D, KITTI, KITTI-360, nuScenes, WayveScenes101, Waymo
```

### 4.2 需要渲染的 mesh/USD/GLB/GS 资产组

这些数据集本身不是帧数据，必须先加载场景/资产，采样相机，再渲染 RGB-D/pose：

```text
3D-FRONT, HM3D, Replica, ReplicaCAD, Objaverse-XL, InteriorAgent,
InteriorGS, Tabletop_Scenes, Maya, Articraft-10K, SAGE-10k, YCB model-only subset
```

### 4.3 只能作为弱监督/伪标签来源的数据组

这些数据集缺少真值 pose/depth，需要 SfM/SLAM/深度估计补充：

```text
BDD100K, OpenVid-1M, 部分 MegaDepth 镜像, 部分 MVImgNet 下载包
```

### 4.4 遥感/地图专用组

这些数据集不是普通透视相机三维重建数据，适合地图构建或遥感语义任务：

```text
OpenSatMap, SEED-MAP, 3D-GloBFP
```

---

## 5. SKILL 实施流程

### 5.1 阶段 0：建立 registry

创建 `registry/source_datasets.yaml`：

```yaml
datasets:
  - id: 117
    name: BlendedMVS
    type: multiview_rgb_depth_pose
    adapter: blendedmvs
    priority: high
    requires_rendering: false
    official_url: https://github.com/YoYo000/BlendedMVS
    expected_modalities: [rgb, depth, camera, pose, pairs]

  - id: 126
    name: nuScenes
    type: autonomous_multisensor
    adapter: nuscenes
    priority: high
    requires_rendering: false
    expected_modalities: [rgb, pose, lidar, boxes3d, map, lidarseg]

  - id: 244
    name: InteriorAgent
    type: usd_scene_asset
    adapter: interioragent_render
    priority: mid_high
    requires_rendering: true
    expected_modalities: [usd, mesh, room_polygon, materials]
```

registry 中每个数据集必须包含：`id, name, aliases, type, priority, license_url, official_url, adapter, expected_modalities, requires_rendering, known_coordinate_systems, validation_rules`。

### 5.2 阶段 1：实现基础 adapter 框架

目录建议：

```text
dataset_skill/
├── adapters/
│   ├── base.py
│   ├── blendedmvs.py
│   ├── colmap.py
│   ├── nerfstudio.py
│   ├── nuscenes.py
│   ├── kitti.py
│   ├── scannet.py
│   ├── arkit_scenes.py
│   ├── ase.py
│   ├── uco3d.py
│   ├── mesh_render.py
│   ├── isaac_render.py
│   └── remote_sensing.py
├── geometry/
│   ├── cameras.py
│   ├── poses.py
│   ├── depth.py
│   ├── pointcloud.py
│   ├── projection.py
│   └── conventions.py
├── render/
│   ├── blender_render.py
│   ├── habitat_render.py
│   └── isaac_render.py
├── validators/
│   ├── file_integrity.py
│   ├── geometry_checks.py
│   ├── masks.py
│   └── visual_debug.py
└── cli.py
```

### 5.3 阶段 2：统一 CLI

```bash
# 识别数据集
python -m dataset_skill inspect \
  --input /data/BlendedMVS \
  --output /work/reports/blendedmvs_inspection.json

# 抽样转换
python -m dataset_skill convert \
  --dataset blendedmvs \
  --input /data/BlendedMVS \
  --output /data/target \
  --split train \
  --sample-scenes 2 \
  --dry-run false

# 验证
python -m dataset_skill validate \
  --input /data/target/scenes/BlendedMVS \
  --make-html-debug true

# 对 mesh/USD 数据渲染
python -m dataset_skill render-convert \
  --dataset interioragent \
  --input /data/InteriorAgent \
  --output /data/target \
  --renderer isaac \
  --views-per-scene 128 \
  --resolution 1024x1024
```

### 5.4 阶段 3：优先级落地顺序

第一批 P0/P1：先打通不需要渲染的高优先级几何数据。

```text
1. BlendedMVS：格式规则、pair 信息明确，是最快闭环。
2. KITTI / KITTI-360：自动驾驶几何标准数据，验证长序列和多相机。
3. nuScenes：验证 token relational schema、多传感器 pose 链。
4. Waymo：验证 TFRecord/protobuf 和 LiDAR-projected depth。
5. DL3DV-10K：验证 COLMAP/Nerfstudio-style 大规模 NVS 数据。
6. uCO3D / CO3D：验证 object-centric multiview + mask + point cloud + GS。
```

第二批 P2：室内 RGB-D/mesh 数据。

```text
ARKitScenes, ScanNet v2, ScanNet++, Matterport3D, ASE, Hypersim
```

第三批 P3：需要渲染的 scene/asset 数据。

```text
3D-FRONT, HM3D, Replica, ReplicaCAD, InteriorAgent, InteriorGS, SAGE-10k,
Tabletop_Scenes, Objaverse-XL, Articraft-10K, YCB model-only subset
```

第四批 P4：遥感/地图/视频弱监督。

```text
BDD100K, OpenVid-1M, OpenSatMap, SEED-MAP, 3D-GloBFP
```

### 5.5 阶段 4：几何质量门禁

每个转换后的 scene 必须通过：

```text
文件完整性：frames.jsonl 中所有路径存在。
相机完整性：每个 frame 有 camera_ref；K 尺寸与图像一致。
位姿合法性：R^T R≈I，det(R)>0，T 第四行为 [0,0,0,1]。
尺度检查：depth 中位数、点云范围、camera baseline 不异常。
深度检查：无效值比例、单位、z-depth/ray-depth 标记正确。
投影检查：随机点云投影到图像应落在合理范围。
mask 检查：semantic/instance id 与 mapping 一致。
pair 检查：baseline、view angle、overlap score 分布合理。
split 检查：object/scene 不泄漏到多个 split，除非明确允许。
渲染检查：黑图、空 mask、相机穿墙、过近裁剪、重复视角过滤。
```

### 5.6 阶段 5：渲染策略

对于 3D-FRONT、HM3D、Replica、ReplicaCAD、InteriorAgent、SAGE-10k、Tabletop、Objaverse-XL、Articraft-10K 等，需要统一渲染策略。

```yaml
render_policy:
  camera_model: OPENCV_PINHOLE
  resolution: [1024, 1024]
  fov_deg: [55, 75]
  views_per_scene: 64
  trajectory_types:
    - random_navigable
    - room_center_orbit
    - object_orbit
    - short_temporal_walk
  outputs:
    - rgb
    - depth_z_m
    - normal_camera
    - semantic
    - instance
    - camera_pose_c2w
  filters:
    min_valid_depth_ratio: 0.55
    max_depth_m: 80
    min_rgb_entropy: 3.0
    max_duplicate_ssim: 0.95
    collision_free_camera: true
```

Mesh/object 资产需要额外标准化：

```text
1. 读取 mesh bounding box。
2. 自动推断单位；无法推断时保留 raw scale，并记录 unknown_scale。
3. 修复空 mesh、负 scale、缺材质、纹理丢失。
4. 计算 canonical object center、up axis、front axis。
5. 渲染 turntable views 和 random views。
```

### 5.7 阶段 6：伪标签策略

对于 BDD100K、OpenVid-1M 等无 pose/depth 数据：

```text
1. 抽帧：按 2-5 FPS 抽取，过滤模糊/低纹理/强动态帧。
2. 相机内参：若无官方内参，使用 EXIF 或估计焦距，并标记为 estimated。
3. pose：COLMAP / VGGSfM / DROID-SLAM / MASt3R-SLAM / VGGT-Long 生成。
4. depth：metric depth model 或 MVS/SfM 稀疏深度补全。
5. 标记：所有生成结果写 confidence 与 pseudo_label=true。
6. 训练：只用于弱监督或预训练，不与 GT geometry 混合计算硬指标。
```

---

## 6. Adapter 编写规范

### 6.1 不同格式的共用读取器

必须先实现以下共用 adapter，避免每个数据集重复造轮子：

```text
COLMAPAdapter：读取 cameras/images/points3D 的 txt/bin。
NerfstudioAdapter：读取 transforms.json，处理 OpenGL/Blender 到 OpenCV 转换。
VideoFrameAdapter：mp4/mkv 抽帧并与 metadata 对齐。
HDF5DepthAdapter：读取 Hypersim/uCO3D/RobustMVD 等 HDF5。
PFMDepthAdapter：读取 BlendedMVS/MVSNet 系列 PFM。
SQLiteMetadataAdapter：读取 uCO3D/自定义 SQLite metadata。
TFRecordWaymoAdapter：读取 Waymo protobuf/tfrecord。
NuScenesTokenAdapter：读取 nuScenes token relational schema。
SensScanNetAdapter：解码 ScanNet .sens。
MeshSceneAdapter：读取 glb/obj/ply/usd/usda/usdz 并交给 renderer。
RemoteSensingAdapter：读取 GeoJSON/COCO/ADE20K/GeoTIFF/shapefile。
```

### 6.2 pose 方向处理

每个 adapter 必须显式声明：

```python
pose_semantics = {
    "raw_pose_name": "extrinsic|c2w|w2c|ego_pose|sensor_to_ego",
    "raw_direction": "world_to_camera|camera_to_world|sensor_to_ego|ego_to_global",
    "raw_coord": "opencv|opengl|arkit|isaac|colmap|nuscenes|kitti_oxts",
    "target_direction": "camera_to_world",
    "target_coord": "opencv_camera_metric_world"
}
```

### 6.3 depth 处理

```python
def normalize_depth(raw_depth, encoding):
    if encoding == "uint16_mm":
        depth_m = raw_depth.astype(np.float32) / 1000.0
    elif encoding == "ray_distance_m":
        # 不要静默转为 z-depth。若需要 z-depth，必须用像素 ray 和 K 显式计算。
        depth_m = ray_to_z_depth(raw_depth, K)
    elif encoding == "inverse_depth":
        depth_m = 1.0 / np.maximum(raw_depth, eps)
    elif encoding == "pfm_m":
        depth_m = raw_depth.astype(np.float32)
    return depth_m
```

### 6.4 pair 构建策略

```yaml
pair_policy:
  temporal:
    offsets: [1, 2, 4, 8]
    max_baseline_m: 10.0
  stereo:
    use_same_timestamp: true
  object_orbit:
    angular_offsets_deg: [10, 20, 40, 80, 120]
  loop:
    retrieval: netvlad|dinov2|mast3r_tokens
    geometry_verify: true
  overlap:
    preferred_method: depth_projection
    fallback_method: pose_heuristic
```

---

## 7. 最小可行实现路线

### 第 1 周：跑通核心框架

```text
- 定义 target schema。
- 实现 DatasetAdapter base class。
- 实现 image/depth/pose/camera 写出器。
- 实现 QualityReport。
- 实现 BlendedMVS adapter。
- 用 1-2 个 scene 输出 target_dataset 样例。
```

### 第 2 周：加入自动驾驶数据

```text
- 实现 KITTI/KITTI-360 adapter。
- 实现 nuScenes adapter。
- 实现 pair 构建和 LiDAR projected depth。
- 增加 pose chain validation。
```

### 第 3 周：加入室内 RGB-D

```text
- 实现 ScanNet .sens 解码 adapter。
- 实现 ARKitScenes adapter。
- 实现 ASE adapter。
- 实现 Hypersim HDF5 adapter。
```

### 第 4 周：加入 object-centric 数据

```text
- 实现 CO3D/uCO3D adapter。
- 实现 SQLite/video/HDF5 对齐。
- 加入 masks、captions、point clouds、GS 复制。
```

### 第 5-6 周：加入渲染型数据

```text
- 实现 MeshSceneAdapter + BlenderProc renderer。
- 实现 Habitat renderer，覆盖 HM3D/Replica/ReplicaCAD。
- 实现 Isaac renderer，覆盖 InteriorAgent/SAGE/InteriorGS。
- 实现渲染质量过滤。
```

### 第 7 周：加入 Agent 自动写 adapter

```text
- Agent 根据 inspection_report 生成 adapter skeleton。
- 自动创建单元测试：detect、inspect、convert_sample、validate_sample。
- 人工审核后允许全量转换。
```

---

## 8. 典型执行伪代码

```python
def run_dataset_skill(user_request):
    plan = DatasetPlannerAgent.plan(user_request)
    registry = load_registry()

    source_profile = SourceResearchAgent.lookup(plan.dataset_name, registry)
    inspection = DatasetInspectorAgent.inspect(plan.input_root, source_profile)

    adapter = AdapterSelectorAgent.select(inspection, registry)
    if adapter is None:
        adapter = AdapterWriterAgent.write_adapter(inspection, source_profile)
        run_adapter_tests(adapter, plan.input_root)

    dry_report = ConversionAgent.convert(
        adapter=adapter,
        input_root=plan.input_root,
        output_root=plan.output_root,
        sample_only=True,
        dry_run=False,
    )

    quality = GeometryValidationAgent.validate(dry_report.output_scene)
    if not quality.passed:
        AdapterWriterAgent.revise(adapter, quality.failures)
        return run_dataset_skill(user_request)

    full_report = ConversionAgent.convert(
        adapter=adapter,
        input_root=plan.input_root,
        output_root=plan.output_root,
        sample_only=False,
        resume=True,
    )

    ReportAgent.write_dataset_card(full_report)
    return full_report
```

---

## 9. 输出报告模板

每次转换完成后必须输出：

```text
inspection_report.json
  - dataset_name/version
  - detected_modalities
  - directory_signature
  - sample_files
  - coordinate_assumptions
  - depth_assumptions
  - license_summary

conversion_report.json
  - converted_scenes
  - converted_frames
  - skipped_frames + reason
  - copied_meshes/pointclouds/gs
  - generated_pairs
  - runtime

quality_report.json
  - failed_checks
  - warning_checks
  - pose_stats
  - depth_stats
  - mask_stats
  - pair_stats
  - visual_debug_paths
```

---

## 10. 当前不确定项

以下数据集需要下载样本或用户补充信息后才能写稳定 adapter：

```text
Maya：无官方 URL，无法确认数据格式。
MegaDepth：官方定义明确，但常用文件结构依赖镜像/预处理版本。
MVImgNet / MVImgNet 2.0：模态明确，但具体下载包目录需样本确认。
YCB：不同子集结构差异大，需确认用户使用 object model set 还是 RGB-D benchmark。
InteriorGS：原始 compressed PLY 与 USDZ 转换版字段不同，需确认是否包含语义/occupancy。
Wayve：本 Skill 按 WayveScenes101 处理；如用户指其他 Wayve 数据，需要重新建 profile。
```

---

## 11. 推荐优先级表

| 优先级 | 数据集 | 原因 |
|---|---|---|
| P0 | BlendedMVS | 多视图 RGB + depth + camera + pair，最适合验证目标格式 |
| P0 | KITTI / KITTI-360 | 自动驾驶几何基准，pose/calib/LiDAR 清晰 |
| P0 | nuScenes | 多传感器 token schema，适合验证复杂 pose chain |
| P0 | Waymo | 高价值自动驾驶数据，适合验证 TFRecord/protobuf 适配 |
| P1 | DL3DV-10K | COLMAP/Nerfstudio 风格，适合 NVS/3DGS 前馈数据 |
| P1 | uCO3D / CO3D | object-centric 多视图，带 mask/depth/point cloud/GS |
| P1 | ScanNet++ / ScanNet v2 | 室内 RGB-D + mesh + semantic，高价值 |
| P2 | ARKitScenes / ASE / Hypersim | 室内 RGB-D/合成数据，适合补充室内分布 |
| P2 | InteriorAgent / SAGE-10k / ReplicaCAD | 可渲染仿真场景，适合大规模合成数据 |
| P3 | 3D-FRONT / HM3D / Replica / Objaverse-XL | mesh/asset-first，需渲染与过滤 |
| P4 | BDD100K / OpenVid-1M | 缺少 GT geometry，适合弱监督/预训练 |
| P4 | OpenSatMap / SEED-MAP / 3D-GloBFP | 遥感/地图专用，不属于常规三维前馈数据 |

---

## 12. 最终建议

先不要一开始追求覆盖所有数据集。建议先实现“格式稳定、几何字段完整”的 6 个 adapter：

```text
BlendedMVS → KITTI/KITTI-360 → nuScenes → DL3DV-10K → ScanNet++ → uCO3D
```

这 6 类覆盖了 MVS、自动驾驶、COLMAP/Nerfstudio、室内 RGB-D、object-centric/GS 五种主格式。等 target schema 和 validator 稳定后，再扩展到 mesh/USD/GLB 渲染型数据集。这样能避免在坐标系、depth 单位、pose 方向和语义映射尚不稳定时引入过多数据源。

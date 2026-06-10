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

合理的边界是把数据源分成两条路径，而不是强行把所有数据集都转成统一中间格式。

| 类型 | 处理方式 | 适用数据 |
| --- | --- | --- |
| 直接读取型 | 保留当前 dataset-specific dataloader，训练时直接从官方/挂载目录读 | BlendedMVS、KITTI odometry、nuScenes、部分 Wayve/CO3D 等已经有 RGB/pose/depth 或只需轻量解析的数据 |
| 离线转换型 | 先离线生成实验室自定义训练包，再用一个统一 loader 读取 | mesh/USD/GLB/GS、需要渲染、需要 SfM/MVS/SLAM、需要 dense depth 补全的数据 |
| 不进入几何训练型 | 不补造几何，只作为 2D/语义/外观辅助数据或拒绝 | 只有 2D 标注、遥感地图、无可靠 pose/depth 的视频数据 |

这个设计的核心约束：

1. 训练阶段不做重计算。训练 dataloader 只做轻量 I/O、resize/crop、batch，不做渲染、SfM、MVS、SLAM、点云投影或大模型推理。
2. 离线转换后只保存训练需要的最小闭包：RGB 或压缩图像、depth 或 sparse depth、valid mask、camera intrinsics、camera pose、metadata/source 标记，以及可选 semantic/normal/pair 信息。
3. mesh、GS、USD、原始点云如果不再被训练直接读取，可以不进入训练包，只在转换日志里记录来源和版本。
4. 对 BlendedMVS 这类本身已经接近训练格式的数据，直接 dataloader 更合适，强行复制中间格式会浪费空间和时间。
5. 实验室格式只服务训练，不追求保存所有原始资产。原始数据集仍由官方目录负责保存和追溯。

整体流程定义为：

```text
official / raw dataset
  -> direct dataloader
  -> training

official / raw dataset
  -> offline converter / renderer / estimator
  -> lab training package
  -> generic lab-format dataloader
  -> training
```

实验室自定义数据集规范重点定义 training-ready package：

```text
lab_dataset/
  dataset.json
  splits/
    train.jsonl
    val.jsonl
  scenes/
    <scene_id>/
      frames.jsonl
      images/
      depth/
      masks/
      cameras.json
      poses.jsonl
      pairs.jsonl
      metadata.json
```

如果是多视图训练，核心是 `frames.jsonl + cameras + poses + depth/mask + pairs`。不要把实验室格式设计成另一个完整数据湖。

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

背景说明：3D-FRONT 由 Alibaba Group/Taobao 技术团队等研究者提出，并与 3D-FUTURE 家具资产体系配套使用。它最初服务于室内布局生成、家具摆放、场景合成和室内设计相关任务，目标是把专业设计房间转化为可计算、可渲染的数据源。后续它被 BlenderProc、室内 layout generation、文本驱动室内场景生成等工作反复使用，逐渐成为合成室内资产和布局研究中的基础数据之一。

原始内容：合成室内家居场景。核心是房屋/房间布局 JSON，家具资产来自 3D-FUTURE，纹理来自 3D-FRONT-texture。论文说明该数据集包含大规模 furnished rooms、布局语义和高质量带纹理家具模型。

Metadata：

```json
{
  "profile": "rendered_indoor_scene",
  "domain": "室内合成场景",
  "storage": {
    "summary": "json + mesh + texture; CAD/mesh原生",
    "unit": "house/flat scene"
  },
  "modalities": {
    "rgb": "渲染生成",
    "depth": "渲染生成",
    "sem2d": "渲染生成",
    "pose": "采样生成",
    "pointcloud": "派生(mesh采样)",
    "pointcloud_semantic": "实例映射",
    "text": "无",
    "gs": "无"
  },
  "convention": "Blender/asset坐标需确认",
  "access": "public/需确认",
  "risk": "JSON到3D-FUTURE资产映射、单位和材质缺失"
}
```

格式/目录：常见下载包包括 `3D-FRONT/`、`3D-FUTURE-model/`、`3D-FRONT-texture/`。`3D-FRONT` 内每个 JSON 表示一个 house/flat；渲染时需要同时传入 JSON、3D-FUTURE 模型路径和 texture 路径。

目标转换：这是 mesh/layout-first 数据集，不直接提供 RGB/depth/pose。需要 RenderAgent 在 BlenderProc/Blender 中加载 house JSON，采样相机，渲染 RGB、depth、normal、semantic/instance mask，再写入目标格式。家具 CAD、房间 polygon、object transform 进入 `meshes/` 与 `annotations/objects.json`。

风险点：需要确认每个 JSON 的坐标单位、家具 id 到 3D-FUTURE 模型的映射、墙/地/天花板纹理缺失情况。直接用于前馈数据集时优先构建合成 RGB-D + pose，而不是直接训练 mesh。

建议：构建前馈数据集：是。优先级：中高。

### 114. Aria Synthetic Environments **

官方/主要参考：
- https://www.projectaria.com/datasets/ase/
- https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_data_format

背景说明：Aria Synthetic Environments 由 Meta Reality Labs Research / Project Aria 团队发布，背景是 Project Aria 眼镜形态设备需要大量可控、可标注的第一人称数据。它最初用于 AR 感知、定位、半稠密建图和室内场景理解，后续继续作为 Project Aria 开放数据生态的一部分，并被扩展到传感器仿真、合成 LiDAR 和 embodied/egocentric 感知研究中。

原始内容：合成室内场景，提供 RGB、depth、instance segmentation、trajectory、scene language、semidense points/observations、object-to-class mapping。

Metadata：

```json
{
  "profile": "rgbd_sequence",
  "domain": "室内合成场景",
  "storage": {
    "summary": "jpg + png + txt + csv.gz + json",
    "unit": "scene trajectory"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "ray_distance_mm原生",
    "sem2d": "instance原生",
    "pose": "trajectory原生",
    "pointcloud": "semidense原生",
    "pointcloud_semantic": "弱映射",
    "text": "scene_language原生",
    "gs": "无"
  },
  "convention": "fisheye + ray depth",
  "access": "public",
  "risk": "不能把fisheye/ray depth当作pinhole/z-depth"
}
```

Format/layout: use the official ASE data format page as the source of truth; do not duplicate the full tree here.
- https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_synthetic_environments_dataset/ase_data_format

Reader notes: parse fisheye RGB, 16-bit ray-distance depth, instance masks, trajectory, semidense points/observations, scene language, and object-instance mapping according to the official schema. Sample files still need to be inspected before conversion.

目标转换：RGB 复制到 `rgb/`；depth 需从 ray distance 转换或至少标记为 `ray_distance_m`；instance mask 写入 `instance/`；`object_instances_to_classes.json` 写入 class mapping；`trajectory.txt` 转为 `poses.jsonl`；semidense CSV 写入 `point_clouds/semidense_points.ply/npz`。

风险点：fisheye 相机模型必须保留，不能未经处理当作 pinhole。若训练模型只支持 pinhole，需要先去畸变并同步重采样 depth/mask。

建议：构建前馈数据集：是。优先级：中高。

### 115. ARKitScenes **

官方/主要参考：
- https://github.com/apple/ARKitScenes
- https://github.com/apple/ARKitScenes/blob/main/DATA.md

背景说明：ARKitScenes 由 Apple 机器学习研究团队发布，抓住了 iPhone/iPad LiDAR 开始普及后移动端 RGB-D 采集能力提升的时间点。数据集最初面向真实室内 3D scene understanding、3D object detection 和 color-guided depth upsampling，后续成为移动端 LiDAR 室内数据的重要基准，被室内 3D 标注、AR/机器人感知和 depth/mesh 学习工作持续复用。

原始内容：真实室内 RGB-D 扫描。包括低/高分辨率 RGB、低/高分辨率 depth、confidence、ARKit pose、mesh/point cloud、3D object annotations 等，具体取决于 raw、3dod、upsampling 子集。

Metadata：

```json
{
  "profile": "indoor_rgbd_scan",
  "domain": "真实室内扫描",
  "storage": {
    "summary": "png + traj + pincam + json + ply + mov + txt; mesh原生",
    "unit": "scan/scene"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "原生",
    "sem2d": "有限/子集",
    "pose": "原生",
    "pointcloud": "原生",
    "pointcloud_semantic": "3D标注子集",
    "text": "无",
    "gs": "无"
  },
  "convention": "ARKit pose/单位需样本确认",
  "access": "public",
  "risk": "子集字段差异、pose缺失、confidence过滤"
}
```

Format/layout: use the official ARKitScenes `DATA.md` for raw, 3dod, and depth upsampling asset groups.
- https://github.com/apple/ARKitScenes/blob/main/DATA.md

Reader notes: first choose the asset group, then parse `.pincam`, `.traj`, JSON annotations, PLY meshes, MOV/video assets, depth, confidence, and image files according to that group. Do frame-level joins explicitly.

目标转换：选择 `raw` 或 `depth_upsampling` 子集时，抽取 RGB/depth/confidence/pose/intrinsics；选择 `threedod` 子集时，额外抽取 3D bounding boxes 和 scene mesh。`.traj` 或 `_pose.txt` 转换为 `T_c2w` 前必须确认方向。depth 通常为 metric depth，但需用样本确认单位与尺度。

风险点：ARKitScenes 存在多个下载资产组，字段不完全一致；部分帧 pose 可能缺失或无效。必须做 frame-level join 和 confidence mask 过滤。

建议：构建前馈数据集：是。优先级：中高。

### 116. BDD100K ***

官方/主要参考：
- https://bair.berkeley.edu/blog/2018/05/30/bdd/
- https://github.com/bdd100k/bdd100k

背景说明：BDD100K 由 UC Berkeley / Berkeley DeepDrive 团队发布，针对早期自动驾驶视觉数据在天气、时间、道路环境和任务类型上覆盖不足的问题而构建。它一开始就服务 heterogeneous multitask learning，而不是单一检测任务；后续长期作为自动驾驶 2D 感知、多任务学习、车道线/可行驶区域/分割 benchmark 和工程预训练数据源使用。

原始内容：真实自动驾驶视频/图像数据。主要提供 RGB、检测框、车道线、可行驶区域、语义/实例/panoptic 分割等 2D 标注。通常不提供深度、相机位姿和点云。

Metadata：

```json
{
  "profile": "driving_2d_perception",
  "domain": "真实道路驾驶",
  "storage": {
    "summary": "jpg + mov + json + png",
    "unit": "image/video frame"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "无",
    "sem2d": "原生",
    "pose": "无",
    "pointcloud": "无",
    "pointcloud_semantic": "无",
    "text": "无",
    "gs": "无"
  },
  "convention": "2D image task schema",
  "access": "public/需注册确认",
  "risk": "不能伪装几何真值，视频和标注同步需确认"
}
```

Format/layout: use the official BDD100K repository and task documentation for images, videos, and label package layouts.
- https://github.com/bdd100k/bdd100k

Reader notes: BDD100K is RGB/video plus 2D task labels. It has no official GT camera pose or depth for this project, so any generated geometry must be marked as estimated or pseudo labels.

目标转换：只可直接构建 RGB + 2D semantics/instances 的前馈弱监督数据。若目标必须包含 pose/depth，需要额外用 SfM/SLAM/单目深度估计生成伪标签。BDD100K 更适合做开放道路场景的外观/语义预训练，而非几何强监督。

风险点：视频帧与标注帧的同步关系需确认。无 GT pose/depth，不能伪装为几何真值。

建议：构建前馈数据集：是，但属于 RGB/2D 标注型。优先级：高。

### 117. BlendedMVS / BlendMVS ***

官方/主要参考：
- https://github.com/YoYo000/BlendedMVS

背景说明：BlendedMVS 由浙江大学等团队在 CVPR 2020 发布，核心动机是补足深度学习 MVS 网络训练数据不足、场景复杂度不够的问题。相比 DTU 等早期数据，它更大、更复杂，也更接近真实应用场景；后来逐渐成为 MVS、NVS、NeRF/3DGS 数据预处理和多视图几何 adapter 验证中常用的公开样本。

原始内容：MVS/NVS 训练数据，包含多视图 RGB、渲染/融合深度图、相机参数、pair 信息，场景涵盖建筑、雕塑、小物体等。

Metadata：

```json
{
  "profile": "multiview_mvs",
  "domain": "多视图重建场景",
  "storage": {
    "summary": "jpg + pfm + txt; 相机+深度/pair",
    "unit": "MVS scene"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "pfm原生",
    "sem2d": "无",
    "pose": "相机参数原生",
    "pointcloud": "MVS可重建",
    "pointcloud_semantic": "无",
    "text": "无",
    "gs": "无"
  },
  "convention": "MVSNet world-to-camera常见",
  "access": "public",
  "risk": "外参方向和depth range解析"
}
```

Format/layout: use the official BlendedMVS repository README for list files, project ids, `blended_images`, `cams/pair.txt`, `*_cam.txt`, and `rendered_depth_maps`.
- https://github.com/YoYo000/BlendedMVS

Reader notes: project ids come from official list files; camera files follow the MVSNet-style format; depth maps are PFM files. This repo should use official list files or explicit `roots.list`, not local derived list names.

目标转换：这是最适合直接转换成前馈多视图几何格式的数据集之一。读取 `pair.txt` 生成 `pairs.jsonl`；读取 `*_cam.txt` 生成 `cameras.json` 与 `poses.jsonl`；`.pfm` depth 转为 `.npy/.exr` 或保留并标注；RGB 转到 `rgb/`。

风险点：需确认 `*_cam.txt` 中外参是 world-to-camera 还是 camera-to-world，MVSNet 系列通常使用 extrinsic world-to-camera。转换时要统一求逆。

建议：构建前馈数据集：是。优先级：高。

### 118. CO3D v1/v2

官方/主要参考：
- https://github.com/facebookresearch/co3d/tree/v1
- https://github.com/facebookresearch/co3d

背景说明：CO3D 由 Facebook AI Research / Meta AI 发布，面向常见物体类别的 category-level 3D reconstruction 和 new-view synthesis。它把真实多视图视频、相机、mask、深度和点云组织成 object-centric 数据，后续从 CO3D v1/v2 延伸到 uCO3D，也推动了 object-centric 3D foundation model、NeRFormer、LRM/Instant3D 类训练数据的发展。

原始内容：真实 object-centric 多视图数据，提供 RGB、foreground masks、depth/depth masks、camera poses、point cloud、category/sequence annotations。v2 相比 v1 有更多 sequences/frames 和更好的 mask。

Metadata：

```json
{
  "profile": "object_multiview_sequence",
  "domain": "真实物体中心序列",
  "storage": {
    "summary": "jpg + png + jgz + json + ply; point cloud原生",
    "unit": "category/sequence/frame"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "原生",
    "sem2d": "foreground mask",
    "pose": "原生",
    "pointcloud": "原生",
    "pointcloud_semantic": "类别级弱语义",
    "text": "类别元数据(非caption)",
    "gs": "无"
  },
  "convention": "PyTorch3D/CO3D camera需归一",
  "access": "public",
  "risk": "v1/v2结构差异和mask/depth有效性"
}
```

Format/layout: use the official CO3D repositories for category, sequence, frame annotation, sequence annotation, image, mask, depth, and point-cloud organization.
- https://github.com/facebookresearch/co3d/tree/v1
- https://github.com/facebookresearch/co3d

Reader notes: `frame_annotations.jgz` and `sequence_annotations.jgz` are the authoritative indexes. Resolve paths from annotations instead of guessing from directory names.

目标转换：按 category/sequence 映射为 scene；frame annotations 转 `frames.jsonl/cameras.json/poses.jsonl`；mask 写 `instance/valid_mask`；depth 写 `depth/`；pointcloud 写 `point_clouds/pointcloud.ply`。object-centric 数据默认以物体为场景中心，适合 pair/multiview 训练。

风险点：CO3D 坐标系、NDC/pytorch3d camera convention 与 OpenCV 不同，需要专门转换。不要直接把 PyTorch3D camera matrix 当作 OpenCV K/R/T。

建议：构建前馈数据集：是。优先级：中高。

### 119. DL3DV-10K

官方/主要参考：
- https://dl3dv-10k.github.io/DL3DV-10K/
- https://huggingface.co/datasets/DL3DV/DL3DV-10K

背景说明：DL3DV-10K 由 Purdue University 等团队发布，目标是把 deep learning-based 3D vision 从小规模、多为受控采集的数据推进到大规模真实场景视频。它最初服务 novel view synthesis 和 NeRF 泛化训练，后续形成 DL3DV-10K benchmark 与下载工具链，并逐渐被用于大规模 NVS、3DGS 和场景级重建模型评测。

原始内容：大规模真实视频数据，面向 NVS/3D reconstruction。提供 4K 视频、COLMAP 标定结果、稀疏点云、downsampled images、`transforms.json` 等。

Metadata：

```json
{
  "profile": "scene_multiview_recon",
  "domain": "真实多视图场景",
  "storage": {
    "summary": "mp4 + jpg + json + txt + bin + ply; RGB/video + calibrated poses + COLMAP/Nerfstudio-style outputs",
    "unit": "scene/video/frame"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "派生(重建深度/非GT)",
    "sem2d": "scene label非像素级",
    "pose": "原生",
    "pointcloud": "派生(COLMAP点云)",
    "pointcloud_semantic": "无",
    "text": "human scene labels",
    "gs": "benchmark/可训练生成"
  },
  "convention": "Nerfstudio/OpenGL与COLMAP需区分",
  "access": "public/需确认",
  "risk": "视频抽帧、相机约定、下载包是否含3DGS需样本确认"
}
```

Format/layout: use the official DL3DV-10K page and Hugging Face dataset page for scene/video shards, COLMAP outputs, transforms, and image folders.
- https://dl3dv-10k.github.io/DL3DV-10K/
- https://huggingface.co/datasets/DL3DV/DL3DV-10K

Reader notes: different shards may contain different derived assets. Prefer explicit `transforms.json` or COLMAP sparse models, then verify filename/frame/pose alignment from samples.

目标转换：优先使用 `transforms.json` 或 COLMAP `cameras/images/points3D` 生成 camera/pose/point cloud；`pairs-netvlad.txt` 可转为候选 pair；RGB 使用 downsampled images 控制训练成本。若 `transforms.json` 是 Nerfstudio/OpenGL convention，需要转换到 OpenCV camera。

风险点：不同发布分片可能结构略有差异；视频抽帧与 COLMAP images 必须按文件名对齐。

建议：构建前馈数据集：是。优先级：高。

### 120. HM3D

官方/主要参考：
- https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d
- https://matterport.com/habitat-matterport-3d-research-dataset

背景说明：HM3D 由 Meta AI Habitat 团队与 Matterport 合作发布，延续了 Habitat 对大规模真实室内环境的需求。它最初为 embodied AI、PointGoal/ObjectNav 导航和 Habitat 仿真提供 1000 个建筑级真实室内 3D 环境，之后扩展出 HM3D Semantics 等语义版本，并成为 Habitat Challenge、导航模型训练和室内仿真评测的核心资产。

原始内容：真实室内/建筑级 textured 3D mesh，常用于 Habitat。包含 `.glb/.obj` mesh、textures、语义版本/语义映射（取决于下载版本）。不直接提供 RGB-D 轨迹帧。

Metadata：

```json
{
  "profile": "rendered_indoor_mesh",
  "domain": "真实室内mesh资产",
  "storage": {
    "summary": "glb + obj + mtl + texture + semantic metadata; mesh原生",
    "unit": "Habitat scene asset"
  },
  "modalities": {
    "rgb": "渲染生成",
    "depth": "渲染生成",
    "sem2d": "语义版可渲染",
    "pose": "采样生成",
    "pointcloud": "派生(mesh采样)",
    "pointcloud_semantic": "语义版映射",
    "text": "无",
    "gs": "无"
  },
  "convention": "Habitat坐标/语义版本需确认",
  "access": "agreement_required/token",
  "risk": "下载门槛、语义版差异和相机采样有效性"
}
```

格式/目录：通常以 scene 为单位存储 mesh asset，例如 `.glb` 或 `.obj + .mtl + textures`；HM3D Semantics 版本提供带实例/类别颜色或语义映射的 mesh。

目标转换：作为 mesh-first 数据集，需要 RenderAgent 在 Habitat-Sim 中采样导航轨迹或相机位姿，渲染 RGB/depth/semantic，生成 `frames.jsonl` 和 `pairs.jsonl`。原始 mesh 存入 `meshes/`。

风险点：下载权限、版本差异、语义版本和非语义版本字段不同。相机采样应避免穿墙、过近、视野无效。

建议：构建前馈数据集：是，但需渲染。优先级：中。

### 121. Hypersim *

官方/主要参考：
- https://github.com/apple/ml-hypersim

背景说明：Hypersim 由 Apple 机器学习研究团队发布，思路是用高真实感合成渲染弥补真实室内数据难以获得完整几何、材质、光照和像素级标注的问题。它最初服务 holistic indoor scene understanding，后来持续被用于室内深度、法线、语义、材质估计和合成到真实泛化研究。

原始内容：大规模 photorealistic synthetic indoor 数据，提供 RGB/HDR、depth、surface normal、3D position、semantic、semantic instance、mesh/object metadata、camera trajectory、lighting/material 信息。

Metadata：

```json
{
  "profile": "indoor_synthetic_fullmodal",
  "domain": "合成室内场景",
  "storage": {
    "summary": "csv + hdf5 + tonemap jpg/png; mesh/object metadata",
    "unit": "scene/camera/frame"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "meters原生",
    "sem2d": "原生",
    "pose": "原生",
    "pointcloud": "派生(position/depth)",
    "pointcloud_semantic": "语义对齐",
    "text": "无",
    "gs": "无"
  },
  "convention": "asset units到meters + camera convention",
  "access": "public",
  "risk": "HDF5体量、官方过滤列表和单位换算"
}
```

Format/layout: use the official Hypersim repository documentation for scene ids, `_detail`, images, metadata CSV files, and HDF5 render passes.
- https://github.com/apple/ml-hypersim

Reader notes: parse camera trajectories, intrinsics, render entity ids, depth, normal, semantic, and metadata from official files. Use official metadata to filter invalid scenes or frames.

目标转换：直接抽取 RGB、depth_meters、normal、semantic、semantic_instance、camera trajectory。由于 Hypersim 的 depth/camera convention 有明确说明，转换时必须保留原始 convention 并统一到 OpenCV target。object metadata/3D bbox 写入 annotations。

风险点：HDF5 文件体量大，读取时必须 lazy loading；某些场景/图像被官方排除，需要使用官方 split/metadata 过滤。

建议：构建前馈数据集：是。优先级：中。

### 122. Matterport3D *

官方/主要参考：
- https://niessner.github.io/Matterport/
- https://github.com/niessner/Matterport

背景说明：Matterport3D 由 Stanford、Princeton、Technical University of Munich 等研究者与 Matterport 数据采集体系共同推动，是早期建筑级真实 RGB-D 室内数据的重要代表。它最初为 RGB-D indoor scene understanding 提供全景、深度、pose、mesh 和语义标注，之后又成为 ScanNet、HM3D、Habitat、Matterport3D Simulator 等室内理解和 embodied AI 工作的重要前序基准。

原始内容：真实建筑级 RGB-D panorama/mesh 数据，包含大量 RGB-D 图像、camera poses、surface reconstructions、2D/3D semantic annotations、region/object annotations。

Metadata：

```json
{
  "profile": "indoor_rgbd_mesh",
  "domain": "真实室内建筑",
  "storage": {
    "summary": "jpg + png + pose + ply + obj + json + house metadata; mesh原生",
    "unit": "building/region/panorama/view"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "原生",
    "sem2d": "原生/可导出",
    "pose": "原生",
    "pointcloud": "派生(RGB-D融合/mesh采样)",
    "pointcloud_semantic": "region/object语义",
    "text": "无",
    "gs": "无"
  },
  "convention": "panorama/local view/mesh坐标需统一",
  "access": "agreement_required",
  "risk": "许可、全景相机模型和多坐标系对齐"
}
```

Format/layout: use the official Matterport3D page and repository for release assets, download options, scans, panoramas, meshes, and semantic files.
- https://niessner.github.io/Matterport/
- https://github.com/niessner/Matterport

Reader notes: different download options expose different assets. Confirm that the current scan contains required images, pose/calibration, mesh, and semantic files before conversion.

目标转换：可直接从 RGB-D + pose 构建前馈训练数据，也可从 mesh 渲染补充视角。全景图可拆分成 perspective views，也可保留 equirectangular camera model；mesh/region/object semantics 写入 annotations 与 scene_graph。

风险点：数据下载需要 license；全景坐标、局部视图相机和 mesh 坐标之间的转换复杂。不要把 panorama 当普通 pinhole 图像。

建议：构建前馈数据集：是。优先级：中。

### 123. MegaDepth

官方/主要参考：
- https://www.cs.cornell.edu/projects/megadepth/

背景说明：MegaDepth 由 Cornell University / Cornell Tech 等团队提出，利用互联网地标照片、SfM 和 MVS 自动生成大规模单目深度训练监督。它一开始主要服务 single-view depth prediction，后来也成为单目深度、相对位姿、局部特征匹配和互联网照片重建领域的常用数据来源，并继续影响 long-tail Internet photo reconstruction 方向。

原始内容：从互联网照片集合通过 SfM/MVS 生成的大规模单目深度数据。主要用于单目深度和几何学习，场景多为室外地标/建筑。

Metadata：

```json
{
  "profile": "internet_photo_depth_sfm",
  "domain": "室外地标/建筑",
  "storage": {
    "summary": "jpg + h5/npy/pfm + COLMAP + SfM metadata; COLMAP/SfM",
    "unit": "photo collection/image"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "预处理原生",
    "sem2d": "无",
    "pose": "SfM/预处理",
    "pointcloud": "SfM稀疏点",
    "pointcloud_semantic": "无",
    "text": "无",
    "gs": "无"
  },
  "convention": "COLMAP/OpenCV常见但来源需记录",
  "access": "public/镜像差异",
  "risk": "版本分散、pose/depth是否官方GT不稳定"
}
```

格式/目录：官方页面说明其由 Internet photo collections 经 SfM/MVS 生成。实际使用中常见预处理包会包含 undistorted images、depth maps、camera intrinsics/extrinsics 或 COLMAP/SfM 输出，但不同镜像/二次处理版本差异较大。

目标转换：若有官方/预处理 pose 与 depth，可转换为 RGB + depth + camera + pose；若只有图像和 depth，需标注 pose unavailable；若使用 COLMAP 输出，按照 COLMAP adapter 处理。

风险点：版本分散，文件结构不统一。必须在 `inspection_report.json` 中记录具体来源。若 pose 来自第三方预处理，不应标为官方 GT。

建议：构建前馈数据集：是，但需样本确认。优先级：中。

### 124. MVImgNet

官方/主要参考：
- https://gaplab.cuhk.edu.cn/projects/MVImgNet/
- https://github.com/GAP-LAB-CUHK-SZ/MVImgNet

背景说明：MVImgNet 由香港中文大学深圳 GAP Lab 等团队发布，通过拍摄日常物体视频来构建大规模 object-centric 多视图图像集。它最初用于多视图表征、物体重建和类别级 3D 学习，后续派生出 MVPNet 点云数据，并进一步发展为 MVImgNet 2.0，提高 360 度覆盖、mask、SfM pose 和 dense point cloud 质量。

原始内容：object-centric 多视图真实视频/图像数据，覆盖大量物体类别。提供 RGB、object masks、camera parameters、point clouds 等标注。

Metadata：

```json
{
  "profile": "object_multiview_colmap",
  "domain": "真实物体多视图",
  "storage": {
    "summary": "jpg + png mask + COLMAP bin + point cloud; COLMAP稀疏模型",
    "unit": "category/object/view"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "无/可估计",
    "sem2d": "mask子集",
    "pose": "COLMAP原生",
    "pointcloud": "稀疏点原生",
    "pointcloud_semantic": "类别/实例级",
    "text": "类别标签(非caption)",
    "gs": "无"
  },
  "convention": "COLMAP/OpenCV",
  "access": "gated/form",
  "risk": "下载门槛、子集不一致和mask可用性"
}
```

格式/目录：官方仓库说明 MVImgNet 包含数百万帧、数十万视频和数百类别；完整数据通常按 category 分包下载。具体目录按 category/object/video 组织，标注包含 masks、camera parameters、point clouds。

目标转换：按 object instance 作为 scene；RGB 与 masks 写入 frame；camera/pose 写入 `cameras/poses`；point cloud 写入 `point_clouds`。适合构建 object-level multiview 前馈数据。

风险点：大规模压缩包，解压和索引耗时。mask/camera/point cloud 的具体文件名需样本确认。

建议：构建前馈数据集：是。优先级：中高。

### 125. MVImgNet 2.0

官方/主要参考：
- https://luyues.github.io/mvimgnet2/

背景说明：MVImgNet 2.0 是 MVImgNet 原团队的继续扩展，重点修正原版本在 360 度覆盖、前景 mask、SfM pose 精度和 dense point cloud 质量上的不足。它更适合作为 object-level reconstruction、NVS 和 3D foundation model 训练数据，也为同类 adapter 提供了从低质量 SfM 到高质量 SfM 的版本演化样例。

原始内容：MVImgNet 的扩展版本，包含更多 object instances/categories，并提供 segment masks、SfM poses、dense point clouds 等。

Metadata：

```json
{
  "profile": "object_multiview_colmap_dense",
  "domain": "真实物体多视图",
  "storage": {
    "summary": "image + mask + SfM pose + dense point cloud; SfM/点云",
    "unit": "category/object/view"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "无/可估计",
    "sem2d": "mask改进",
    "pose": "SfM原生",
    "pointcloud": "dense原生",
    "pointcloud_semantic": "类别/实例级",
    "text": "类别标签(非caption)",
    "gs": "无"
  },
  "convention": "SfM/OpenCV需确认",
  "access": "gated/需确认",
  "risk": "与MVImgNet版本差异和dense点云尺度"
}
```

格式/目录：以 object/category 为主组织，提供 360-degree views、masks、camera parameters/SfM poses、dense point clouds。具体目录需下载样本确认。

目标转换：同 MVImgNet，但优先使用更高质量的 masks/poses/dense point clouds。可直接生成 object-level multiview chunks。

风险点：官方页面与数据下载可能分批发布，具体文件命名和可下载字段需要验证。

建议：构建前馈数据集：是。优先级：中高。

### 126. nuScenes ***

官方/主要参考：
- https://www.nuscenes.org/nuscenes
- https://github.com/nutonomy/nuscenes-devkit

背景说明：nuScenes 由 nuTonomy/Aptiv 团队发布，后续由 Motional 维护。它在 Boston 和 Singapore 的复杂城市道路中采集 camera、LiDAR、radar、map 和 3D 标注，最初服务自动驾驶多传感器感知；之后扩展出 nuImages、lidarseg/panoptic nuScenes、nuPlan 和 nuReality，形成自动驾驶感知、规划和仿真研究的一整套系列基准。

原始内容：真实自动驾驶多传感器数据。提供 6 cameras、LiDAR、radar、ego pose、calibrated sensor、3D boxes、map、lidarseg/panoptic 等。

Metadata：

```json
{
  "profile": "driving_multisensor_sequence",
  "domain": "真实城市驾驶",
  "storage": {
    "summary": "json tables + jpg + pcd/bin + map files",
    "unit": "scene/sample/sample_data"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "派生(LiDAR投影)",
    "sem2d": "2D框/分割可用",
    "pose": "ego/sensor原生",
    "pointcloud": "LiDAR/Radar原生",
    "pointcloud_semantic": "lidarseg/panoptic可选",
    "text": "scene metadata(非caption)",
    "gs": "无"
  },
  "convention": "global/ego/sensor链路",
  "access": "public/需注册",
  "risk": "token关系表、时间同步和坐标链路"
}
```

Format/layout: nuScenes is a token-based relational schema. Use the official devkit schema for tables, `samples`, `sweeps`, maps, calibration, and ego pose links.
- https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md
- https://www.nuscenes.org/nuscenes

Reader notes: build indexes from JSON tables such as `scene`, `sample`, `sample_data`, `calibrated_sensor`, `ego_pose`, and `sensor`; then resolve files through `sample_data.filename`. Do not replace schema joins with directory scans.

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

背景说明：Objaverse-XL 由 Allen Institute for AI、University of Washington、Columbia University、Stability AI、LAION、Caltech 等机构合作发布。它通过聚合网络多源 3D 资产，为 3D 生成、NVS、资产检索和大规模 3D 表征学习提供开放语料；后续成为 Zero123/3D generative model 训练的重要资产库，同时也引发了围绕 license、质量清洗和资产来源治理的持续讨论。

原始内容：超大规模 3D object asset 数据集，包含 10M+ 3D objects。对象来自多源，许可证逐对象不同。通常通过 Python API 下载 metadata 和 object assets。

Metadata：

```json
{
  "profile": "asset_bank",
  "domain": "物体资产",
  "storage": {
    "summary": "api metadata + glb + gltf + obj + fbx + usd; mesh多格式原生",
    "unit": "object asset UID"
  },
  "modalities": {
    "rgb": "渲染生成",
    "depth": "渲染生成",
    "sem2d": "asset id渲染",
    "pose": "采样生成",
    "pointcloud": "派生(mesh采样)",
    "pointcloud_semantic": "metadata弱映射",
    "text": "metadata/tags(非caption)",
    "gs": "无"
  },
  "convention": "asset坐标/尺度需归一",
  "access": "mixed_license",
  "risk": "逐对象license、可渲染性和格式清洗"
}
```

格式/目录：通过 `objaverse` API 或 Hugging Face 访问。对象文件可能为 glb/gltf/obj/fbx/usd 等多种格式，metadata 提供 uid、source、license、tags/captions 等。仓库提供 rendering scripts。

目标转换：这是 asset-first 数据集。需要先筛选可渲染、可归一化、license 允许的对象，再用 Blender/Trimesh 渲染多视角 RGB-D/normal/mask，构建 object-level 前馈数据。原始 asset 复制到 `meshes/`，metadata 写入 captions/tags。

风险点：质量参差、尺度/朝向/材质/动画复杂，逐对象 license 不一致。必须执行 mesh repair、scale normalization、empty render filtering。

建议：构建前馈数据集：是，但需渲染与质量过滤。优先级：中。

### 128. OpenVid-1M

官方/主要参考：
- https://huggingface.co/datasets/nkp37/OpenVid-1M

背景说明：OpenVid-1M 由南京大学 PCALab、ByteDance、南开大学等团队发布，面向 text-to-video generation 构建百万级高质量视频-文本训练数据。它后续被用于视频生成、视频理解和多模态预训练研究；但放到本项目中，它只能作为视觉-语言背景数据或伪几何挖掘来源，而不能被视作几何真值数据集。

原始内容：大规模 text-video 数据，提供视频文件及 CSV/JSON 描述，包含 caption/metadata。无官方深度、相机位姿、点云。

Metadata：

```json
{
  "profile": "video_text_pretrain",
  "domain": "开放域视频",
  "storage": {
    "summary": "csv + json + mp4 + zip shards",
    "unit": "video/caption record"
  },
  "modalities": {
    "rgb": "视频帧原生",
    "depth": "无",
    "sem2d": "无",
    "pose": "无",
    "pointcloud": "无",
    "pointcloud_semantic": "无",
    "text": "caption原生",
    "gs": "无"
  },
  "convention": "video fps/resolution schema",
  "access": "public/需确认",
  "risk": "仅能作为弱监督/预训练，几何标签需标注为pseudo"
}
```

格式/目录：Hugging Face 数据集包含 `OpenVid-1M.csv`、`OpenVidHD.csv`、`OpenVidHD.json` 与多个视频 zip 分片。CSV 可用 pandas 读取，视频分辨率至少 512×512，OpenVidHD 包含较多 1080p 视频。

目标转换：只能直接构建 RGB/video + text 数据。若用于前馈三维，需要 VideoFrameExtractor 抽帧，再调用 SfM/SLAM/DepthEstimator 生成伪 pose/depth，并标注为 pseudo labels。

风险点：视频版权/许可、镜头运动不足、动态物体、文本与帧级内容未必严格对应。不能作为几何真值数据。

建议：构建前馈数据集：可选，主要做文本/视频预训练。优先级：低到中。

### 129. Replica **

官方/主要参考：
- https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#replica-dataset
- https://github.com/facebookresearch/Replica-Dataset

背景说明：Replica 由 Facebook Reality Labs、Facebook AI Research、Georgia Tech、Simon Fraser University 等团队发布，目标是为 Habitat 和 embodied agents 提供高真实感室内数字复刻。它包含 dense mesh、HDR texture 和语义标注，后续成为 Habitat-compatible 室内场景资产，并进一步被 ReplicaCAD 改造为可交互、可物理仿真的公寓场景。

原始内容：高质量室内空间重建，提供 clean dense geometry、HDR textures、semantic class/instance、planar segmentation、Habitat export。

Metadata：

```json
{
  "profile": "rendered_semantic_mesh",
  "domain": "真实室内重建",
  "storage": {
    "summary": "ply + json + texture + habitat export + navmesh; mesh原生",
    "unit": "scene asset"
  },
  "modalities": {
    "rgb": "渲染生成",
    "depth": "渲染生成",
    "sem2d": "语义mesh渲染",
    "pose": "采样生成",
    "pointcloud": "派生(mesh采样)",
    "pointcloud_semantic": "原生语义映射",
    "text": "无",
    "gs": "无"
  },
  "convention": "Habitat/mesh坐标",
  "access": "public",
  "risk": "渲染采样、语义映射和Habitat导出版本"
}
```

Format/layout: use Habitat and Replica official documentation for mesh, texture, semantic mesh, semantic metadata, and navmesh assets.
- https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#replica-dataset
- https://github.com/facebookresearch/Replica-Dataset

Reader notes: Replica is mesh/semantic-mesh first. Do not assume existing RGB-D trajectories; render RGB/depth/semantic views from the official assets.

目标转换：作为 mesh-first/semantic mesh 数据集，使用 Habitat-Sim 采样相机位姿并渲染 RGB/depth/semantic/instance。mesh 与 semantic metadata 写入 `meshes/` 与 annotations。

风险点：原始 Replica 不一定提供现成 RGB-D trajectory；需要渲染。语义 id 与类别映射应从 `info_semantic.json` 读取。

建议：构建前馈数据集：是。优先级：中高。

### 130. ReplicaCAD **

官方/主要参考：
- https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#replicacad

背景说明：ReplicaCAD 由 Meta AI / Habitat 2.0 团队基于 Replica 场景重建并由 3D artists 重新 CAD 化。它面向 Home Assistant Benchmark 和移动操作任务，提供可交互、带关节物体和物理属性的室内仿真环境；后续成为 Habitat 2.0 的核心交互场景资产，也推动静态 mesh 数据向 physics-ready interactive digital twin 演化。

原始内容：基于 Replica FRL apartment 的可交互 CAD/仿真室内场景，包含 static background、object assets、URDF/physical properties、receptacle metadata、scene configs、navmesh。

Metadata：

```json
{
  "profile": "interactive_sim_scene",
  "domain": "室内交互仿真",
  "storage": {
    "summary": "scene config + glb + urdf + navmesh + object config; GLB/URDF/navmesh原生",
    "unit": "interactive scene"
  },
  "modalities": {
    "rgb": "渲染生成",
    "depth": "渲染生成",
    "sem2d": "渲染生成",
    "pose": "采样生成",
    "pointcloud": "派生(mesh采样)",
    "pointcloud_semantic": "object metadata映射",
    "text": "配置元数据(非caption)",
    "gs": "无"
  },
  "convention": "Habitat scene dataset config",
  "access": "public",
  "risk": "刚体/关节体/navmesh/receptacle元数据不能丢"
}
```

Format/layout: use Habitat official documentation for ReplicaCAD scene dataset config, stages, rigid objects, articulated objects, scene instances, receptacles, and navmeshes.
- https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#replicacad

Reader notes: `scene_dataset_config` is the entry point. Preserve articulated object, receptacle, physics, and scene-instance metadata instead of flattening the dataset into static meshes.

目标转换：使用 Habitat-Sim/Isaac/Habitat renderer 从 scene config 渲染 RGB-D/semantic，并可保留物体物理属性、可交互关系到 `scene_graph.json`。原始 CAD/URDF/scene config 存入 `meshes/` 或 `annotations/sim_config`。

风险点：交互对象、可动关节、receptacle 与静态语义需要分开编码。训练前馈几何时可先只做静态渲染，具身任务再保留物理字段。

建议：构建前馈数据集：是。优先级：中高。

### 131. ScanNet v2 *

官方/主要参考：
- https://www.scan-net.org/
- https://github.com/ScanNet/ScanNet

背景说明：ScanNet 由 Stanford、Princeton、Technical University of Munich 等团队发布，通过大规模真实 RGB-D video scans 支撑 3D semantic segmentation、scene reconstruction、CAD retrieval 和室内场景理解。它后来成为 3D scene understanding 最常用基准之一，并直接影响 ScanNet++、开放词汇室内理解和大量 2D/3D 语义标注工作。

原始内容：真实 RGB-D video scans，包含 2.5M views、1500+ scans、3D camera poses、surface reconstruction、instance-level semantic segmentations。

Metadata：

```json
{
  "profile": "indoor_rgbd_semantic_scan",
  "domain": "真实室内扫描",
  "storage": {
    "summary": "sens + ply + json + zip + txt; mesh原生",
    "unit": "scan scene"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "原生",
    "sem2d": "原生导出",
    "pose": "原生",
    "pointcloud": "mesh/点云原生",
    "pointcloud_semantic": "原生",
    "text": "无",
    "gs": "无"
  },
  "convention": "ScanNet sensor/mesh坐标",
  "access": "agreement_required",
  "risk": ".sens解包、2D/3D标注对齐和许可"
}
```

Format/layout: use the official ScanNet page and repository for `.sens`, meshes, aggregation, segments, and 2D label/depth/instance assets.
- https://www.scan-net.org/
- https://github.com/ScanNet/ScanNet

Reader notes: `.sens` is the source sensor stream with color, depth, pose, and intrinsics/extrinsics. Exported `color`, `depth`, `pose`, and `intrinsic` folders are derived artifacts and must be declared as such.

目标转换：先用官方脚本或 adapter 解码 `.sens` 为 RGB/depth/pose；mesh/semantic ply 转 `meshes`；aggregation/segs 转 instance/semantic；按时间邻近生成 pairs。

风险点：`.sens` 解码耗时；pose 可能存在 invalid frames；depth 多为 uint16 mm，需要转米；ScanNet 相机坐标与 mesh/world 坐标需要验证。

建议：构建前馈数据集：是。优先级：中。

### 132. StaticThings3D

官方/主要参考：
- https://github.com/lmb-freiburg/robustmvd/blob/master/rmvd/data/README.md#staticthings3d

背景说明：StaticThings3D 在当前调研中主要通过 University of Freiburg 的 RobustMVD 工程流被使用，源头属于合成多视图/深度训练数据生态。它适合为 robust multi-view depth estimation 提供结构简单、监督明确的合成样本；后续更多作为 MVD/MVS 代码库中的转换后训练/测试格式存在，适合用作 adapter 的小型回归测试，而不是独立发展的通用场景基准。

原始内容：合成/静态物体多视图数据，常用于 MVD/MVS。RobustMVD 转换格式提供 images、poses、intrinsics、depth、invdepth、depth_range 等。

Metadata：

```json
{
  "profile": "synthetic_mvd_regression",
  "domain": "合成多视图场景",
  "storage": {
    "summary": "RobustMVD records + image + depth + pose + intrinsics",
    "unit": "MVD sample"
  },
  "modalities": {
    "rgb": "原生/转换后",
    "depth": "原生/转换后",
    "sem2d": "无",
    "pose": "原生/转换后",
    "pointcloud": "派生(depth反投影)",
    "pointcloud_semantic": "无",
    "text": "无",
    "gs": "无"
  },
  "convention": "RobustMVD schema",
  "access": "public/需下载源确认",
  "risk": "转换目录结构和无效depth处理"
}
```

格式/目录：RobustMVD 统一格式中 sample record 包含：`images` list、`poses` 4×4、`intrinsics`、`keyview_idx`、`depth`、`invdepth`、`depth_range`。depth 单位为米，无效值通常为 0。

目标转换：直接从 RobustMVD format 转为 scene/frame/pair。keyview 与 source views 可对应到 `pairs.jsonl`。depth/invdepth 保留其中一种，优先 depth_m。

风险点：用户表格将其标为 Real，但 StaticThings3D 更常见于 synthetic rendered 数据，需要按具体来源修正 `Real/Synth` 字段。

建议：构建前馈数据集：是。优先级：中。

### 133. uCO3D

官方/主要参考：
- https://github.com/facebookresearch/uco3d

背景说明：uCO3D 由 Meta AI Research 发布，面向真实物体 3D deep learning 和 3D generative AI。它试图补足 CO3D/MVImgNet 在类别多样性、全 360 度覆盖和多模态标注上的不足，并配套 3D Gaussian、caption、depth、point cloud 和 PyTorch3D 工具链；后续已被用于训练 LRM、CAT3D、Instant3D 等真实数据驱动的 3D 模型。

原始内容：object-centric 真实 turntable 视频数据。提供 RGB video、mask video、depth maps、camera poses、point clouds、segmented point clouds、3D Gaussian splats、LVIS 类别、短/长文本描述。

Metadata：

```json
{
  "profile": "object_video_fullmodal",
  "domain": "真实物体中心序列",
  "storage": {
    "summary": "sqlite + mp4 + mkv + h5 + ply + gaussian_splats; point cloud/GS",
    "unit": "category/sequence/frame"
  },
  "modalities": {
    "rgb": "视频原生",
    "depth": "H5原生",
    "sem2d": "mask原生",
    "pose": "原生",
    "pointcloud": "多级点云原生",
    "pointcloud_semantic": "segmented point cloud",
    "text": "caption原生",
    "gs": "原生"
  },
  "convention": "uCO3D sqlite/video时间轴",
  "access": "public",
  "risk": "sqlite索引、视频帧同步和多模态路径解析"
}
```

Format/layout: use the official uCO3D repository for SQLite metadata, set lists, RGB/mask videos, HDF5 depth, point clouds, and Gaussian splats.
- https://github.com/facebookresearch/uco3d

Reader notes: `metadata.sqlite` is the authoritative index. Resolve video frames, depth, pose, mask, caption, point cloud, and GS paths from the schema instead of fixed directory traversal.

目标转换：读取 sqlite 作为主索引；视频解码成 RGB/mask frame；HDF5 读取 depth；pose 转 `T_c2w`；point cloud 与 GS 复制；LVIS/category/caption 写 annotations。适合 object-level multiview/3DGS 训练。

风险点：sqlite schema 需固定版本解析；视频帧、depth_maps.h5 与 metadata frame index 必须严格对齐。数据量大，建议流式转换。

建议：构建前馈数据集：是。优先级：高。

### 134. YCB Benchmark / YCB Object and Model Set

官方/主要参考：
- https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#ycb-benchmarks---object-and-model-set
- https://www.ycbbenchmarks.com/

背景说明：YCB Object and Model Set 由 Yale、Carnegie Mellon University、UC Berkeley 等团队共同提出，目的是为机器人抓取、操作、假肢和康复研究建立统一物体集合和 benchmark protocol。它后来成为 6D pose、grasping、manipulation、仿真资产和真实机器人评测中持续复用的标准物体库。

原始内容：常见物体 object set，提供每个物体多视角 RGB-D/RGB 图像、segmentation masks、camera calibration、texture-mapped 3D mesh。面向机器人抓取/操作。

Metadata：

```json
{
  "profile": "object_asset_rgbd",
  "domain": "物体资产",
  "storage": {
    "summary": "rgbd capture + mesh + glb + json config + calibration; mesh/GLB原生",
    "unit": "object asset/capture sequence"
  },
  "modalities": {
    "rgb": "扫描原生/渲染生成",
    "depth": "扫描原生/渲染生成",
    "sem2d": "渲染生成",
    "pose": "采样/标定生成",
    "pointcloud": "原生扫描/派生(mesh采样)",
    "pointcloud_semantic": "object id映射",
    "text": "object metadata(非caption)",
    "gs": "无"
  },
  "convention": "object frame/renderer frame需统一",
  "access": "public",
  "risk": "原始YCB与Habitat版字段不同"
}
```

格式/目录：YCB 视频/模型包通常以 object 为单位，包含 RGB/RGB-D capture、calibration、mask、mesh。部分版本每个物体有多个相机/turntable 视角。

目标转换：按 object 构建 scene；RGB-D/camera/mask 直接转 frame；mesh 转 `meshes/`；可从 turntable 顺序生成 pairs。若只下载 object model set，则需自行渲染。

风险点：YCB 有多个子集/镜像，文件结构和命名不同。需确认是否包含 pose，若没有物体/相机位姿则只能渲染或用标定轨迹推导。

建议：构建前馈数据集：是。优先级：中。

### 177. 3D-GloBFP / gloBFPr

官方/主要参考：
- https://github.com/billbillbilly/gloBFPr
- https://zenodo.org/records/10570660

背景说明：3D-GloBFP 由遥感与地理信息方向研究团队在 Earth System Science Data 发表，并通过 Zenodo/gloBFPr 分发。它融合多源 Earth Observation 数据和机器学习方法，估计全球单体建筑 footprint-level height；后续被用于城市形态、气候、社会经济和全球 3D 建筑制图研究。本项目中它属于 GIS/遥感格式家族，而不是相机几何数据。

原始内容：全球建筑物 footprint + height 数据。面向遥感/建筑高度提取，不是图像多视图数据。数据通常以 GIS vector/tile 形式提供，包含 building footprint polygon 和 height 属性。

Metadata：

```json
{
  "profile": "geo_polygon_height",
  "domain": "遥感地理建筑",
  "storage": {
    "summary": "shapefile + GeoPackage + GeoJSON + raster; 3D footprint polygons",
    "unit": "geospatial tile/city region"
  },
  "modalities": {
    "rgb": "无",
    "depth": "无",
    "sem2d": "polygon/raster原生",
    "pose": "地理坐标非相机",
    "pointcloud": "无",
    "pointcloud_semantic": "无",
    "text": "属性表(非caption)",
    "gs": "无"
  },
  "convention": "CRS/地理坐标",
  "access": "public",
  "risk": "CRS、tile范围和height单位"
}
```

格式/目录：`gloBFPr` 工具用于搜索、下载和处理全球建筑物 footprint tiles with height；常见格式为 shapefile/GeoPackage/GeoJSON 等 GIS 数据，具体取决于下载接口。

目标转换：不直接构建前馈三维重建图像数据。可转换为 `annotations/geospatial_buildings.geojson`，或与卫星影像/DEM 结合生成遥感 2.5D 训练数据。

风险点：坐标系通常是 WGS84/投影坐标，不能直接与相机 SE(3) 混用。高度可能是模型估计值，不是激光/测量真值。

建议：构建前馈数据集：否，除非目标是遥感高度/地图任务。优先级：低。

### 188. ScanNet++

官方/主要参考：
- https://scannetpp.mlsg.cit.tum.de/scannetpp/
- https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation
- https://github.com/scannetpp/scannetpp

背景说明：ScanNet++ 由 Technical University of Munich / MLSG 等团队在 ScanNet 体系上继续建设，使用高端激光扫描、DSLR 和 iPhone RGB-D 数据提升室内场景几何与图像质量。它面向 high-fidelity reconstruction、NVS 和语义理解，后续已发布 v2，场景规模扩展到 1000+，并成为 NeRF/3DGS、高保真室内重建和多传感器配准的重要 benchmark。

原始内容：高保真真实室内场景，提供 laser scans、DSLR images、iPhone RGB-D、mesh、semantic/instance annotations、point clouds、panocam 等。

Metadata：

```json
{
  "profile": "indoor_multisensor_recon",
  "domain": "真实室内扫描",
  "storage": {
    "summary": "ply + json + COLMAP txt + Nerfstudio json + mkv + bin + png; mesh/COLMAP/Nerfstudio",
    "unit": "scene/sensor stream/frame"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "原生",
    "sem2d": "原生/可导出",
    "pose": "多约定原生",
    "pointcloud": "原生",
    "pointcloud_semantic": "原生",
    "text": "无",
    "gs": "无"
  },
  "convention": "COLMAP/OpenCV + Nerfstudio/OpenGL + iPhone轨迹",
  "access": "agreement_required/需确认",
  "risk": "多相机约定并存、scene graph导出和对齐"
}
```

Format/layout: use the official ScanNet++ documentation for splits, metadata, scans, DSLR, iPhone, panocam, COLMAP, Nerfstudio, and RGB-D assets.
- https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation
- https://github.com/scannetpp/scannetpp

Reader notes: DSLR/COLMAP, Nerfstudio, iPhone, and mesh assets use different conventions. Parse each stream according to the official docs and keep coordinate conversions explicit.

目标转换：优先使用 DSLR undistorted + COLMAP metric poses 构建高质量 RGB+pose；用 mesh 渲染 high-res depth；iPhone RGB-D 可作为低分辨率 depth source；semantic mesh 可投影到 2D 生成 semantic/instance masks。场景语义图可由 segments/object annotations 构建。

风险点：DSLR fisheye/OpenCV_FISHEYE、Nerfstudio OpenGL convention、iPhone ARKit right-handed +Z forward 三种坐标需分开处理。NVS test split 不含 3D 信息，不能用于深度/语义真值。

建议：构建前馈数据集：是。优先级：高。

### 244. InteriorAgent

官方/主要参考：
- https://huggingface.co/datasets/spatialverse/InteriorAgent

背景说明：InteriorAgent 由 SpatialVerse Research Team / Manycore Tech Inc. 发布，提供 Isaac Sim 可用的交互式 USD 室内场景。它主要面向 synthetic data generation、导航和物理仿真，后续进入 SpatialVerse/SAGE-3D 资产生态，并被 NVIDIA Isaac Sim 第三方 USD 资产文档引用为可用仿真数据源。

原始内容：高质量 USD/USDa 室内场景资产，面向 NVIDIA Isaac Sim。提供 materials、meshes、lighting、floorplan、room metadata，适用于导航、操作、布局理解。

Metadata：

```json
{
  "profile": "usd_scene_asset",
  "domain": "合成室内仿真资产",
  "storage": {
    "summary": "usd + material files + scene description + physics geometry; USD原生",
    "unit": "USD scene asset"
  },
  "modalities": {
    "rgb": "渲染生成",
    "depth": "渲染生成",
    "sem2d": "metadata渲染",
    "pose": "采样生成",
    "pointcloud": "派生(资产采样)",
    "pointcloud_semantic": "metadata映射",
    "text": "scene description原生",
    "gs": "无"
  },
  "convention": "USD/Isaac Sim坐标",
  "access": "public/需确认",
  "risk": "USD材质/物理属性和仿真版本兼容"
}
```

Format/layout: use the Hugging Face dataset page for InteriorAgent USD/USDA scenes, materials, meshes, and room metadata.
- https://huggingface.co/datasets/spatialverse/InteriorAgent

Reader notes: this is an Isaac Sim/USD scene-asset dataset, not pre-rendered RGB-D frames. Generate RGB/depth/semantic/pose through a renderer after loading official assets.

目标转换：使用 Isaac Sim/Omniverse RenderAgent 加载 `.usda`，采样 camera，渲染 RGB-D/semantic；`rooms.json` 写 `scene_graph.json` 或 `floorplan.json`；USD 原文件复制到 `meshes/` 或 `sim_assets/`。

风险点：不是现成 RGB-D dataset；必须渲染。Isaac 坐标系与 OpenCV camera/world 需要转换。

建议：构建前馈数据集：是。优先级：中高。

### 245. InteriorGS / SAGE-3D InteriorGS USDZ

官方/主要参考：
- https://github.com/manycore-research/InteriorGS
- https://huggingface.co/datasets/spatialverse/SAGE-3D_InteriorGS_usdz

背景说明：InteriorGS/SAGE-3D InteriorGS 来自 SpatialVerse/Manycore 及 SAGE-3D 相关研究生态，关注 embodied navigation 和 XR 中的 3D Gaussian 场景。它为 GS 场景补充 object-level semantic grounding、occupancy 和可执行/碰撞信息，推动 3DGS 从视觉渲染表示转向带语义与物理接口的可导航环境；本项目应把它视为独立几何载体，而不是 mesh 的附属格式。

原始内容：室内 3D Gaussian Splatting 场景，带语义标注和空间占用信息。用户给出的 HF 数据是将 InteriorGS compressed PLY 转换为 USDZ 的版本，面向 Isaac Sim/Omniverse。

Metadata：

```json
{
  "profile": "gaussian_indoor_scene",
  "domain": "室内Gaussian场景",
  "storage": {
    "summary": "usdz + ply gaussian + json + png; USDZ/structure metadata",
    "unit": "Gaussian scene"
  },
  "modalities": {
    "rgb": "预览/可渲染",
    "depth": "渲染生成",
    "sem2d": "labels渲染",
    "pose": "采样生成",
    "pointcloud": "派生(GS转换可选)",
    "pointcloud_semantic": "object labels映射",
    "text": "无",
    "gs": "原生"
  },
  "convention": "3DGS/USdz坐标需确认",
  "access": "gated/需接受条件",
  "risk": "GS渲染器、occupancy/labels对齐和访问条件"
}
```

Format/layout: use the official repository and Hugging Face dataset page for InteriorGS / SAGE-3D InteriorGS USDZ assets.
- https://github.com/manycore-research/InteriorGS
- https://huggingface.co/datasets/spatialverse/SAGE-3D_InteriorGS_usdz

Reader notes: treat USDZ, Gaussian splats, metadata, and previews as official asset types from the file pages. Do not maintain a static local tree here.

目标转换：若目标支持 GS，直接复制 `.usdz` 到 `gs/` 并记录 metadata；若目标为 RGB-D/pose，则通过 Isaac Sim/3DGRUT renderer 渲染多视角 RGB、depth、semantic/occupancy。GS 本身不等同 mesh，物理碰撞需配合 collision mesh。

风险点：3DGS 渲染与真实 mesh depth/normal 不完全等价；semantic/occupancy 字段是否随 USDZ 一起保留需样本确认。

建议：构建前馈数据集：是。优先级：中高。

### 246. Tabletop_Scenes / TabletopGen-Assets

官方/主要参考：
- https://huggingface.co/datasets/xinjue1/TabletopGen-Assets/tree/main/scene_gallery

背景说明：TabletopGen-Assets 由 TabletopGen 相关作者通过 Hugging Face 发布，支持从文本或单图生成 instance-level interactive tabletop scenes，并配套 Isaac Sim pick-and-place manipulation demo。它目前更像论文/项目资产包而非成熟 benchmark，但可作为小尺度桌面操作和生成式场景 adapter 的早期样例。

原始内容：TabletopGen 生成的预制 3D 桌面场景和机器人 manipulation demo assets，面向文本/单图到可交互 3D tabletop scene。

Metadata：

```json
{
  "profile": "tabletop_asset_scene",
  "domain": "桌面操作场景",
  "storage": {
    "summary": "glb + Isaac Sim demo code + asset metadata; GLB原生",
    "unit": "tabletop scene asset"
  },
  "modalities": {
    "rgb": "渲染生成",
    "depth": "渲染生成",
    "sem2d": "渲染生成",
    "pose": "采样生成",
    "pointcloud": "派生(mesh采样)",
    "pointcloud_semantic": "asset id映射",
    "text": "任务/配置元数据(非caption)",
    "gs": "无"
  },
  "convention": "Isaac Sim/GLB坐标",
  "access": "public",
  "risk": "场景资产与manipulation demo代码分离"
}
```

Format/layout: use the Hugging Face dataset page for TabletopGen-Assets scene gallery and GLB scene assets.
- https://huggingface.co/datasets/xinjue1/TabletopGen-Assets/tree/main/scene_gallery

Reader notes: this is GLB/asset-first data. Load official scene assets and render RGB-D/normal/mask/pose with Blender or another renderer.

目标转换：作为 GLB scene-first 数据集，需要 Blender/Isaac RenderAgent 渲染多视图 RGB-D/normal/mask，并保留 GLB scene 到 `meshes/`。适合构建桌面物体密集布局的前馈数据。

风险点：当前公开资产行数较少，规模有限；语义标注、物理属性、pose 是否内嵌在 GLB 中需样本确认。

建议：构建前馈数据集：是。优先级：中。

### 247. Maya **

官方/主要参考：用户表格未提供 URL。

背景说明：当前表格中的 “Maya” 未能唯一对应到一个有明确发布团队、论文和官网的公开 benchmark，更可能指 Autodesk Maya/DCC 资产来源或内部整理场景。若作为数据源，它应被理解为人工建模或 DCC 导出的室内外 mesh/scene 资产；在正式适配前需要补齐数据来源、授权、目录样例和导出规范，否则不宜按公开数据集归因。

原始内容：按用户描述，是室内外 mesh 场景，包含奇特场景，如宫殿、货船等。现阶段它不是可唯一识别的公开 benchmark，只能作为自定义 DCC 场景源占位。

Metadata：

```json
{
  "profile": "custom_dcc_scene_placeholder",
  "domain": "自定义三维场景占位",
  "storage": {
    "summary": "ma/mb + mesh exports + textures + custom metadata; DCC/mesh原生",
    "unit": "custom DCC scene"
  },
  "modalities": {
    "rgb": "渲染生成",
    "depth": "渲染生成",
    "sem2d": "对象ID/材质渲染",
    "pose": "采样生成",
    "pointcloud": "派生(mesh采样)",
    "pointcloud_semantic": "对象层级映射",
    "text": "需人工补充",
    "gs": "无/需转换"
  },
  "convention": "Maya坐标/单位需人工确认",
  "access": "unknown",
  "risk": "不是可唯一识别的公开benchmark；缺官方数据集身份和schema"
}
```

格式/目录：无可确认的官方数据集组织格式。当前条目只写作 “Maya”，未提供官方 URL，也不能唯一对应某个公开 benchmark。不得把 Autodesk Maya 的 `.ma/.mb` 资产格式、用户内部资产库或任意 DCC 工程目录写成官方数据集目录。

```text
Maya/
└── <no confirmed official dataset layout>
```

若后续用户提供官方页面、下载包或样本目录，应在本条目中记录该来源的实际组织方式；在此之前只能作为用户自定义 DCC/mesh 资产占位处理。

目标转换：如果是 `.ma/.mb`，需要 Maya/Blender/Assimp 可读转换链；如果是 `.fbx/.obj/.glb`，可走通用 MeshAdapter。转换为前馈数据集必须渲染 RGB-D/pose。

风险点：缺少 URL 和样本，不能确定 license、目录结构、材质贴图路径、坐标单位、是否含语义。

建议：构建前馈数据集：是，但需要用户提供来源或样本。优先级：中高（按用户表格）。

### 283. OpenSatMap

官方/主要参考：
- https://opensatmap.github.io/
- https://huggingface.co/datasets/z-hb/OpenSatMap

背景说明：OpenSatMap 由中国科学院自动化研究所、腾讯地图、北京邮电大学等团队发布，关注从高分辨率卫星图中提取细粒度道路结构。它最初服务大规模地图构建和自动驾驶 HD map 更新，后续进入 NeurIPS Datasets and Benchmarks 体系，并作为 satellite-based map construction、道路矢量化和遥感地图理解的公开 benchmark。

原始内容：高分辨率卫星图像 + 细粒度 instance-level road structure annotations。覆盖多国多城市，并与 nuScenes/Argoverse2 等自动驾驶区域有对齐关系。

Metadata：

```json
{
  "profile": "geo_vector_map",
  "domain": "遥感道路地图",
  "storage": {
    "summary": "satellite image + vector polyline + mask + attribute table; vector polylines",
    "unit": "map tile/line instance"
  },
  "modalities": {
    "rgb": "卫星图原生",
    "depth": "无",
    "sem2d": "polyline/attribute原生",
    "pose": "地理配准非相机",
    "pointcloud": "无",
    "pointcloud_semantic": "无",
    "text": "属性表(非caption)",
    "gs": "无"
  },
  "convention": "tile level/CRS/像素地理映射",
  "access": "public",
  "risk": "矢量属性、mask栅格化和地图坐标对齐"
}
```

格式/目录：官方说明包含 OpenSatMap19 与 OpenSatMap20：level 19 约 0.3 m/pixel，level 20 约 0.15 m/pixel。标注对象包括 lane line、curb、virtual line，并提供八类属性，如颜色、线型、线数、特殊功能、边界、遮挡、清晰度等。标注以 vectorized polylines 表示，同时可能提供 mask。

目标转换：不属于普通相机多视图 3D 数据，但可转为遥感地图任务格式。RGB satellite image 写入 `rgb/overhead`；polyline 写入 `annotations/map_polylines.geojson/json`；mask 写入 `semantic/`。若与 nuScenes/Argoverse 对齐，可建立 global map prior。

风险点：坐标基准、tile origin、像素坐标到地理坐标映射需确认。不能直接生成 camera pose/depth。

建议：构建前馈数据集：否，除非目标是地图/遥感前馈模型。优先级：中。

### 284. SEED-MAP / SatelliteLaneDataset2024

官方/主要参考：
- https://github.com/rilab314/SatelliteLaneDataset2024

背景说明：SEED-MAP / SatelliteLaneDataset2024 由 RILAB 等团队以 GitHub 数据与工具形式发布，面向卫星图中的道路、车道线和路面符号检测。它提供从 shapefile/vector 到 COCO/ADE20K 训练格式的构建链路，目前更偏早期开放数据工程，适合作为遥感矢量标注到语义分割格式转换的验证样例。

原始内容：韩国首尔/仁川卫星道路标注数据，包含 image-label pairs，也提供 COCO form 和 ADE20K form。用户表格说明包含大量车道线和路面符号标注。

Metadata：

```json
{
  "profile": "geo_road_segmentation",
  "domain": "遥感道路地图",
  "storage": {
    "summary": "image + COCO json + ADE20K mask + shapefile + coordinate list; shapefile/vector源",
    "unit": "satellite image/label pair"
  },
  "modalities": {
    "rgb": "image原生",
    "depth": "无",
    "sem2d": "COCO/ADE20K原生",
    "pose": "地理配准非相机",
    "pointcloud": "无",
    "pointcloud_semantic": "无",
    "text": "属性/类别表(非caption)",
    "gs": "无"
  },
  "convention": "NGII shapefile到image alignment",
  "access": "public",
  "risk": "矢量到栅格转换链路和类别映射"
}
```

Format/layout: use the official repository for image/label, COCO, and ADE20K-style satellite lane data organization.
- https://github.com/rilab314/SatelliteLaneDataset2024

Reader notes: this is remote-sensing road/lane annotation data, not perspective multi-view 3D data. Preserve resolution, CRS, and task format metadata when converting.

目标转换：优先读取 COCO/ADE20K 格式，因为 schema 标准。卫星图写 `rgb/overhead`，label/mask 写 `semantic/`，COCO annotations 写 `annotations/coco.json`。若需要 polyline，需要从原始 label 或 NGII HD map 数据回溯。

风险点：COCO/ADE20K 转换版本可能丢失 lane instance/polyline 几何细节。必须确认原始 label 的精度。

建议：构建前馈数据集：否，除非目标是地图/遥感语义。优先级：中。

### 293. Articraft-10K **

官方/主要参考：
- https://articraft3d.github.io/
- https://github.com/mattzh72/articraft
- https://arxiv.org/html/2605.15187v1

背景说明：Articraft-10K 由 Articraft agentic articulated asset generation 系统的研究团队发布，直接针对 articulated 3D asset 稀缺问题。它为机器人仿真、VR 和可动对象生成提供 10K+ URDF/mesh/semantic part/joint 样本；作为 2026 年左右的新数据集，它的主要价值在于验证 agentic asset generation 和 articulated-object model training，数据质量与通用 benchmark 地位仍需后续社区验证。

原始内容：Articraft 是 agentic articulated 3D asset generation 系统；Articraft-10K 包含 10K+ articulated 3D assets，覆盖日常物体类别。每个资产由程序生成，输出 URDF、3D meshes、semantic parts、articulated joints、joint axes 与 motion ranges。

Metadata：

```json
{
  "profile": "articulated_asset_bank",
  "domain": "可动3D物体",
  "storage": {
    "summary": "urdf + python record + mesh + metadata + visualization asset; URDF原生",
    "unit": "articulated object record"
  },
  "modalities": {
    "rgb": "渲染生成",
    "depth": "渲染生成",
    "sem2d": "渲染生成",
    "pose": "采样生成",
    "pointcloud": "派生(URDF/mesh采样)",
    "pointcloud_semantic": "link/joint映射",
    "text": "类别/生成元数据(非caption)",
    "gs": "无"
  },
  "convention": "URDF joint/link frame",
  "access": "public/需确认",
  "risk": "关节限制、collision mesh和材质层级"
}
```

Format/layout: use the official Articraft page and repository for code-first records, Git LFS hydration, `data/records/**`, and `model.py` assets.
- https://articraft3d.github.io/
- https://github.com/mattzh72/articraft

Reader notes: each record centers on executable or parseable asset-generation code plus associated mesh/URDF/metadata. Pin repository version and hydration state before parsing.

目标转换：若目标是仿真/具身任务，应保留 URDF、关节、part semantics 到 `annotations/articulation.json`。若目标是前馈三维训练，需要按关节状态采样多个 articulation configurations，并渲染多视图 RGB-D/mask/part segmentation。

风险点：安全性重要：不要执行不可信 `model.py`。必须在 sandbox/container 中运行，禁用网络和危险系统调用。关节状态采样会影响同一 object 的几何一致性。

建议：构建前馈数据集：是。优先级：中高。

### 294. SAGE-10k **

官方/主要参考：
- https://huggingface.co/datasets/nvidia/SAGE-10k
- https://github.com/NVlabs/sage
- https://research.nvidia.com/labs/dir/sage/

背景说明：SAGE-10k 由 NVIDIA 发布，来源于 “SAGE: Scalable Agentic 3D Scene Generation for Embodied AI” 管线。它为 embodied AI 和 physics-based simulation 生成 simulation-ready interactive indoor scenes，后续进入 Hugging Face 分发，作为 agentic scene generation、Isaac/物理仿真和世界模型训练的早期大规模室内场景资产库。

原始内容：大规模交互式室内场景数据，包含 10,000 diverse scenes、50 room types/styles、565K generated 3D objects。面向 Isaac Sim、embodied AI、physics-based simulation。

Metadata：

```json
{
  "profile": "interactive_indoor_asset",
  "domain": "交互式室内场景",
  "storage": {
    "summary": "scene config + object assets + materials + json layout + preview; objects/materials/layout原生",
    "unit": "simulation-ready scene"
  },
  "modalities": {
    "rgb": "preview/渲染生成",
    "depth": "渲染生成",
    "sem2d": "asset id渲染",
    "pose": "采样生成",
    "pointcloud": "派生(mesh采样)",
    "pointcloud_semantic": "object metadata映射",
    "text": "layout metadata(非caption)",
    "gs": "无"
  },
  "convention": "Isaac Sim/scene layout坐标",
  "access": "public/需确认",
  "risk": "assets/materials/layout路径闭环和仿真版本"
}
```

Format/layout: use the official SAGE repository, research page, and Hugging Face dataset page for code assets and scene zip distribution.
- https://huggingface.co/datasets/nvidia/SAGE-10k
- https://github.com/NVlabs/sage
- https://research.nvidia.com/labs/dir/sage/

Reader notes: Hugging Face scenes are distributed as `*_layout_*.zip` packages. Code repository assets/scripts are not the dataset body; parse official packages and metadata together.

目标转换：和 InteriorAgent 类似，属于 simulation scene-first 数据集。用 Isaac Sim RenderAgent 渲染 RGB-D/semantic/instance/normal/pose；保留 scene config、object metadata、room/task 信息。若包含机器人动作数据，可写入 `annotations/embodied_tasks.jsonl`。

风险点：数据格式可能依赖 Isaac Sim 版本；大规模渲染成本高。需要明确相机采样策略和物理碰撞过滤。

建议：构建前馈数据集：是。优先级：中高。

### 10146. KITTI ***

官方/主要参考：
- https://www.cvlibs.net/datasets/kitti/
- https://registry.opendata.aws/kitti/

背景说明：KITTI Vision Benchmark Suite 由 Karlsruhe Institute of Technology 和 Toyota Technological Institute at Chicago 团队发布，是自动驾驶计算机视觉早期最有影响力的公开基准之一。它最初覆盖 stereo、optical flow、visual odometry、3D detection/tracking 和 road estimation，后续长期作为自动驾驶几何基准的事实标准，并推动 KITTI-360、SemanticKITTI、nuScenes、Waymo 等更大规模数据集的发展。

原始内容：真实自动驾驶数据，包含 stereo cameras、Velodyne LiDAR、GPS/IMU localization。任务覆盖 stereo、optical flow、visual odometry、3D object detection/tracking、road/semantic 等。

Metadata：

```json
{
  "profile": "driving_stereo_lidar_sequence",
  "domain": "真实道路驾驶",
  "storage": {
    "summary": "png + bin + txt + xml",
    "unit": "drive/frame/sensor"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "派生(LiDAR投影)",
    "sem2d": "检测/分割任务可用",
    "pose": "GPS/IMU原生",
    "pointcloud": "Velodyne原生",
    "pointcloud_semantic": "3D boxes/任务标注",
    "text": "无",
    "gs": "无"
  },
  "convention": "calib/oxts/camera/Velodyne坐标链",
  "access": "public",
  "risk": "benchmark版本差异、calib解析和LiDAR投影稀疏性"
}
```

Format/layout: KITTI layouts differ by benchmark. Use official benchmark pages for Visual Odometry / SLAM, raw data, stereo, detection, tracking, and related downloads.
- https://www.cvlibs.net/datasets/kitti/
- https://www.cvlibs.net/datasets/kitti/eval_odometry.php

Reader notes: the current KITTI odometry loader targets odometry-style `sequences/<seq>/calib.txt`, image folders, and `poses/<seq>.txt`. Raw KITTI/OXTS and other benchmarks require separate roots and parsers.

目标转换：优先选择 odometry/raw 子集。相机内参从 calib 读取；pose 从 odometry poses 或 OXTS GPS/IMU 推导；LiDAR 转 point cloud/depth projection；stereo pairs 直接生成 `pair_type=stereo`，temporal pairs 按帧邻近生成。

风险点：KITTI 不同任务文件结构差异很大；raw data 的 OXTS pose 需地理坐标转换到局部 ENU/metric world。

建议：构建前馈数据集：是。优先级：高。

### 10147. KITTI-360 ***

官方/主要参考：
- https://www.cvlibs.net/datasets/kitti-360/

背景说明：KITTI-360 由 Karlsruhe/Tübingen 自动驾驶视觉研究团队在 KITTI 基础上发布，面向 urban scene understanding in 2D and 3D。它提供更长距离、更丰富传感器和密集语义/实例标注，后来成为连接自动驾驶、图形学、机器人和场景重建的长序列 benchmark，常用于语义地图、NVS 和持续重建研究。

原始内容：大规模自动驾驶数据，包含 320K+ images、100K laser scans、73.7 km driving distance、准确地理定位、2D/3D dense semantic & instance annotations。

Metadata：

```json
{
  "profile": "driving_multicamera_mapping",
  "domain": "真实道路建图",
  "storage": {
    "summary": "png + bin + txt + xml + pose + semantic labels",
    "unit": "sequence/frame/sensor"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "派生(LiDAR/SICK)",
    "sem2d": "原生",
    "pose": "原生",
    "pointcloud": "原生",
    "pointcloud_semantic": "3D语义/标注可用",
    "text": "无",
    "gs": "无"
  },
  "convention": "perspective/fisheye/Velodyne/SICK多坐标",
  "access": "public/需注册确认",
  "risk": "多相机模型、跨帧实例ID和语义格式"
}
```

Format/layout: use the official KITTI-360 documentation as the source of truth; do not duplicate the full directory tree here.
- https://www.cvlibs.net/datasets/kitti-360/documentation.php

Reader notes: the current loader uses perspective images, calibration, and poses through `roots.calibration`, `roots.images`, and `roots.poses`. Velodyne, SICK, 2D/3D semantics, and 3D boxes should be explicit optional roots or future loader fields.

目标转换：适合高优先级构建多相机/长序列前馈数据。读取 calibration 和 poses，生成每个 camera 的 `T_c2w`；LiDAR/semantic point cloud 写入 `point_clouds`；2D/3D semantics 写 annotations；temporal、stereo、loop pairs 可全部构建。

风险点：perspective 与 fisheye 相机模型不同；2D/3D 标注跨目录关联复杂。长序列转换必须支持分块和断点续跑。

建议：构建前馈数据集：是。优先级：高。

### 10148. Wayve / WayveScenes101 ***

官方/主要参考：
- https://wayve.ai/science/wayvescenes101/
- https://github.com/wayveai/wayve_scenes

背景说明：WayveScenes101 由 Wayve 研究团队发布，面向自动驾驶中的 novel view synthesis 和 scene reconstruction。它提供真实多相机场景与 held-out evaluation camera，代表自动驾驶数据从感知检测 benchmark 向可生成、可重建世界模型数据演化，适合验证 camera-only driving NVS adapter。

原始内容：真实自动驾驶 NVS/scene reconstruction 数据集，包含 101 scenes，每个 scene 20 秒，5 个 time-synchronised cameras，10 FPS，共约 101,000 images，并提供 camera poses、held-out evaluation camera、scene-level metadata。

Metadata：

```json
{
  "profile": "driving_nvs_sequence",
  "domain": "真实道路驾驶",
  "storage": {
    "summary": "images + camera poses + metadata + Nerfstudio-style records",
    "unit": "scene/camera/frame"
  },
  "modalities": {
    "rgb": "多相机原生",
    "depth": "无/可估计",
    "sem2d": "无",
    "pose": "原生",
    "pointcloud": "无",
    "pointcloud_semantic": "无",
    "text": "无",
    "gs": "可训练生成/非原生"
  },
  "convention": "Nerfstudio/NVS camera schema需样本确认",
  "access": "public/需确认",
  "risk": "无GT几何、仅适合NVS/重建评测"
}
```

Format/layout: use the official WayveScenes101 page and repository for download, COLMAP calibration, camera names, baselines, and scene metadata.
- https://wayve.ai/science/wayvescenes101/
- https://github.com/wayveai/wayve_scenes

Reader notes: official WayveScenes101 describes a COLMAP/metadata workflow. The current `WayveScenesPi3XDataset` reads one Nerfstudio-style `transforms.json` per scene, which is a different input format.

目标转换：直接转为 driving multiview NVS/front-feed 数据。5 相机同步帧可构建 cross-camera pairs；10 FPS temporal frames 可构建 temporal pairs；held-out camera 应进入 test/novel-view split，不参与训练。

风险点：用户表格写 “wayve” 不够明确，此处按 WayveScenes101 处理；若用户实际指 Wayve 其他内部/公开数据，需要重查。pose convention 可能与 Nerfstudio/OpenGL 相关，需转换。

建议：构建前馈数据集：是。优先级：高。

### 10149. Waymo Open Dataset ***

官方/主要参考：
- https://waymo.com/open/
- https://github.com/waymo-research/waymo-open-dataset

背景说明：Waymo Open Dataset 由 Waymo 团队公开发布，初衷是向研究社区开放高质量多传感器自动驾驶数据，支持 perception、motion forecasting 和端到端驾驶研究。它从 2019 年 Perception 数据扩展到 Motion、End-to-End Driving、object assets、Parquet component format 和年度挑战，已经成为自动驾驶工程适配中最复杂的公开格式家族之一。

原始内容：真实自动驾驶大规模多传感器数据。官方仓库说明包含 Perception dataset、Motion dataset、End-To-End Driving dataset；Perception 提供高分辨率传感器数据和多任务 labels，Motion 提供 103,354 scenes 的 object trajectories 和 3D maps。

Metadata：

```json
{
  "profile": "driving_multisensor_sequence",
  "domain": "真实道路驾驶",
  "storage": {
    "summary": "TFRecord + Protocol Buffer + v2 component tables + jpg + range image/point cloud",
    "unit": "segment/frame/sensor"
  },
  "modalities": {
    "rgb": "原生",
    "depth": "派生(LiDAR投影)",
    "sem2d": "2D框/panoptic可用",
    "pose": "vehicle/sensor原生",
    "pointcloud": "LiDAR原生",
    "pointcloud_semantic": "3D框/分割任务可用",
    "text": "无",
    "gs": "无"
  },
  "convention": "classic=TFRecord/Proto；v2=component tables；vehicle/global/sensor坐标链",
  "access": "public/需注册",
  "risk": "版本差异、Proto读取、组件表读取和KITTI-like预转换信息损失"
}
```

Format/layout: use Waymo Open Dataset official docs and repository for Perception, Motion, End-to-End, and v2 component formats.
- https://waymo.com/open/
- https://github.com/waymo-research/waymo-open-dataset

Reader notes: classic Perception data is sharded TFRecord plus protocol buffers; v2 adds component/table formats. The current `WaymoKittiPi3XDataset` reads converted KITTI-style data and is not a native Waymo raw-format loader.

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

### 4.5 关键训练字段缺失时的统一补全策略

Pi3X 类前馈几何训练不能把缺失字段伪装成真值。特别是 depth：如果源数据没有提供可监督的 depth，不允许写入常数图、全 1 图、全 0 图或任意 placeholder 当作 `depthmap`。

每个 adapter 必须记录以下来源字段：

```json
{
  "depth_source": "gt_dense|gt_sparse|rendered|projected_lidar_sparse|mvs_pseudo|mono_pseudo|missing",
  "pose_source": "gt|sensor|colmap|slam|rendered|estimated|missing",
  "intrinsics_source": "gt|metadata|colmap|exif_estimated|assumed",
  "pseudo_label": false,
  "valid_mask_required": true
}
```

补全方案按数据源能力分组，不按单个数据集重复编写：

| 组别 | 已有数据源 | 示例 | 补全方案 | 训练准入 |
| --- | --- | --- | --- | --- |
| A | RGB + intrinsics + pose + dense depth | BlendedMVS；ScanNet/ARKitScenes/Hypersim 的 RGB-D 子集；uCO3D depth 子集 | 直接读取 depth，统一单位和 depth definition，生成 `valid_mask`，过滤 invalid frame | 可作为 GT depth 训练 |
| B | RGB + intrinsics + pose + LiDAR/SICK/radar/point cloud，但无 dense image depth | KITTI；KITTI-360；nuScenes；Waymo Perception | 按官方 calibration 和 pose chain 将点云投影到目标相机，生成 sparse depth | 仅用于支持 sparse depth supervision 的训练；dense GT depth 训练必须拒绝或先离线补全 |
| C | RGB + pose/calibration + mesh 或 semantic mesh，但无传感器 depth | Replica；HM3D；Matterport3D mesh release；ScanNet++ mesh/DSLR 组合 | 用 mesh renderer 从已有或采样相机位姿渲染 z-depth、normal、semantic/instance | 可作为 rendered geometry supervision；不能标记为真实传感器 depth |
| D | mesh/USD/GLB/URDF/3DGS/scene config，无现成 RGB-D 帧和相机轨迹 | 3D-FRONT；ReplicaCAD；InteriorAgent；InteriorGS；Tabletop；Objaverse-XL；Articraft-10K；SAGE-10k；Maya-like assets | 定义 `render_policy`，采样相机轨迹，渲染 RGB/depth/normal/mask/pose | 作为合成/渲染训练数据；不得混同真实采集数据 |
| E | RGB/video，可估计 intrinsics/pose，但无官方几何真值 | BDD100K；OpenVid-1M；部分 MVImgNet/MegaDepth 镜像；用户自有视频 | 抽帧后运行 COLMAP/VGGSfM/DROID-SLAM/MASt3R-SLAM/VGGT-Long 估计 pose，再用 MVS 或 metric depth model 生成 pseudo depth | 只用于弱监督、预训练或 teacher-student；不能作为 GT 指标 |
| F | COLMAP/Nerfstudio/SfM sparse model，但 dense depth 缺失或不稳定 | DL3DV-10K；WayveScenes101 COLMAP output；CO3D/MVImgNet 的部分处理包 | 读取 camera/pose/sparse point cloud；可用 MVS densification 或 depth model 生成 pseudo dense depth | pose 可按来源使用；depth 必须按 sparse/pseudo 区分训练权重 |
| G | 只有 2D 语义/地图/遥感标注，不属于透视多视图几何 | OpenSatMap；SEED-MAP；3D-GloBFP；BDD100K 的纯 2D task 用法 | 不补造 Pi3X 几何训练字段；只转换为对应 2D/地图任务或外观/语义辅助数据 | 不进入需要 depth/pose 的 Pi3X 几何训练 |

关键来源标记：

| 场景 | 必须写入 |
| --- | --- |
| 点云投影稀疏深度 | `depth_source=projected_lidar_sparse`；`valid_mask` 只覆盖有投影点的像素；记录传感器、时间同步和遮挡处理 |
| 渲染深度 | `depth_source=rendered`；`pose_source=rendered`；记录 renderer、mesh version、near/far、采样策略和过滤条件 |
| 伪几何 | `pseudo_label=true`；`pose_source=estimated|colmap|slam`；`depth_source=mvs_pseudo|mono_pseudo`；写入 confidence |
| 无几何 | `depth_source=missing`；`pose_source=missing`；`requires_depth=true` 的训练配置必须拒绝 |

执行规则：

1. `requires_depth=true` 且只有 sparse depth 时，训练代码必须显式支持 sparse `valid_mask`，否则拒绝该数据集。
2. 需要 dense depth 但只有 Group E/F 的可估计来源时，必须先运行离线补全流程，并写出 `pseudo_label=true`。
3. dataloader 不负责在 `__getitem__` 中临时运行 SfM、MVS、SLAM 或 renderer；这些属于 preprocessing/rendering pipeline。
4. validator 必须检查 `depth_source`、`pose_source`、`valid_mask` 和 `pseudo_label`，不能只检查字段是否存在。
5. placeholder depth 只能用于调试 shape，不得进入训练样本；测试 fixture 也必须用 `depth_source=missing` 或显式 fake 标记。

## 5. 优先级表

以下数据集需要下载样本或用户补充信息后才能写稳定 adapter：

```text
Maya：无官方 URL，无法确认数据格式。
MegaDepth：官方定义明确，但常用文件结构依赖镜像/预处理版本。
MVImgNet / MVImgNet 2.0：模态明确，但具体下载包目录需样本确认。
YCB：不同子集结构差异大，需确认用户使用 object model set 还是 RGB-D benchmark。
InteriorGS：原始 compressed PLY 与 USDZ 转换版字段不同，需确认是否包含语义/occupancy。
Wayve：本 Skill 按 WayveScenes101 处理；如用户指其他 Wayve 数据，需要重新建 profile。
```

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


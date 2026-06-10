# Uni3D DataSkill

用于将多源 3D 数据集接入 Pi3X 兼容 PyTorch dataloader 的工具。

English documentation: [README.md](README.md)

## 当前状态

当前流程：

```text
raw dataset root
  -> Pi3X-compatible Dataset implementation
  -> PyTorch DataLoader
  -> validation report
```

已实现的 loader 包括 KITTI-360、KITTI odometry、nuScenes table layout、
WayveScenes-style transforms、Waymo KITTI-style converted layout 和 BlendedMVG。

## 环境要求

- Python 3.10 或更新版本。
- 本地 conda 或 Python 环境，并安装 PyTorch 相关依赖。
- Pi3 training 分支固定放在 `thirdparty/Pi3`。

## 安装

克隆本仓库并创建或激活 Python 环境：

```bash
git clone <this-repository-url>
cd Uni3D-DataSkill
conda activate <env-name>
```

以 editable 模式安装本仓库：

```bash
python -m pip install -e .
```

将 Pi3 克隆到固定 third-party 路径：

```bash
git clone https://github.com/yyfz/Pi3.git thirdparty/Pi3
cd thirdparty/Pi3
git checkout training
```

安装 Pi3 依赖：

```bash
python -m pip install -r thirdparty/Pi3/requirements.txt
```

## 数据集配置

使用本地 JSON 配置文件维护 dataset label 和数据集根目录的映射。可以从示例文件开始：

```bash
cp dataset_config.example.json dataset_config.local.json
```

示例：

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

`root` 可以是本地磁盘路径，或当前 Python 进程可读取的 Windows UNC 共享路径。

```json
{
  "datasets": [
    {
      "label": "kitti360_local",
      "dataset": "kitti360",
      "root": "D:/datasets/KITTI-360",
      "sequences": ["2013_05_28_drive_0000_sync"]
    },
    {
      "label": "kitti360_unc_share",
      "dataset": "kitti360",
      "root": "\\\\10.1.1.1\\123123\\KITTI-360",
      "sequences": ["2013_05_28_drive_0000_sync"]
    }
  ]
}
```

JSON 中的 UNC 路径需要转义反斜杠，例如
`\\\\10.1.1.1\\123123\\KITTI-360`，对应实际路径 `\\10.1.1.1\123123\KITTI-360`。

当前支持的数据集 key：

| Dataset key | Loader |
| --- | --- |
| `kitti360`, `kitti-360` | KITTI-360 raw perspective layout |
| `kitti`, `kitti-odometry` | KITTI odometry-style layout |
| `nuscenes` | nuScenes JSON table layout |
| `wayve`, `wayvescenes`, `wayvescenes101` | WayveScenes/Nerfstudio-style transforms |
| `waymo-kitti`, `waymo_kitti`, `waymo-converted-kitti` | Waymo converted to KITTI-style layout |
| `blendedmvs`, `blendedmvg` | BlendedMVG layout |

## 使用

验证配置文件中的一个数据集条目：

```bash
python -m unidata_skill validate-config \
  --config dataset_config.local.json \
  --label kitti360_train
```

验证器会输出 JSON 报告：

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

## 数据集目录结构

每个 dataloader 直接读取对应数据集的官方或转换后目录结构。KITTI-360 示例：

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

## 开发

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 已知限制

- direct dataloader 当前不生成 index、cache 或中间 target schema。
- 当前自动驾驶类 direct loader 尚未加载 dense depth、点云投影、语义标签、instance 标签和 3D boxes。
- 当前 direct loader 尚未实现 nuScenes 和 Waymo native point cloud/depth projection。
- Waymo 当前支持 KITTI-style converted layout，不直接读取 native TFRecord/protobuf table。

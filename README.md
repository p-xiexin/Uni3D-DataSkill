# Uni3D-DataSkill：KITTI-360 至 Pi3X 数据加载验证说明

## 1. 目标与范围

本仓库当前实现的第一版流程用于验证 KITTI-360 原始数据目录能否直接接入 Pi3X 训练分支的数据集接口。处理链路如下：

```text
KITTI-360 原始数据目录
  -> Kitti360Pi3XDataset
  -> Pi3X training BaseDataset
  -> PyTorch DataLoader
  -> 数据完整性验证
```

本流程不生成 index、cache 或其他中间表示。对于 KITTI-360 这类原始目录结构稳定、且已提供图像、标定和位姿的数据集，第一阶段采用直接读取方式，以减少数据转换层带来的额外状态。

当前覆盖范围限定为：

- KITTI-360 perspective rectified camera。
- `image_00` 与可选 `image_01`。
- `calibration/perspective.txt` 中的相机内参。
- `data_poses/<sequence>/cam0_to_world.txt` 中的相机位姿。
- Pi3X `BaseDataset` 所需的基本 view 字段。

暂不覆盖 fisheye camera、Velodyne、SICK、semantic、instance、3D bbox 和 dense depth。

## 2. 环境配置

进入仓库目录：

```powershell
cd E:\Projects\Uni3D-DataSkill
```

安装本仓库为 editable 包：

```powershell
python -m pip install -e .
```

也可以不安装包，直接在当前 PowerShell 会话中设置源码路径：

```powershell
$env:PYTHONPATH='src'
```

如需使用 Pi3X training 分支中的真实 `BaseDataset`，应在本地准备 Pi3 仓库：

```powershell
git clone https://github.com/yyfz/Pi3.git E:\Projects\Pi3
cd E:\Projects\Pi3
git checkout training
```

运行验证命令时，通过 `--pi3-root` 指向该 Pi3 仓库目录。若未指定 `--pi3-root`，程序将使用本仓库内置的轻量 fallback dataset。fallback 仅用于检查 KITTI-360 读取逻辑，不代表 Pi3X training 的完整运行环境。

## 3. 数据目录要求

验证脚本要求 KITTI-360 数据目录至少包含以下文件：

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
      image_01/
        data_rect/
          0000000000.png
          ...
  data_poses/
    2013_05_28_drive_0000_sync/
      cam0_to_world.txt
```

默认只读取 `image_00`。如需同时检查 `image_01`，可重复传入 `--camera`：

```powershell
--camera image_00 --camera image_01
```

## 4. 验证命令

推荐使用配置文件维护数据集 label 与路径映射。复制示例配置：

```powershell
Copy-Item dataset_config.example.json dataset_config.local.json
```

编辑 `dataset_config.local.json`：

```json
{
  "datasets": [
    {
      "label": "kitti360_train",
      "dataset": "kitti360",
      "root": "E:/Datasets/KITTI-360",
      "sequences": ["2013_05_28_drive_0000_sync"],
      "cameras": ["image_00"],
      "frame_num": 8,
      "stride": 5,
      "resolution": "512x384",
      "max_samples": 4,
      "batch_size": 1
    },
    {
      "label": "blendedmvs_train",
      "dataset": "blendedmvs",
      "root": "E:/Datasets/BlendedMVG",
      "mode": "train",
      "frame_num": 8,
      "resolution": "768x576",
      "max_samples": 4,
      "batch_size": 1
    }
  ]
}
```

然后按 label 自动推导并验证：

```powershell
python -m unidata_skill validate-config `
  --config dataset_config.local.json `
  --label kitti360_train `
  --pi3-root E:\Projects\Pi3
```

未传 `--label` 时默认验证配置中的第一个数据集。

`dataset: "kitti360"` 会调用 `Kitti360Pi3XDataset`。`dataset: "blendedmvs"` 会调用用户提供的 `BlendedMVGDataset`，其目录下应包含 `BlendedMVG_training.txt` 或 `validation_list.txt`，并依赖 Pi3X training 分支的 `datasets.base.*`。

使用已安装包时：

```powershell
python -m unidata_skill validate-kitti360-pi3x `
  --kitti360-root E:\Datasets\KITTI-360 `
  --pi3-root E:\Projects\Pi3 `
  --sequence 2013_05_28_drive_0000_sync `
  --frame-num 8 `
  --stride 5 `
  --resolution 512x384 `
  --max-samples 4 `
  --batch-size 1
```

使用源码路径时：

```powershell
$env:PYTHONPATH='src'
python -m unidata_skill validate-kitti360-pi3x `
  --kitti360-root E:\Datasets\KITTI-360 `
  --sequence 2013_05_28_drive_0000_sync `
  --frame-num 8 `
  --stride 5
```

参数含义如下：

| 参数 | 含义 |
| --- | --- |
| `--kitti360-root` | KITTI-360 数据集根目录。 |
| `--pi3-root` | 本地 Pi3 training 分支目录，可选。 |
| `--sequence` | 待验证的 KITTI-360 sequence，可重复传入；未指定时扫描 `data_2d_raw/` 下全部 sequence。 |
| `--camera` | 待验证相机，可选 `image_00` 或 `image_01`；未指定时使用 `image_00`。 |
| `--frame-num` | 单个样本包含的帧数，默认值为 `8`。 |
| `--stride` | 同一样本内相邻帧的间隔，默认值为 `5`。 |
| `--resolution` | 传入 Pi3X `BaseDataset` 的目标分辨率，格式为 `宽x高`。 |
| `--max-samples` | 抽样检查的 dataset sample 数量。 |
| `--batch-size` | 构造 PyTorch `DataLoader` 时使用的 batch size。 |

## 5. 输出格式

验证命令输出 JSON 报告：

```json
{
  "status": "ok",
  "dataset_len": 1234,
  "checked_samples": 4,
  "checked_batches": 1,
  "errors": [],
  "warnings": [
    "dense depth is not available in the first KITTI-360 workflow; depthmap is a placeholder"
  ]
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `status` | 验证状态。`ok` 表示通过，`error` 表示存在数据或加载问题。 |
| `dataset_len` | 可形成的 frame window 数量。 |
| `checked_samples` | 实际检查的 dataset sample 数量。 |
| `checked_batches` | 实际检查的 PyTorch `DataLoader` batch 数量。 |
| `errors` | 缺失图像、非法相机内参、非法 pose、非法 depthmap 等错误。 |
| `warnings` | 当前流程的限制或降级说明。 |

## 6. 已知限制

- 不生成 index、cache 或中间格式。
- 不复制 KITTI-360 原始图像。
- 不处理 fisheye camera、Velodyne、SICK、semantic、instance 和 3D bbox。
- 当前未接入 KITTI-360 dense depth；`depthmap` 为 placeholder，仅用于满足 Pi3X `BaseDataset` 字段契约，不能作为 ground truth depth 使用。
- 未安装 PyTorch 时，验证脚本会跳过 PyTorch `DataLoader` batch 检查，并在 `warnings` 中记录该情况。

## 7. 单元测试

运行测试：

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests
```

测试用例会临时构造 tiny KITTI-360 fixture，不依赖真实 KITTI-360 数据集。

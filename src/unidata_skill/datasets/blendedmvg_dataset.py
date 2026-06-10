import os
import os.path as osp

import numpy as np
import cv2

from datasets.base.base_dataset import BaseDataset

def read_pfm(filename):
    """
    读取 PFM 格式的深度图文件
    Args:
        filename: PFM 文件路径
    Returns:
        depthmap: 深度图 (H, W) numpy 数组
    """
    with open(filename, 'rb') as f:
        header = f.readline().decode('ascii').strip()
        if header == 'PF':
            channels = 3
        elif header == 'Pf':
            channels = 1
        else:
            raise ValueError(f'Not a PFM file: {filename}')

        # 读取尺寸
        dim_match = f.readline().decode('ascii').strip()
        width, height = map(int, dim_match.split())

        # 读取比例因子
        scale = float(f.readline().decode('ascii').strip())
        if scale < 0:
            endian = '<'  # little endian
            scale = -scale
        else:
            endian = '>'  # big endian

        # 读取数据
        data = f.read()
        dtype = np.float32
        data = np.frombuffer(data, dtype=dtype)

        # 重塑数组
        if channels == 1:
            data = data.reshape(height, width)
        else:
            data = data.reshape(height, width, channels)

        # PFM 存储时是上下颠倒的，需要翻转
        data = np.flipud(data)

        # 应用比例因子
        data = data * scale

        return data

class BlendedMVGDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'BlendedMVG'
        self.data_root = data_root

        # 根据 mode 读取相应的序列列表
        if self.mode == 'train':
            list_file = osp.join(data_root, 'BlendedMVG_training.txt')
        else:
            list_file = osp.join(data_root, 'validation_list.txt')

        if not osp.exists(list_file):
            raise FileNotFoundError(f'List file not found: {list_file}')

        # 读取序列列表
        with open(list_file, 'r') as f:
            self.sequences = [line.strip() for line in f.readlines() if line.strip()]

        if self.verbose:
            print(f'[{self.dataset_label}] Sequences of {self.dataset_label} dataset:', self.sequences)

        print(f'[{self.dataset_label}] Found {len(self.sequences)} unique videos in {data_root}', flush=True)

        # 存储每个序列的图像数量
        self.num_imgs = {}
        for seq in self.sequences:
            img_path = osp.join(data_root, seq, 'blended_images')
            if osp.exists(img_path):
                # 只计算 .jpg 文件，排除 _masked.jpg
                img_files = [f for f in os.listdir(img_path) if f.endswith('.jpg') and not f.endswith('_masked.jpg')]
                self.num_imgs[seq] = len(img_files)
            else:
                self.num_imgs[seq] = 0

    def __len__(self):
        return len(self.sequences)

    def _get_views(self, index, resolution, rng, is_test = False):
        scene = self.sequences[index]
        num_imgs = self.num_imgs[scene]

        # 随机选择帧索引
        if num_imgs <= self.frame_num:
            self.frame_num = num_imgs
            idxs = range(num_imgs)
        else:
            img_idx = rng.integers(0, num_imgs)
            front_num = (self.frame_num - 1) // 2
            back_num = self.frame_num - 1 - front_num
            if img_idx - front_num < 0:
                begin = 0
                end = self.frame_num
            elif img_idx + back_num >= num_imgs:
                begin = num_imgs - self.frame_num
                end = num_imgs
            else:
                begin = img_idx - front_num
                end = img_idx + back_num + 1
            idxs = range(begin, end)

        self.this_views_info = dict(
            scene=scene,
            idxs=list(idxs),
        )

        views = []
        scene_path = osp.join(self.data_root, scene)

        for idx in idxs:
            # 构建文件路径
            img_name = f'{idx:08d}.jpg'
            img_path = osp.join(scene_path, 'blended_images', img_name)
            depth_name = f'{idx:08d}.pfm'
            depth_path = osp.join(scene_path, 'rendered_depth_maps', depth_name)
            cam_name = f'{idx:08d}_cam.txt'
            cam_path = osp.join(scene_path, 'cams', cam_name)

            # 检查文件是否存在
            if not osp.exists(img_path):
                print(f'Warning: Image not found: {img_path}', flush=True)
                continue
            if not osp.exists(depth_path):
                print(f'Warning: Depth not found: {depth_path}', flush=True)
                continue
            if not osp.exists(cam_path):
                print(f'Warning: Camera not found: {cam_path}', flush=True)
                continue

            # 加载图像
            img = cv2.imread(img_path)
            if img is None:
                print(f'Warning: Failed to load image: {img_path}', flush=True)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 加载深度图
            try:
                depthmap = read_pfm(depth_path)
            except Exception as e:
                print(f'Warning: Failed to load depth: {depth_path}, error: {e}', flush=True)
                continue

            # 加载相机参数
            try:
                with open(cam_path, 'r') as f:
                    lines = f.readlines()

                # 解析 extrinsic 矩阵 (4x4)
                extrinsic_start = None
                intrinsic_start = None
                for i, line in enumerate(lines):
                    if 'extrinsic' in line.lower():
                        extrinsic_start = i + 1
                    elif 'intrinsic' in line.lower():
                        intrinsic_start = i + 1

                if extrinsic_start is None or intrinsic_start is None:
                    raise ValueError("Camera file format error: extrinsic or intrinsic not found")

                # 读取外参矩阵 (4行)
                camera_pose = np.zeros((4, 4), dtype=np.float32)
                camera_pose[3, 3] = 1.0
                for i in range(4):
                    values = list(map(float, lines[extrinsic_start + i].strip().split()))
                    camera_pose[i, :] = values

                # 读取内参矩阵 (3行)
                intrinsics = np.zeros((3, 3), dtype=np.float32)
                for i in range(3):
                    values = list(map(float, lines[intrinsic_start + i].strip().split()))
                    intrinsics[i, :] = values

                # 读取深度参数 (最后一行，可选)
                # 格式: DEPTH_MIN DEPTH_INTERVAL DEPTH_NUM DEPTH_MAX
                depth_params = None
                if len(lines) > intrinsic_start + 3:
                    last_line = lines[-1].strip()
                    if last_line and 'extrinsic' not in last_line.lower() and 'intrinsic' not in last_line.lower():
                        try:
                            depth_params = list(map(float, last_line.split()))
                        except:
                            pass

            except Exception as e:
                print(f'Warning: Failed to load camera: {cam_path}, error: {e}', flush=True)
                # 使用默认相机参数
                intrinsics = np.array([
                    [500.0, 0, img.shape[1] / 2],
                    [0, 500.0, img.shape[0] / 2],
                    [0, 0, 1]
                ], dtype=np.float32)
                camera_pose = np.eye(4, dtype=np.float32)

            # 获取原始图像尺寸
            original_height, original_width = img.shape[:2] # 576 x 768
            target_width, target_height = resolution        # 768 x 576

            # 根据分辨率调整内参矩阵
            factor_w = target_width / original_width
            factor_h = target_height / original_height
            intrinsics[0, 0] *= factor_w  # fx
            intrinsics[1, 1] *= factor_h  # fy
            intrinsics[0, 2] *= factor_w  # cx
            intrinsics[1, 2] *= factor_h  # cy

            # 调整分辨率
            img, depthmap, intrinsics = self._crop_resize_if_necessary(
                img, depthmap, intrinsics, resolution, rng=rng, info=img_path)

            views.append(dict(
                img=img,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset=self.dataset_label,
                label=scene,
                instance=img_name,
                prefix=f'{scene}_{img_name}',
            ))

        return views

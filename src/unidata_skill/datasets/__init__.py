import sys
from pathlib import Path


PI3_ROOT = Path(__file__).resolve().parents[3] / "thirdparty" / "Pi3"
if str(PI3_ROOT) not in sys.path:
    sys.path.insert(0, str(PI3_ROOT))

from .pi3x_validator import ValidationResult, validate_pi3x_dataset, validate_pi3x_view

__all__ = [
    "Kitti360Pi3XDataset",
    "KittiOdometryPi3XDataset",
    "NuScenesPi3XDataset",
    "ValidationResult",
    "WaymoKittiPi3XDataset",
    "WayveScenesPi3XDataset",
    "validate_pi3x_dataset",
    "validate_pi3x_view",
]


def __getattr__(name: str):
    if name == "Kitti360Pi3XDataset":
        from .kitti360_dataset import Kitti360Pi3XDataset

        return Kitti360Pi3XDataset
    if name == "KittiOdometryPi3XDataset":
        from .kitti_odometry_dataset import KittiOdometryPi3XDataset

        return KittiOdometryPi3XDataset
    if name == "NuScenesPi3XDataset":
        from .nuscenes_dataset import NuScenesPi3XDataset

        return NuScenesPi3XDataset
    if name == "WaymoKittiPi3XDataset":
        from .waymo_kitti_dataset import WaymoKittiPi3XDataset

        return WaymoKittiPi3XDataset
    if name == "WayveScenesPi3XDataset":
        from .wayve_dataset import WayveScenesPi3XDataset

        return WayveScenesPi3XDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

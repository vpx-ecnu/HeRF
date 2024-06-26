from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .your_own_data import YourOwnDataset
from .scannet import ScannetDataset



dataset_dict = {
    'blender': BlenderDataset,
    'llff': LLFFDataset,
    'tankstemple': TanksTempleDataset,
    'nsvf': NSVF,
    'own_data': YourOwnDataset,
    'scannet': ScannetDataset,
}
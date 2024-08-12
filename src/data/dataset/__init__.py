from ..name import init_name_dataset_dict

init_name_dataset_dict()

from .NuSceneLIDARSegmentationDataset import *
from .ModelNet40Dataset import *
from .LRADataset import *
from .LRGBDataset import *
from .ShapeNetPartDataset import *
from .from_str import parse_dataset_str

__all__ = ['parse_dataset_str']
from .DatasetName import init_name_dataset_dict

init_name_dataset_dict()

from .DatasetName import dataset_parser_name_dict
from ..dataset import *

__all__ = ['dataset_parser_name_dict',
           'init_name_dataset_dict']
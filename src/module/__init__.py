from ..module.name import init_name_method_dict

init_name_method_dict()

from .from_str import str_to_module_list, str_to_sequential_model

from .layer import *
from .head import *

wrapper = {
    "MODULELIST": str_to_module_list,
    "SEQUENTIAL": str_to_sequential_model,   
}

__all__ = [
    "str_to_module_list",
    "str_to_sequential_model",
    "wrapper"
]
from ..name import init_name_optimizer_dict, init_name_scheduler_dict

init_name_optimizer_dict()
init_name_scheduler_dict()

from .StepLRScheduler import *
from .CosineAnnealing import *
# Add similar lines for other scheduler modules as needed

from .from_str import parse_scheduler_str

__all__ = ['parse_scheduler_str']
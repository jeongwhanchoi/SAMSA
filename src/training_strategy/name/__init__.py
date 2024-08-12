from .OptimizerName import init_name_optimizer_dict
from .SchedulerName import init_name_scheduler_dict

init_name_optimizer_dict()
init_name_scheduler_dict()

from .OptimizerName import optimizer_getfn_name_dict, optimizer_parser_name_dict
from .SchedulerName import scheduler_getfn_name_dict, scheduler_parser_name_dict
from ..optimizer import *
from ..scheduler import *

__all__ = ['optimizer_getfn_name_dict',
           'optimizer_parser_name_dict',
           'scheduler_getfn_name_dict',
           'scheduler_parser_name_dict',
           'init_name_optimizer_dict',
           'init_name_scheduler_dict']
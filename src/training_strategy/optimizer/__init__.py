from ..name import init_name_optimizer_dict, init_name_scheduler_dict

init_name_optimizer_dict()
init_name_scheduler_dict()


from .AdamOptimizer import parse_adam_optimizer, get_adam_optimizer
from .AdamWOptimizer import parse_adamw_optimizer, get_adamw_optimizer
from .SGDOptimizer import parse_sgd_optimizer, get_sgd_optimizer
# from .AdamW8bitOptimizer import parse_adamw8bit_optimizer, get_adamw8bit_optimizer
# from .LowBitOptimizer import *

from .from_str import parse_optimizer_str

__all__ = ['parse_adam_optimizer',
           'get_adam_optimizer',
           'parse_adamw_optimizer',
           'get_adamw_optimizer',
           'parse_sgd_optimizer',
           'get_sgd_optimizer',
           'parse_optimizer_str',]
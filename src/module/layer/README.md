This is the general design of layer module; please adhere (or improve) the rules/design philosophy

Each file consists of one type of layer/PyTorch module and supporting modules if needed.
If you believe the supporting modules can be used in more than one module/main layer, please write that module in sublayer.
Every module should consist of for readability:
- Name
- Description of module:
    - What the module does
    - The hyperparameters of the module and its meaning
    - The type of input it receives and output it sends
- Module settings: this is to make a hashcode of the entire model
- Implementation

To define the name, please "from ...name import module_name_dict"
Then module_name_dict\[model_name\] = class_name
For example, please refer to the implementation of GenericTransformerEncoderLayer; defined in "generic_transformer_layer.py"

Thank you
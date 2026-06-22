"""Method 1: SCBert with Masked Language Modeling (MLM) pretraining."""
from .model import MethylationAgeModel, MethylationEncoder
from .lora import inject_lora, get_lora_parameters

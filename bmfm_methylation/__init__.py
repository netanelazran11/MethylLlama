import torch
import torch.serialization

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
torch.serialization.load = _patched_torch_load

from .shared.config import create_methylation_config, BMFMConfig, PretrainingConfig
from .shared.tokenizer import (
    create_methylation_multifield_tokenizer,
    create_indexed_tokenizer,
    extract_cpg_sites_from_h5ad,
)
from .shared.data_module import MethylationDataset, MethylationDataModule, WCEDCollator, BMFMWCEDCollator
from .wced.wced_module import WCEDTrainingModule

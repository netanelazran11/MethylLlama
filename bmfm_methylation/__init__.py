import torch
import torch.serialization

# Checkpoints of this project store Python objects (OmegaConf hyper-parameters), which
# torch>=2.6 does not unpickle by default. Loading them relies on weights_only=False, so the
# package sets it on import (as do shared/data_module.py and llama/finetune_llama.py).
# Load only checkpoints from trusted sources.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
torch.serialization.load = _patched_torch_load

from .shared.tokenizer import (
    create_methylation_multifield_tokenizer,
    create_indexed_tokenizer,
    extract_cpg_sites_from_h5ad,
)
from .shared.data_module import MethylationDataset, MethylationDataModule, WCEDCollator, BMFMWCEDCollator

"""Method 3: LLaMA-based architecture with WCED pretraining."""
from .model import MethylLlamaConfig, MethylLlamaModel, build_methyl_llama
from .wced_llama import WCEDLlamaModule, WCEDDecoder, ProjectionHead

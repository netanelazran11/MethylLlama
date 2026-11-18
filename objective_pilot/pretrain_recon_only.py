#!/usr/bin/env python3
"""
Entry point for the reconstruction-only arm of the InfoNCE ablation.

Runs the unmodified pretraining entry point (bmfm_methylation.llama.pretrain_llama.main,
same Hydra config, same CLI overrides) with one substitution: the training module
class is WCEDLlamaReconOnlyModule instead of WCEDLlamaModule. Nothing on disk in
bmfm_methylation/ is changed; the swap exists only inside this process.

Run from the repository root:
    python -m objective_pilot.pretrain_recon_only <same overrides as pretrain_llama>
Must be launched with wced_contrastive=true (so the collator builds view 2).
"""

# Import order matters: pretrain_llama patches torch.load at import time.
from bmfm_methylation.llama import pretrain_llama

import os

from objective_pilot.wced_llama_recon_only import (
    WCEDLlamaAblationModule,
    WCEDLlamaReconOnlyModule,
    configure_from_env,
)

# ABLATION_MODULE=recon_only (default): λ forced to 0 — the full-corpus arm.
# ABLATION_MODULE=sweep: λ = wced_contrastive_weight — every pilot arm, λ = 0 included.
MODULES = {"recon_only": WCEDLlamaReconOnlyModule, "sweep": WCEDLlamaAblationModule}

if __name__ == "__main__":
    configure_from_env()
    module_cls = MODULES[os.environ.get("ABLATION_MODULE", "recon_only")]
    pretrain_llama.WCEDLlamaModule = module_cls  # name looked up inside main()
    # Hydra resolves config_path="configs" from the task function's module. Imported (not run
    # as __main__), that module name would send Hydra to pkg://bmfm_methylation.llama/configs,
    # which it rejects (configs/ has no __init__.py). Marking it __main__ makes Hydra use the
    # file location instead, i.e. bmfm_methylation/llama/configs/ — the same directory the
    # published run (python -m bmfm_methylation.llama.pretrain_llama) read.
    pretrain_llama.main.__wrapped__.__module__ = "__main__"
    print(f"[objective_pilot] module class = {pretrain_llama.WCEDLlamaModule.__name__}  "
          f"analysis_epochs={sorted(module_cls.analysis_epochs)}  "
          f"stop_after_epoch={module_cls.stop_after_epoch}  "
          f"grad_checkpointing={module_cls.grad_checkpointing}", flush=True)
    pretrain_llama.main()

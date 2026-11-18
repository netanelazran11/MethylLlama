#!/usr/bin/env python3
"""
Equivalence test for WCEDLlamaReconOnlyModule against the unmodified WCEDLlamaModule.

Builds a tiny model and a synthetic WCEDCollator-format batch, copies identical
weights into both modules, and checks:
  1. recon loss of the arm == recon_loss of the published objective (same batch, both views)
  2. published loss == recon + 0.05 * InfoNCE (sanity of the decomposition)
  3. encoder/decoder gradients of the arm == gradients of the published recon term alone
  4. projection head gets no gradient in the arm; it does in the published objective
  5. the published module with weight 0 uses one view only (the confound this arm removes)
  6. monitor values are finite and in range
  7. sweep module at λ=0.05 == published objective (loss, and gradients of every parameter incl. the head)
  8. sweep module at λ=0 == recon-only module (loss and gradients)
  9. gradient checkpointing gives the same loss and gradients (eval and train mode, dropout included)

CPU only, no data. Run from the repository root:  python -m objective_pilot.test_recon_only
bmfm_targets is stubbed when it is not installed (it is only imported, not used here).
"""

import importlib.abc
import importlib.machinery
import sys
import types
from unittest.mock import MagicMock

try:
    import bmfm_targets  # noqa: F401
except ImportError:
    class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path=None, target=None):
            if name == "bmfm_targets" or name.startswith("bmfm_targets."):
                return importlib.machinery.ModuleSpec(name, self, is_package=True)
            return None

        def create_module(self, spec):
            m = types.ModuleType(spec.name)
            m.__path__ = []
            m.__getattr__ = lambda attr: MagicMock(name=f"{spec.name}.{attr}")
            return m

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _StubFinder())

import torch

from bmfm_methylation.llama.model import MethylLlamaConfig
from bmfm_methylation.llama.wced_llama import WCEDLlamaModule
from objective_pilot.wced_llama_recon_only import WCEDLlamaAblationModule, WCEDLlamaReconOnlyModule

torch.manual_seed(0)
B, V, CLS, PAD = 8, 40, 3, 2
L = int(V * 0.5) + 1


def make_view(betas, valid):
    ids = torch.full((B, L), PAD, dtype=torch.long)
    vals = torch.zeros(B, L)
    attn = torch.zeros(B, L, dtype=torch.long)
    pos = torch.zeros(B, L, dtype=torch.long)
    imask = torch.zeros(B, V, dtype=torch.bool)
    for i in range(B):
        vi = torch.nonzero(valid[i]).squeeze(1)
        sel = vi[torch.randperm(len(vi))[: int(len(vi) * 0.5)]].sort().values
        n = len(sel)
        ids[i, 0] = CLS
        vals[i, 0] = 0.0
        attn[i, : n + 1] = 1
        ids[i, 1 : n + 1] = sel + 5
        vals[i, 1 : n + 1] = betas[i, sel]
        pos[i, 1 : n + 1] = sel + 1
        imask[i, sel] = True
    return ids, vals, attn, pos, imask


betas = torch.rand(B, V)
valid = torch.rand(B, V) > 0.05
betas = torch.where(valid, betas, torch.zeros_like(betas))
v1, v2 = make_view(betas, valid), make_view(betas, valid)
batch = {
    "cpg_ids": v1[0], "beta_values": v1[1], "attention_mask": v1[2], "position_ids": v1[3], "input_mask": v1[4],
    "cpg_ids_v2": v2[0], "beta_values_v2": v2[1], "attention_mask_v2": v2[2], "position_ids_v2": v2[3],
    "input_mask_v2": v2[4],
    "all_betas": betas, "valid_mask": valid, "age": torch.full((B,), float("nan")),
}

cfg = MethylLlamaConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, vocab_size=V + 5, max_seq_len=128)
kw = dict(model_config=cfg, vocab_size=V, contrastive_temp=0.1, normalize_loss=True, age_weight=0.0)

pub = WCEDLlamaModule(contrastive_weight=0.05, **kw)
arm = WCEDLlamaReconOnlyModule(contrastive_weight=0.05, **kw)  # weight forced to 0 inside
pub0 = WCEDLlamaModule(contrastive_weight=0.0, **kw)
arm.load_state_dict(pub.state_dict())
pub0.load_state_dict(pub.state_dict())
for m in (pub, arm, pub0):
    m.eval()  # no dropout, so forward passes are comparable

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


check("arm forces contrastive_weight=0", arm.contrastive_weight == 0.0)

o_pub = pub._shared_step(batch, "train")
o_arm = arm._shared_step(batch, "train")
check("1 recon loss identical", torch.allclose(o_pub["recon_loss"], o_arm["recon_loss"], atol=1e-7),
      f"pub={o_pub['recon_loss'].item():.7f} arm={o_arm['recon_loss'].item():.7f}")
check("2 published loss = recon + 0.05*InfoNCE",
      torch.allclose(o_pub["loss"], o_pub["recon_loss"] + 0.05 * o_pub["contrastive_loss"], atol=1e-7))
check("  arm loss = recon only", torch.equal(o_arm["loss"], o_arm["recon_loss"]))

# gradients
pub.zero_grad(); arm.zero_grad()
o_pub["recon_loss"].backward()
o_arm["loss"].backward()
max_diff, n = 0.0, 0
for (na, pa), (nb, pb) in zip(pub.named_parameters(), arm.named_parameters()):
    if na.startswith(("encoder.", "decoder.")):
        if pa.grad is None and pb.grad is None:
            continue
        max_diff = max(max_diff, (pa.grad - pb.grad).abs().max().item()); n += 1
check("3 encoder/decoder grads == published recon-term grads", max_diff < 1e-6, f"({n} tensors, max |diff|={max_diff:.2e})")
head_grads = [p.grad for p in arm.projection_head.parameters()]
check("4 arm projection head has no grad", all(g is None for g in head_grads))
pub.zero_grad()
pub._shared_step(batch, "train")["loss"].backward()
check("  published projection head has grad", all(p.grad is not None for p in pub.projection_head.parameters()))

o_pub0 = pub0._shared_step(batch, "train")
check("5 published module at weight 0 uses view 1 only (the confound)",
      not torch.allclose(o_pub0["recon_loss"], o_arm["recon_loss"]),
      f"weight0 recon={o_pub0['recon_loss'].item():.5f} vs two-view {o_arm['recon_loss'].item():.5f}")

mon = arm._monitor
check("6 monitor finite and in range",
      all(torch.isfinite(v) for v in mon.values()) and 0 <= mon["monitor_cls_top1"].item() <= 1,
      {k: round(v.item(), 4) for k, v in mon.items()})



def grad_diff(ma, mb, prefixes=None):
    d, n = 0.0, 0
    for (na, pa), (nb, pb) in zip(ma.named_parameters(), mb.named_parameters()):
        assert na == nb
        if prefixes and not na.startswith(prefixes):
            continue
        if pa.grad is None and pb.grad is None:
            continue
        if (pa.grad is None) != (pb.grad is None):
            return float("inf"), n
        d = max(d, (pa.grad - pb.grad).abs().max().item()); n += 1
    return d, n


sw05 = WCEDLlamaAblationModule(contrastive_weight=0.05, **kw)
sw0 = WCEDLlamaAblationModule(contrastive_weight=0.0, **kw)
for m in (sw05, sw0):
    m.load_state_dict(pub.state_dict()); m.eval()

pub.zero_grad(); sw05.zero_grad()
lp = pub._shared_step(batch, "train")["loss"]; ls = sw05._shared_step(batch, "train")["loss"]
lp.backward(); ls.backward()
d, n = grad_diff(pub, sw05)
check("7 sweep λ=0.05 loss == published loss", torch.allclose(lp, ls, atol=1e-7), f"pub={lp.item():.7f} sweep={ls.item():.7f}")
check("  sweep λ=0.05 grads == published grads (all params)", d < 1e-6, f"({n} tensors incl. head, max |diff|={d:.2e})")

arm.zero_grad(); sw0.zero_grad()
la = arm._shared_step(batch, "train")["loss"]; l0 = sw0._shared_step(batch, "train")["loss"]
la.backward(); l0.backward()
d, n = grad_diff(arm, sw0)
check("8 sweep λ=0 == recon-only (loss, grads)", torch.equal(la, l0) and d == 0.0, f"({n} tensors)")


for mode in ("eval", "train"):
    m_plain = WCEDLlamaAblationModule(contrastive_weight=0.05, **kw)
    m_ckpt = WCEDLlamaAblationModule(contrastive_weight=0.05, **kw)
    m_plain.load_state_dict(pub.state_dict()); m_ckpt.load_state_dict(pub.state_dict())
    m_ckpt.enable_grad_checkpointing()
    for m in (m_plain, m_ckpt):
        getattr(m, mode)()
    torch.manual_seed(123); lp_ = m_plain._shared_step(batch, "train")["loss"]; lp_.backward()
    torch.manual_seed(123); lc_ = m_ckpt._shared_step(batch, "train")["loss"]; lc_.backward()
    d, n = grad_diff(m_plain, m_ckpt)
    check(f"9 grad checkpointing ({mode}) == plain (loss, grads)",
          torch.allclose(lp_, lc_, atol=1e-6) and d < 1e-5, f"loss {lp_.item():.7f} vs {lc_.item():.7f}, {n} tensors, max |diff|={d:.2e}")

print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)

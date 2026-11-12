"""
view_encoding.py
================
Two-view construction and encoder call shared by the cross-view analyses
(view_consistency_heldout_full.py): for every profile, two views of input_ratio x its measured
CpGs -- two independent draws ("overlap", the pretraining construction) or a random partition
("disjoint", no shared CpG) -- each ordered by genomic rank, with genomic-rank position_ids,
encoded to the CLS pooler output.
"""
import numpy as np
import torch


def build_views(valid_idx, n_input, disjoint, rng):
    """Return (idx_v1, idx_v2) as arrays of column indices."""
    if disjoint:
        perm = rng.permutation(valid_idx)
        # a 50/50 partition: the two views share no CpG by construction
        return np.sort(perm[:n_input]), np.sort(perm[n_input:2 * n_input])
    return (np.sort(rng.choice(valid_idx, size=n_input, replace=False)),
            np.sort(rng.choice(valid_idx, size=n_input, replace=False)))


def encode_views(encoder, betas_mat, valid_mat, cpg_vocab_ids, genomic_rank,
                 cls_id, pad_id, cls_beta, pad_beta, disjoint, seed, a):
    """Encode both views for every profile; returns two [N, D] float arrays."""
    rng = np.random.default_rng(seed)
    n_prof = betas_mat.shape[0]
    embs = [[], []]
    for start in range(0, n_prof, a.batch_size):
        stop = min(start + a.batch_size, n_prof)
        batch_idx = range(start, stop)
        views = [[], []]
        for i in batch_idx:
            valid_idx = np.where(valid_mat[i])[0]
            n_input = int(len(valid_idx) * a.input_ratio)
            if disjoint:
                n_input = min(n_input, len(valid_idx) // 2)
            views[0].append(None); views[1].append(None)
            v1, v2 = build_views(valid_idx, n_input, disjoint, rng)
            views[0][-1], views[1][-1] = v1, v2

        for v in (0, 1):
            L = max(len(x) for x in views[v]) + 1
            B = stop - start
            ids = torch.full((B, L), pad_id, dtype=torch.long)
            bet = torch.full((B, L), pad_beta, dtype=torch.float32)
            att = torch.zeros(B, L, dtype=torch.long)
            pos = torch.zeros(B, L, dtype=torch.long)
            for b, i in enumerate(batch_idx):
                sel = views[v][b]
                sel = sel[np.argsort(genomic_rank[sel])]      # genomic order
                k = len(sel)
                ids[b, 0] = cls_id; bet[b, 0] = cls_beta
                ids[b, 1:k + 1] = torch.from_numpy(cpg_vocab_ids[sel].astype(np.int64))
                bet[b, 1:k + 1] = torch.from_numpy(betas_mat[i, sel].astype(np.float32))
                att[b, :k + 1] = 1
                pos[b, 1:k + 1] = torch.from_numpy((genomic_rank[sel] + 1).astype(np.int64))
            with torch.no_grad():
                inp = torch.stack([ids.float(), bet], dim=1).to(a.device)
                out = encoder(input_ids=inp, attention_mask=att.to(a.device),
                              position_ids=pos.to(a.device))
            embs[v].append(out.pooler_output.cpu().float().numpy())
    return np.concatenate(embs[0]), np.concatenate(embs[1])

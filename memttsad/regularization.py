"""Memory-slot diversity regularisation (squared pairwise cosine, i.e. soft orthogonality).

    L_div(M) = 2 / (m (m - 1)) * sum_{i<j} ( <M_i, M_j> / (|M_i| |M_j|) )^2

Applied to
  * the RAW learnable slots  (model.mem_tokens / model.mem_token, shape [m, 1, d]) and/or
  * the UPDATED slots Hmem that the decoder actually reads (shape [m, B, d], one tensor per
    TranAD phase), recorded by the model during the SAME forward pass used for the
    reconstruction loss (no extra forward).

Disabled by default (mode 'none'): the training loop then executes exactly the original code
path, so baseline checkpoints and metrics are unchanged. The loss is only ever added inside
the training branch; validation / test scoring never sees it.
"""
import os
import torch

MODES = ('none', 'raw', 'updated', 'both')


def cosine_sq_diversity(M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean squared pairwise cosine similarity between the m slot vectors.

    M: [m, d]  -> scalar
       [m, B, d] -> scalar, mean over the batch of the per-sample value
    m == 1 (no pairs): returns a differentiable zero (M.sum() * 0), never NaN.
    """
    if M.dim() == 2:
        M = M.unsqueeze(1)                                   # [m, 1, d]
    m = M.shape[0]
    if m < 2:
        return M.sum() * 0.0
    Mn = M / M.norm(dim=-1, keepdim=True).clamp_min(eps)     # unit vectors
    G = torch.einsum('ibd,jbd->bij', Mn, Mn)                 # [B, m, m] cosine Gram matrices
    iu = torch.triu_indices(m, m, offset=1, device=M.device)
    pairs = G[:, iu[0], iu[1]] ** 2                          # [B, m(m-1)/2]
    return pairs.mean()                                      # == 2/(m(m-1)) * sum_{i<j}, averaged over B


def mean_abs_cosine(M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Diagnostic companion: mean |cos| over slot pairs (same conventions as above)."""
    if M.dim() == 2:
        M = M.unsqueeze(1)
    m = M.shape[0]
    if m < 2:
        return M.sum() * 0.0
    Mn = M / M.norm(dim=-1, keepdim=True).clamp_min(eps)
    G = torch.einsum('ibd,jbd->bij', Mn, Mn)
    iu = torch.triu_indices(m, m, offset=1, device=M.device)
    return G[:, iu[0], iu[1]].abs().mean()


def memory_parameter(model):
    """The learnable slot bank ([m, 1, d]) of a memory model, or None for a plain model."""
    for name in ('mem_tokens', 'mem_token'):
        p = getattr(model, name, None)
        if isinstance(p, torch.nn.Parameter):
            return p
    return None


class DiversityConfig:
    """Resolved regularisation settings. Precedence: explicit CLI flag > environment variable > default.

    CLI (memttsad/cli.py):  --memory_diversity {none,raw,updated,both}
                            --diversity_lambda L          (applies to raw and updated unless overridden)
                            --diversity_lambda_raw L_r   --diversity_lambda_updated L_u
    Env (for shell runners): MEMTRANAD_DIVERSITY, MEMTRANAD_DIV_LAMBDA, MEMTRANAD_DIV_LAMBDA_RAW,
                             MEMTRANAD_DIV_LAMBDA_UPD
    """

    def __init__(self, mode=None, lam=None, lam_raw=None, lam_upd=None):
        mode = mode if mode is not None else os.environ.get('MEMTRANAD_DIVERSITY', 'none')
        if mode not in MODES:
            raise ValueError(f'memory_diversity must be one of {MODES}, got {mode!r}')
        lam = float(lam if lam is not None else os.environ.get('MEMTRANAD_DIV_LAMBDA', 0.0))
        lam_raw = lam_raw if lam_raw is not None else os.environ.get('MEMTRANAD_DIV_LAMBDA_RAW')
        lam_upd = lam_upd if lam_upd is not None else os.environ.get('MEMTRANAD_DIV_LAMBDA_UPD')
        self.mode = mode
        self.lambda_raw = float(lam_raw) if lam_raw is not None else lam
        self.lambda_updated = float(lam_upd) if lam_upd is not None else lam
        if mode in ('none', 'updated'): self.lambda_raw = 0.0
        if mode in ('none', 'raw'): self.lambda_updated = 0.0

    @property
    def enabled(self):
        return self.mode != 'none'

    @property
    def use_raw(self):
        return self.mode in ('raw', 'both')

    @property
    def use_updated(self):
        return self.mode in ('updated', 'both')

    def ckpt_suffix(self):
        """Appended to the checkpoint directory ONLY when enabled, so baselines are never overwritten."""
        if not self.enabled:
            return ''
        return f'_div{self.mode}_r{self.lambda_raw:g}_u{self.lambda_updated:g}'

    def as_dict(self):
        return {'mode': self.mode, 'lambda_raw': self.lambda_raw, 'lambda_updated': self.lambda_updated}

    @classmethod
    def from_args(cls, args):
        return cls(getattr(args, 'memory_diversity', None), getattr(args, 'diversity_lambda', None),
                   getattr(args, 'diversity_lambda_raw', None), getattr(args, 'diversity_lambda_updated', None))


def diversity_terms(model, cfg: DiversityConfig):
    """Compute (L_raw, L_updated) for the current step from the model's recorded state.

    L_raw     : cosine_sq_diversity of the learnable slot bank.
    L_updated : mean over recorded phases of cosine_sq_diversity(Hmem_phase), Hmem_phase being the
                [m, B, d] tensor the decoder read in that phase (recorded during model.forward when
                model.collect_memory_reprs is True). Averaging over phases keeps the loss scale
                independent of the number of phases (no double counting).
    Returns zeros (as tensors on the right device) when a term is not requested.
    """
    p = memory_parameter(model)
    if p is None:
        raise ValueError(f'{model.name} has no learnable memory bank; use --memory_diversity none')
    zero = p.sum() * 0.0
    l_raw = cosine_sq_diversity(p[:, 0, :]) if cfg.use_raw else zero
    l_upd = zero
    if cfg.use_updated:
        reprs = getattr(model, '_memory_reprs', None) or []
        if not reprs:
            raise RuntimeError('updated-memory regularisation requested but the model recorded no Hmem; '
                               'set model.collect_memory_reprs = True before the forward pass')
        l_upd = torch.stack([cosine_sq_diversity(h) for h in reprs]).mean()
    return l_raw, l_upd

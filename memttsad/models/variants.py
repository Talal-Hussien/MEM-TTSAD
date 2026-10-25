"""Default-off experiment switches and the regularisation hook.

All switches are read from MEMTRANAD_* environment variables at model
construction time (the CLI copies explicitly set flags into these variables,
see memttsad.cli.bridge_args_to_env). With every variable unset the models are
bit-identical to their original form.
"""
import math
import os


def record_memory(model, hmem):
    """Optional regularisation hook: keep a reference to the updated memory Hmem that the
    decoder reads in the current phase. No-op unless model.collect_memory_reprs is True
    (attribute absent / False by default), so the numerical forward pass is unchanged."""
    if getattr(model, 'collect_memory_reprs', False):
        model._memory_reprs.append(hmem)


def mem_token_init_scale(feats):
    """Scale of the initial memory slots. Default 0.02 (original). With MEMTRANAD_MEM_INIT_NORM=N
    (CLI --memory_init_norm) the slots are drawn as randn * N/sqrt(F), so E||slot|| ~= N: parity with the sequence
    embedding norm measured on TRAINING data. Same RNG draw as the default, only the multiplier changes."""
    v = os.environ.get('MEMTRANAD_MEM_INIT_NORM')
    if v is None or str(v) == '': return 0.02, None
    n = float(v); assert n > 0, n
    return n / math.sqrt(feats), n


def decoder_variant_config():
    """Default-off decoder/encoder path switches for the Basic MemCtrl model (experiments/memory_diversity/decoder).
    Returns (query, keys, history_only, no_memory)."""
    env = os.environ
    q, k = env.get('MEMTRANAD_DECODER_QUERY') or 'target', env.get('MEMTRANAD_DECODER_KEYS') or 'all'
    h, nm = env.get('MEMTRANAD_HISTORY_ONLY', '') in ('1', 'true', 'True'), env.get('MEMTRANAD_NO_MEMORY', '') in ('1', 'true', 'True')
    assert q in ('target', 'learned') and k in ('all', 'memory', 'sequence'), (q, k)
    if nm: assert k == 'sequence', 'no_memory requires --decoder_keys sequence (there is no memory to read)'
    return q, k, h, nm


def memory_retrieval_config():
    """Default-off input-conditioned memory retrieval (experiments/memory_diversity/retrieval). Returns (enabled, dim, context)."""
    env = os.environ
    on = env.get('MEMTRANAD_MEMORY_RETRIEVAL', '') in ('1', 'true', 'True')
    dim = int(env.get('MEMTRANAD_RETRIEVAL_DIM') or 16)
    ctx = env.get('MEMTRANAD_RETRIEVAL_CONTEXT') or 'mean'
    assert ctx in ('mean', 'last') and dim > 0
    return on, dim, ctx

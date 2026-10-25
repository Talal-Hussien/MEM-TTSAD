"""Command-line interface.

The model code never imports this module: experiment switches are read from
``MEMTRANAD_*`` environment variables at model construction time, and
``bridge_args_to_env`` copies explicitly set CLI flags into those variables
(flag > environment > default — the original precedence).
"""
import argparse
import os


def build_parser():
    parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
    parser.add_argument('--dataset',
                        metavar='-d',
                        type=str,
                        required=False,
                        default='SMD',
                        help="dataset from ['SMD', 'SMAP', 'MSL', 'SWaT', 'WADI', ...]")
    parser.add_argument('--model',
                        metavar='-m',
                        type=str,
                        required=False,
                        default='TranAD',
                        help="model name")
    parser.add_argument('--test',
                        action='store_true',
                        help="test the model")
    parser.add_argument('--retrain',
                        action='store_true',
                        help="retrain the model")
    parser.add_argument('--less',
                        action='store_true',
                        help="train using less data")
    parser.add_argument('--memory_diversity',
                        type=str,
                        default=None,
                        choices=['none', 'raw', 'updated', 'both'],
                        help="memory-slot diversity regularisation (default none; env MEMTRANAD_DIVERSITY)")
    parser.add_argument('--diversity_lambda',
                        type=float,
                        default=None,
                        help="weight of the diversity loss for both raw and updated memory (env MEMTRANAD_DIV_LAMBDA)")
    parser.add_argument('--diversity_lambda_raw',
                        type=float,
                        default=None,
                        help="weight for the raw learnable slots; overrides --diversity_lambda")
    parser.add_argument('--diversity_lambda_updated',
                        type=float,
                        default=None,
                        help="weight for the updated memory Hmem read by the decoder; overrides --diversity_lambda")
    parser.add_argument('--memory_init_norm',
                        type=float,
                        default=None,
                        help="memory-token INITIALISATION scale parity: initial slots are randn * norm/sqrt(F) so that the "
                             "expected slot norm equals this constant (computed from TRAINING sequence embeddings); default "
                             "None keeps the original randn * 0.02 (env MEMTRANAD_MEM_INIT_NORM). Basic MemCtrl/MemBottleneck only; "
                             "initialisation only, the parameter is free afterwards")
    parser.add_argument('--decoder_query',
                        type=str, default=None, choices=['target', 'learned'],
                        help="Basic MemCtrl decoder query: 'target' (original: the window's last row) or 'learned' (a zero-initialised "
                             "learnable F-vector; the target no longer enters the decoder query/residual path). Env MEMTRANAD_DECODER_QUERY")
    parser.add_argument('--decoder_keys',
                        type=str, default=None, choices=['all', 'memory', 'sequence'],
                        help="Basic MemCtrl decoder cross-attention keys/values: 'all' (original [Hmem; Hseq]), 'memory' (Hmem only), "
                             "'sequence' (Hseq only). Env MEMTRANAD_DECODER_KEYS")
    parser.add_argument('--history_only',
                        action='store_true', default=False,
                        help="Basic MemCtrl: drop the window's last row from EVERY encoder input (memory update and sequence stream see "
                             "rows 0..W-2 only); with a learned query this is one-step prediction of the last row. Env MEMTRANAD_HISTORY_ONLY=1")
    parser.add_argument('--no_memory',
                        action='store_true', default=False,
                        help="Basic MemCtrl: delete the memory bank and memory layer after construction (same RNG draw as the memory model, "
                             "so all other initial weights are identical); the sequence stream attends to itself only. Env MEMTRANAD_NO_MEMORY=1")
    parser.add_argument('--memory_retrieval',
                        action='store_true', default=False,
                        help="Basic MemCtrl: input-conditioned memory retrieval. A context vector from the encoder sequence stream (Hseq) "
                             "attends over the m updated slots (Hmem); the retrieved vector is added to the decoder output through a "
                             "learnable per-feature gate (zero-initialised output projection: exact original function at initialisation). "
                             "Env MEMTRANAD_MEMORY_RETRIEVAL=1")
    parser.add_argument('--retrieval_dim', type=int, default=None, help="retrieval query/key projection width (default 16; env MEMTRANAD_RETRIEVAL_DIM)")
    parser.add_argument('--retrieval_context', type=str, default=None, choices=['mean', 'last'],
                        help="context vector: mean of Hseq over timesteps or Hseq at the last timestep (default mean; env MEMTRANAD_RETRIEVAL_CONTEXT)")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def bridge_args_to_env(args):
    """Copy explicitly set experiment flags into the MEMTRANAD_* environment
    variables the model constructors read. An unset flag leaves the variable
    alone, so the original CLI > environment > default precedence holds."""
    if args.memory_init_norm is not None:
        os.environ['MEMTRANAD_MEM_INIT_NORM'] = str(args.memory_init_norm)
    if args.decoder_query:
        os.environ['MEMTRANAD_DECODER_QUERY'] = args.decoder_query
    if args.decoder_keys:
        os.environ['MEMTRANAD_DECODER_KEYS'] = args.decoder_keys
    if args.history_only:
        os.environ['MEMTRANAD_HISTORY_ONLY'] = '1'
    if args.no_memory:
        os.environ['MEMTRANAD_NO_MEMORY'] = '1'
    if args.memory_retrieval:
        os.environ['MEMTRANAD_MEMORY_RETRIEVAL'] = '1'
    if args.retrieval_dim:
        os.environ['MEMTRANAD_RETRIEVAL_DIM'] = str(args.retrieval_dim)
    if args.retrieval_context:
        os.environ['MEMTRANAD_RETRIEVAL_CONTEXT'] = args.retrieval_context
    return args

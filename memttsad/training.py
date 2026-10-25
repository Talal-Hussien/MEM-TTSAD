"""Training/evaluation loop and checkpoint management for the TranAD protocol.

backprop() is the original two-phase training branch, verbatim: every kept
model is trained with windowed input, the adversarially weighted
(1/n, 1-1/n) phase losses and an AdamW + StepLR schedule.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from memttsad.config import get_lr
from memttsad.models import build_model
from memttsad.plotting import color
from memttsad.regularization import DiversityConfig, diversity_terms, memory_parameter

_DISABLED_DIV = DiversityConfig('none')

_DEFAULT_ENTITY = {'SMD': 'machine-1-1', 'SMAP': 'P-1', 'MSL': 'C-1',
                   'UCR': '136', 'NAB': 'ec2_request_latency_system_failure'}  # same defaults as load_dataset
# *_FULL datasets have no default entity: TRANAD_ENTITY is always set explicitly, so the
# checkpoint tag always carries it and no two entities can share a checkpoint.


def ckpt_tag(dataset):
    # Checkpoint directory tag. Without TRANAD_ENTITY this is exactly the
    # historical '{dataset}'; with an explicit entity it is
    # '{dataset}-{entity}' so entity runs never overwrite the default
    # checkpoint (same convention as the earlier per-entity checkpoints).
    ent = os.environ.get('TRANAD_ENTITY')
    return f'{dataset}-{ent}' if ent else dataset


def ckpt_dir(model, model_name, dataset, div=_DISABLED_DIV):
    # Experiment-unique checkpoint directory for memory models:
    #   checkpoints/<model>_<dataset>[_<entity>]_m<m>_seed<S>_e<E>_b<B>
    # so different m / seed / epoch / batch settings can never share a file.
    # Models without memory slots (e.g. TranAD) keep the legacy layout.
    m = getattr(model, 'n_memory', None)
    if m is None:
        return f'checkpoints/{model_name}_{ckpt_tag(dataset)}'
    ent = os.environ.get('TRANAD_ENTITY', _DEFAULT_ENTITY.get(dataset))
    ds = f'{dataset}_{ent}' if ent else dataset
    seed = int(os.environ.get('TRANAD_SEED', 1))
    epochs = int(os.environ.get('TRANAD_EPOCHS', 5))
    mn = getattr(model, 'mem_init_norm', None)
    msuf = f'_meminit{mn:g}' if mn else ''                       # only when scale parity is enabled: baselines keep their path
    dq, dk, ho, nm = getattr(model, 'decoder_query_mode', 'target'), getattr(model, 'decoder_keys_mode', 'all'), getattr(model, 'history_only', False), getattr(model, 'no_memory', False)
    msuf += ('' if dq == 'target' else f'_dq{dq}') + ('' if dk == 'all' else f'_dk{dk}') + ('_hist' if ho else '') + ('_nomem' if nm else '')   # only when a switch is on
    if getattr(model, 'memory_retrieval', False): msuf += f'_ret{model.retrieval_context}{model.retrieval_dim}'
    return f'checkpoints/{model_name}_{ds}_m{m}_seed{seed}_e{epochs}_b{model.batch}{div.ckpt_suffix()}{msuf}'


def save_model(model, optimizer, scheduler, epoch, accuracy_list,
               model_name, dataset, div=_DISABLED_DIV, div_history=None):
    folder = ckpt_dir(model, model_name, dataset, div) + '/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}
    if div.enabled:
        payload['memory_diversity'] = {'config': div.as_dict(), 'history': div_history or []}
    if getattr(model, 'mem_init_norm', None):
        payload['memory_init_norm'] = model.mem_init_norm
    if getattr(model, 'decoder_query_mode', 'target') != 'target' or getattr(model, 'decoder_keys_mode', 'all') != 'all' or getattr(model, 'history_only', False) or getattr(model, 'no_memory', False):
        payload['decoder_variant'] = {'decoder_query': model.decoder_query_mode, 'decoder_keys': model.decoder_keys_mode, 'history_only': model.history_only, 'no_memory': model.no_memory}
    if getattr(model, 'memory_retrieval', False):
        payload['memory_retrieval'] = {'retrieval_dim': model.retrieval_dim, 'retrieval_context': model.retrieval_context}
    torch.save(payload, file_path)


def load_model(modelname, dims, dataset, retrain=False, test=False, div=_DISABLED_DIV):
    model = build_model(modelname, dims, lr=get_lr(dataset)).double()
    if div.enabled and memory_parameter(model) is None:
        raise ValueError(f'--memory_diversity {div.mode} requires a memory model; {modelname} has no learnable memory bank')
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'{ckpt_dir(model, modelname, dataset, div)}/model.ckpt'
    legacy = f'checkpoints/{modelname}_{ckpt_tag(dataset)}/model.ckpt'
    if not os.path.exists(fname) and os.path.exists(legacy) and (not retrain or test):
        print(f"{color.GREEN}Using legacy checkpoint path {legacy}{color.ENDC}")  # backward compatibility
        fname = legacy
    if os.path.exists(fname) and (not retrain or test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True,
             div=_DISABLED_DIV, div_history=None):
    """One training epoch (training=True) or a full evaluation pass
    (training=False) under the TranAD protocol."""
    l = nn.MSELoss(reduction='none')
    feats = dataO.shape[1]
    data_x = torch.DoubleTensor(data)
    dataset = TensorDataset(data_x, data_x)
    # evaluation is chunked (stateless per window): identical outputs, bounded RAM
    bs = model.batch if training else len(data)
    dataloader = DataLoader(dataset, batch_size=bs)
    n = epoch + 1
    l1s = []
    if training:
        div_on = div.enabled
        if div_on:
            model.collect_memory_reprs = div.use_updated      # record Hmem during the SAME forward pass
            d_raw, d_upd, g_mem = [], [], []
        for d, _ in dataloader:
            local_bs = d.shape[0]
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, local_bs, feats)
            if div_on: model._memory_reprs = []
            z = model(window, elem)
            l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
            if isinstance(z, tuple): z = z[1]
            l1s.append(torch.mean(l1).item())
            loss = torch.mean(l1)
            if div_on:
                l_raw, l_upd = diversity_terms(model, div)
                loss = loss + div.lambda_raw * l_raw + div.lambda_updated * l_upd
                d_raw.append(l_raw.item()); d_upd.append(l_upd.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            if div_on: g_mem.append(memory_parameter(model).grad.norm().item())
            optimizer.step()
            if div_on: model._memory_reprs = []                 # drop the recorded Hmem (and its graph) right away
        scheduler.step()
        tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
        if div_on:
            model.collect_memory_reprs = False; model._memory_reprs = []
            rec = {'epoch': epoch, 'recon_loss': float(np.mean(l1s)), 'div_raw': float(np.mean(d_raw)), 'div_updated': float(np.mean(d_upd)),
                   'mem_grad_norm_mean': float(np.mean(g_mem)), 'mem_grad_norm_max': float(np.max(g_mem)),
                   'lambda_raw': div.lambda_raw, 'lambda_updated': div.lambda_updated}
            if div_history is not None: div_history.append(rec)
            tqdm.write(f'Epoch {epoch},\tL_div_raw = {rec["div_raw"]:.5f},\tL_div_updated = {rec["div_updated"]:.5f},\t|grad mem| = {rec["mem_grad_norm_mean"]:.3e}')
        return np.mean(l1s), optimizer.param_groups[0]['lr']
    else:
        for d, _ in dataloader:
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, bs, feats)
            z = model(window, elem)
            if isinstance(z, tuple): z = z[1]
        loss = l(z, elem)[0]
        return loss.detach().numpy(), z.detach().numpy()[0]

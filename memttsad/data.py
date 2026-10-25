"""Dataset loading and windowing (unchanged logic from the original pipeline)."""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from memttsad.config import output_folder, uses_tranad_protocol
from memttsad.plotting import color


def convert_to_windows(data, model):
    """Sliding windows of length model.n_window; positions before the first
    full window are left-padded by repeating the first timestamp."""
    windows = []
    w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(w if uses_tranad_protocol(model.name) else w.view(-1))
    return torch.stack(windows)


def cut_array(percentage, arr):
    print(f'{color.BOLD}Slicing dataset to {int(percentage * 100)}%{color.ENDC}')
    mid = round(arr.shape[0] / 2)
    window = round(arr.shape[0] * percentage * 0.5)
    return arr[mid - window: mid + window, :]


def load_dataset(dataset, less=False):
    """Load {train,test,labels}.npy for the dataset (entity-prefixed where the
    dataset is multi-entity; TRANAD_ENTITY selects the entity)."""
    folder = os.path.join(output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        if dataset == 'SMD': file = os.environ.get('TRANAD_ENTITY', 'machine-1-1') + '_' + file
        if dataset == 'SMAP': file = os.environ.get('TRANAD_ENTITY', 'P-1') + '_' + file
        if dataset == 'MSL': file = os.environ.get('TRANAD_ENTITY', 'C-1') + '_' + file
        # entity-complete FULL datasets: TRANAD_ENTITY is REQUIRED, no default, one model per entity
        if dataset in ('SMD_FULL', 'SMAP_FULL', 'MSL_FULL'):
            ent = os.environ.get('TRANAD_ENTITY')
            if not ent:
                raise Exception(f'{dataset} requires TRANAD_ENTITY (entities are listed in processed/{dataset}/manifest.json)')
            file = ent + '_' + file
        if dataset == 'UCR': file = os.environ.get('TRANAD_ENTITY', '136') + '_' + file
        if dataset == 'NAB': file = os.environ.get('TRANAD_ENTITY', 'ec2_request_latency_system_failure') + '_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    if less: loader[0] = cut_array(0.2, loader[0])
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]
    return train_loader, test_loader, labels

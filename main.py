"""Train and evaluate a Mem-TTSAD model.

Usage:
    python main.py --model TranAD --dataset SMD --retrain
    MEMTRANAD_M=4 TRANAD_SEED=1 python main.py --model Mem_TTSAD --dataset SMD --retrain
"""
from time import time
from pprint import pprint

from memttsad.cli import parse_args, bridge_args_to_env

args = bridge_args_to_env(parse_args())

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from memttsad.config import get_lm
from memttsad.data import load_dataset, convert_to_windows
from memttsad.evaluation import pot_eval, hit_att, ndcg
from memttsad.plotting import color, plot_accuracies, plotter
from memttsad.regularization import DiversityConfig
from memttsad.training import backprop, load_model, save_model

# Memory-slot diversity regularisation (disabled unless --memory_diversity / MEMTRANAD_DIVERSITY is set).
DIV = DiversityConfig.from_args(args)
DIV_HISTORY = []   # per-epoch diversity diagnostics, saved with the checkpoint only when enabled


if __name__ == '__main__':
    train_loader, test_loader, labels = load_dataset(args.dataset, args.less)
    model, optimizer, scheduler, epoch, accuracy_list = load_model(
        args.model, labels.shape[1], args.dataset, retrain=args.retrain, test=args.test, div=DIV)

    ## Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
    print(trainD.shape, testD.shape)

    ### Training phase
    if not args.test:
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        num_epochs = int(os.environ.get('TRANAD_EPOCHS', 5))
        e = epoch + 1
        start = time()
        for e in tqdm(list(range(epoch + 1, epoch + num_epochs + 1))):
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler,
                                 div=DIV, div_history=DIV_HISTORY)
            accuracy_list.append((lossT, lr))
        print(color.BOLD + 'Training time: ' + "{:10.4f}".format(time() - start) + ' s' + color.ENDC)
        save_model(model, optimizer, scheduler, e, accuracy_list,
                   args.model, args.dataset, div=DIV, div_history=DIV_HISTORY)
        plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

    ### Testing phase
    model.eval()
    print(f'Parameter count: {sum(p.numel() for p in model.parameters())}')
    print(f'n_memory: {getattr(model, "n_memory", None)}')
    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
    test_start = time()
    with torch.no_grad():
        loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
    print(color.BOLD + 'Testing time: ' + "{:10.4f}".format(time() - test_start) + ' s' + color.ENDC)

    ### Plot curves
    if not args.test:
        testO = torch.roll(testO, 1, 0)   # one-step score alignment (TranAD protocol)
        plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

    ### Scores
    lm = get_lm(args.dataset, args.model)
    preds = []
    df = pd.DataFrame()
    with torch.no_grad():
        lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    for i in range(loss.shape[1]):
        _t0 = time()
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        result, pred = pot_eval(lt, l, ls, lm)
        print(f'POT dim {i + 1}/{loss.shape[1]} ({time() - _t0:.1f}s)', flush=True)
        preds.append(pred)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal, lm)
    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))
    print(df)
    pprint(result)

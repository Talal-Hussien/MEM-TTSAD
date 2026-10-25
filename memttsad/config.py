"""Dataset hyperparameters and the TranAD training-protocol predicate.

Values are inherited verbatim from the original TranAD code base (src/constants.py);
changing them changes thresholds and learning rates for every experiment.
"""

# Folder with the preprocessed .npy datasets ({train,test,labels}.npy per entity)
# and the raw-dataset folder read by preprocess.py.
output_folder = 'processed'
data_folder = 'data'

# POT threshold parameters per dataset: [(level, lm_factor) non-protocol, (level, lm_factor) protocol].
lm_d = {
    'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
    'SMD_FULL': [(0.99995, 1.04), (0.99995, 1.06)],   # mirrors SMD exactly (entity-complete variant)
    'synthetic': [(0.999, 1), (0.999, 1)],
    'SWaT': [(0.993, 1), (0.993, 1)],
    'SWaT_no_clip': [(0.993, 1), (0.993, 1)],
    'UCR': [(0.993, 1), (0.99935, 1)],
    'NAB': [(0.991, 1), (0.99, 1)],
    'SMAP': [(0.98, 1), (0.98, 1)],
    'SMAP_FULL': [(0.98, 1), (0.98, 1)],   # mirrors SMAP exactly (entity-complete variant)
    'MSL': [(0.97, 1), (0.999, 1.04)],
    'MSL_FULL': [(0.97, 1), (0.999, 1.04)],   # mirrors MSL exactly (entity-complete variant)
    'WADI': [(0.99, 1), (0.999, 1)],
    'MSDS': [(0.91, 1), (0.9, 1.04)],
    'MSDS_FULL': [(0.91, 1), (0.9, 1.04)],   # paper-scale MSDS (stride 1); same POT settings as MSDS
    'MBA': [(0.87, 1), (0.93, 1.04)],
    'PSM': [(0.993, 1), (0.993, 1)],   # not in the TranAD paper; PROVISIONAL start = SWaT setting (27.8% anomaly ratio, so SMD's 0.99995 level would be far too strict)
}

# Learning rate per dataset.
lr_d = {
    'SMD': 0.0001,
    'SMD_FULL': 0.0001,   # mirrors SMD exactly (entity-complete variant)
    'synthetic': 0.0001,
    'SWaT': 0.008,
    'SWaT_no_clip': 0.008,
    'SMAP': 0.001,
    'SMAP_FULL': 0.001,   # mirrors SMAP exactly (entity-complete variant)
    'MSL': 0.002,
    'MSL_FULL': 0.002,   # mirrors MSL exactly (entity-complete variant)
    'WADI': 0.0001,
    'MSDS': 0.001,
    'MSDS_FULL': 0.001,
    'UCR': 0.006,
    'NAB': 0.009,
    'MBA': 0.001,
    'PSM': 0.0001,   # provisional: SMD value (server-metrics analogue)
}

# Models trained and evaluated with the TranAD protocol: windowed input, the
# two-phase backprop branch, the second POT setting and the one-step score
# alignment. The memory models carry TTSAD in their names, the hosts TranAD;
# both must take this path, so the test is on either token rather than on
# 'TranAD' alone.
PROTOCOL_NAME_TOKENS = ('TranAD', 'TTSAD')


def uses_tranad_protocol(model_name):
    return any(token in model_name for token in PROTOCOL_NAME_TOKENS)


def get_lr(dataset):
    """Learning rate for the dataset (originally the module-level `lr`)."""
    return lr_d[dataset]


def get_lm(dataset, model_name):
    """(level, factor) POT setting for the dataset/model (originally the module-level `lm`)."""
    return lm_d[dataset][1 if uses_tranad_protocol(model_name) else 0]

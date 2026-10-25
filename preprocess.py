import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from memttsad.config import output_folder, data_folder
from shutil import copyfile

datasets = ['synthetic', 'SMD', 'SWaT', 'SWaT_no_clip', 'SWaT_excerpt', 'SMAP', 'MSL', 'WADI', 'MSDS', 'MSDS_FULL', 'UCR', 'MBA', 'NAB', 'PSM']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape):
	temp = np.zeros(shape)
	with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
		ls = f.readlines()
	for line in ls:
		pos, values = line.split(':')[0], line.split(':')[1].split(',')
		start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
		temp[start-1:end-1, indx] = 1
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def normalize(a):
	a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
	return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = min(a), max(a)
	return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def convertNumpy(df):
	x = df[df.columns[3:]].values[::10, :]
	return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def load_data(dataset):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	if dataset == 'synthetic':
		train_file = os.path.join(data_folder, dataset, 'synthetic_data_with_anomaly-s-1.csv')
		test_labels = os.path.join(data_folder, dataset, 'test_anomaly.csv')
		dat = pd.read_csv(train_file, header=None)
		split = 10000
		train = normalize(dat.values[:, :split].reshape(split, -1))
		test = normalize(dat.values[:, split:].reshape(split, -1))
		lab = pd.read_csv(test_labels, header=None)
		lab[0] -= split
		labels = np.zeros(test.shape)
		for i in range(lab.shape[0]):
			point = lab.values[i][0]
			labels[point-30:point+30, lab.values[i][1:]] = 1
		test += labels * np.random.normal(0.75, 0.1, test.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'SMD':
		dataset_folder = 'data/SMD'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
				s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
				load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
	elif dataset == 'UCR':
		dataset_folder = 'data/UCR'
		file_list = os.listdir(dataset_folder)
		for filename in file_list:
			if not filename.endswith('.txt'): continue
			vals = filename.split('.')[0].split('_')
			dnum, vals = int(vals[0]), vals[-3:]
			vals = [int(i) for i in vals]
			temp = np.genfromtxt(os.path.join(dataset_folder, filename),
								dtype=np.float64,
								delimiter=',')
			min_temp, max_temp = np.min(temp), np.max(temp)
			temp = (temp - min_temp) / (max_temp - min_temp)
			train, test = temp[:vals[0]], temp[vals[0]:]
			labels = np.zeros_like(test)
			labels[vals[1]-vals[0]:vals[2]-vals[0]] = 1
			train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
			for file in ['train', 'test', 'labels']:
				np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))
	elif dataset == 'NAB':
		dataset_folder = 'data/NAB'
		file_list = os.listdir(dataset_folder)
		with open(dataset_folder + '/labels.json') as f:
			labeldict = json.load(f)
		for filename in file_list:
			if not filename.endswith('.csv'): continue
			df = pd.read_csv(dataset_folder+'/'+filename)
			vals = df.values[:,1]
			labels = np.zeros_like(vals, dtype=np.float64)
			for timestamp in labeldict['realKnownCause/'+filename]:
				tstamp = timestamp.replace('.000000', '')
				index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
				labels[index-4:index+4] = 1
			min_temp, max_temp = np.min(vals), np.max(vals)
			vals = (vals - min_temp) / (max_temp - min_temp)
			train, test = vals.astype(float), vals.astype(float)
			train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
			fn = filename.replace('.csv', '')
			for file in ['train', 'test', 'labels']:
				np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
	elif dataset == 'MSDS':
		dataset_folder = 'data/MSDS'
		df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
		df_test  = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
		df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
		_, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
		train, _, _ = normalize3(df_train, min_a, max_a)
		test, _, _ = normalize3(df_test, min_a, max_a)
		labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
		labels = labels.values[::1, 1:]
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
	elif dataset == 'MSDS_FULL':
		# Paper-scale MSDS (stride 1). Identical raw files, columns and
		# normalization to 'MSDS' (train+test min-max, as in the original
		# TranAD code); the only differences are (a) no [::5] stride, so
		# train/test keep their 146,430 rows each (the counts in TranAD paper
		# Table 1), and (b) the 29,286-row labels.csv shipped by the TranAD
		# authors is expanded x5 so that label row k covers raw rows
		# 5k..5k+4 (raw row 5k is the row the stride-5 protocol scored with
		# label k). The Zenodo release has no 1 Hz ground truth for this
		# period, so these are DERIVED labels. Details are written to
		# processed/MSDS_FULL/preprocess_report.json. 'MSDS' is untouched.
		import hashlib, datetime
		dataset_folder = 'data/MSDS'
		df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
		df_test  = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
		df_labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
		raw_train, raw_test = df_train.values[:, 1:].astype('float64'), df_test.values[:, 1:].astype('float64')
		_, min_a, max_a = normalize3(np.concatenate((raw_train, raw_test), axis=0))
		train, _, _ = normalize3(raw_train, min_a, max_a)
		test, _, _ = normalize3(raw_test, min_a, max_a)
		labels5 = df_labels.values[:, 1:].astype('float64')
		assert labels5.shape[0] * 5 == raw_test.shape[0], (labels5.shape, raw_test.shape)
		labels = np.repeat(labels5, 5, axis=0)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
		md5 = lambda fn: hashlib.md5(open(os.path.join(dataset_folder, fn), 'rb').read()).hexdigest()
		report = {
			'dataset': 'MSDS_FULL',
			'preprocessing_timestamp': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
			'raw_source': {fn: {'path': os.path.join(dataset_folder, fn), 'md5': md5(fn)} for fn in ['train.csv', 'test.csv', 'labels.csv']},
			'raw_origin': 'Zenodo record 3549604, concurrent data/metrics/wally{113,117,122,123,124}_metrics_concurrent.csv -> data/MSDS/clean.py (drop load.*, keep cpu.user/mem.used per host, discard first 10%, 50/50 split)',
			'feature_columns': list(df_train.columns[1:]),
			'timestamp_column': df_train.columns[0],
			'raw_shapes': {'train': list(df_train.shape), 'test': list(df_test.shape), 'labels': list(df_labels.shape)},
			'processed_shapes': {'train': list(train.shape), 'test': list(test.shape), 'labels': list(labels.shape)},
			'feature_count': int(train.shape[1]),
			'sampling': {'stride': 1, 'note': "dataset 'MSDS' uses values[::5] on train/test; MSDS_FULL keeps every row"},
			'window': {'size': 10, 'stride': 1, 'target': 'row i is scored against the window of rows i-10..i-1; first 10 windows are padded with row 0 (main.py convert_to_windows)'},
			'normalization': 'per-feature min-max, (x - min) / (max - min + 1e-4), statistics from concatenated train+test (original TranAD normalize3; test leakage inherited by design)',
			'label_mapping': 'labels.csv row k (29,286 rows, per-feature 0/1, provenance undocumented by TranAD authors) assigned to raw test rows 5k..5k+4 via np.repeat(labels, 5, axis=0); DERIVED, not ground truth',
			'label_stats': {'rows_any_feature': int((labels.sum(1) >= 1).sum()), 'rows_total': int(labels.shape[0]), 'cell_fraction': float(labels.mean())},
			'time_span': {'train': [str(df_train.iloc[0, 0]), str(df_train.iloc[-1, 0])], 'test': [str(df_test.iloc[0, 0]), str(df_test.iloc[-1, 0])]},
		}
		with open(os.path.join(folder, 'preprocess_report.json'), 'w') as f:
			json.dump(report, f, indent=2)
		print(json.dumps(report['raw_shapes']), json.dumps(report['processed_shapes']))
	elif dataset == 'SWaT':
		# Official iTrust SWaT (Dec 2015): 51 sensor/actuator channels.
		# Normal_v1 = 7-day normal operation (train); Attack_v0 = attack
		# campaign with Normal/Attack labels (test). Header is on sheet
		# row 2; data starts at row 3. The official Attack_v0 file contains
		# the label typo 'A ttack', mapped to Attack here. Pipeline v2:
		# the first 21,600 s of Normal_v1 are discarded (plant stabilization,
		# literature-standard trim; 21/51 channels had their min/max set by
		# this period), stride-10 downsampling is applied identically to
		# train, test and labels, normalization uses train statistics only,
		# and test is clipped to [0, 1] -- the model's representable output
		# range (v1's [-1, 2] clip left 88% of normal test rows with
		# unreachable reconstruction targets, inverting the ranking).
		# The historical univariate excerpt remains as dataset 'SWaT_excerpt'.
		from openpyxl import load_workbook
		dataset_folder = 'data/SWaT'
		STRIDE = 10
		TRIM = 21600  # stabilization rows dropped from the start of Normal_v1

		def read_swat_xlsx(fname, with_labels, skip=0):
			wb = load_workbook(os.path.join(dataset_folder, fname), read_only=True)
			ws = wb[wb.sheetnames[0]]
			rows_iter = ws.iter_rows(min_row=2, values_only=True)
			header = [str(h).strip() if h is not None else '' for h in next(rows_iter)]
			assert header[0] == 'Timestamp' and header[-1] == 'Normal/Attack', header[:2] + header[-2:]
			feats, labs = [], []
			for i, row in enumerate(rows_iter):
				if row[0] is None: continue
				if i < skip: continue
				if (i - skip) % STRIDE: continue
				feats.append([float(v) for v in row[1:-1]])
				if with_labels:
					lab = str(row[-1]).strip().replace(' ', '')  # 'A ttack' -> 'Attack'
					labs.append(1.0 if lab == 'Attack' else 0.0)
			wb.close()
			return np.asarray(feats, dtype=np.float64), np.asarray(labs, dtype=np.float64)

		train, _ = read_swat_xlsx('SWaT_Dataset_Normal_v1.xlsx', with_labels=False, skip=TRIM)
		test, lab = read_swat_xlsx('SWaT_Dataset_Attack_v0.xlsx', with_labels=True)
		assert train.shape[1] == 51 and test.shape[1] == 51, (train.shape, test.shape)
		train, min_a, max_a = normalize3(train)
		test, _, _ = normalize3(test, min_a, max_a)
		test = np.clip(test, 0.0, 1.0)
		labels = np.repeat(lab.reshape(-1, 1), test.shape[1], axis=1)  # repo convention (as SMAP/MSL)
		print('SWaT official:', train.shape, test.shape, labels.shape,
			  f'anomaly%={100 * lab.mean():.2f}')
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'SWaT_no_clip':
		# Clipping ablation of the official-SWaT v2 pipeline: identical in
		# every respect (Normal_v1 minus first 21,600 s stabilization rows,
		# stride-10, per-column normalize3 with train stats only, 'A ttack'
		# typo normalized) EXCEPT that test is NOT clipped. Constant-in-train
		# channels therefore map test deviations to +-1/eps = +-10000; test
		# spans [-10000, 10000] with 9.8% of cells outside [0,1]. Saved to
		# processed/SWaT_no_clip; the main SWaT (v2) output is untouched.
		from openpyxl import load_workbook
		dataset_folder = 'data/SWaT'
		STRIDE = 10
		TRIM = 21600  # stabilization rows dropped from the start of Normal_v1

		def read_swat_xlsx(fname, with_labels, skip=0):
			wb = load_workbook(os.path.join(dataset_folder, fname), read_only=True)
			ws = wb[wb.sheetnames[0]]
			rows_iter = ws.iter_rows(min_row=2, values_only=True)
			header = [str(h).strip() if h is not None else '' for h in next(rows_iter)]
			assert header[0] == 'Timestamp' and header[-1] == 'Normal/Attack', header[:2] + header[-2:]
			feats, labs = [], []
			for i, row in enumerate(rows_iter):
				if row[0] is None: continue
				if i < skip: continue
				if (i - skip) % STRIDE: continue
				feats.append([float(v) for v in row[1:-1]])
				if with_labels:
					lab = str(row[-1]).strip().replace(' ', '')  # 'A ttack' -> 'Attack'
					labs.append(1.0 if lab == 'Attack' else 0.0)
			wb.close()
			return np.asarray(feats, dtype=np.float64), np.asarray(labs, dtype=np.float64)

		train, _ = read_swat_xlsx('SWaT_Dataset_Normal_v1.xlsx', with_labels=False, skip=TRIM)
		test, lab = read_swat_xlsx('SWaT_Dataset_Attack_v0.xlsx', with_labels=True)
		assert train.shape[1] == 51 and test.shape[1] == 51, (train.shape, test.shape)
		train, min_a, max_a = normalize3(train)
		test, _, _ = normalize3(test, min_a, max_a)
		labels = np.repeat(lab.reshape(-1, 1), test.shape[1], axis=1)  # repo convention (as SMAP/MSL)
		print('SWaT_no_clip:', train.shape, test.shape, labels.shape,
			  f'anomaly%={100 * lab.mean():.2f}')
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'SWaT_excerpt':
		# Historical TranAD-repo univariate excerpt (kept for reproducibility
		# of pre-official-SWaT logs; processed output lives in SWaT_old).
		dataset_folder = 'data/SWaT'
		file = os.path.join(dataset_folder, 'series.json')
		df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
		df_test  = pd.read_json(file, lines=True)[['val']][7000:12000]
		train, min_a, max_a = normalize2(df_train.values)
		test, _, _ = normalize2(df_test.values, min_a, max_a)
		labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset in ['SMAP', 'MSL']:
		dataset_folder = 'data/SMAP_MSL'
		file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
		values = pd.read_csv(file)
		values = values[values['spacecraft'] == dataset]
		filenames = values['chan_id'].values.tolist()
		for fn in filenames:
			train = np.load(f'{dataset_folder}/train/{fn}.npy')
			test = np.load(f'{dataset_folder}/test/{fn}.npy')
			train, min_a, max_a = normalize3(train)
			test, _, _ = normalize3(test, min_a, max_a)
			np.save(f'{folder}/{fn}_train.npy', train)
			np.save(f'{folder}/{fn}_test.npy', test)
			labels = np.zeros(test.shape)
			indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
			indices = indices.replace(']', '').replace('[', '').split(', ')
			indices = [int(i) for i in indices]
			for i in range(0, len(indices), 2):
				labels[indices[i]:indices[i+1], :] = 1
			np.save(f'{folder}/{fn}_labels.npy', labels)
	elif dataset == 'WADI':
		dataset_folder = 'data/WADI'
		ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
		train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
		test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
		train.dropna(how='all', inplace=True); test.dropna(how='all', inplace=True)
		train.fillna(0, inplace=True); test.fillna(0, inplace=True)
		test['Time'] = test['Time'].astype(str)
		test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
		labels = test.copy(deep = True)
		for i in test.columns.tolist()[3:]: labels[i] = 0
		for i in ['Start Time', 'End Time']:
			ls[i] = ls[i].astype(str)
			ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
		for index, row in ls.iterrows():
			to_match = row['Affected'].split(', ')
			matched = []
			for i in test.columns.tolist()[3:]:
				for tm in to_match:
					if tm in i:
						matched.append(i); break
			st, et = str(row['Start Time']), str(row['End Time'])
			labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
		train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
		print(train.shape, test.shape, labels.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'MBA':
		dataset_folder = 'data/MBA'
		ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
		train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
		test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
		train, test = train.values[1:,1:].astype(float), test.values[1:,1:].astype(float)
		train, min_a, max_a = normalize3(train)
		test, _, _ = normalize3(test, min_a, max_a)
		ls = ls.values[:,1].astype(int)
		labels = np.zeros_like(test)
		for i in range(-20, 20):
			labels[ls + i, :] = 1
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'PSM':
		# eBay Pooled Server Metrics (Abdulaal et al., KDD 2021, RANSynCoders
		# release): single entity, 25 features, 1-min resolution. train.csv
		# (132,481 rows) / test.csv (87,841 rows) / test_label.csv. The
		# publishers already min-max scaled every feature to [0, 1], so no
		# further normalization is applied (values stay in the model's
		# representable range). train.csv contains 4,195 NaN cells (2,697
		# rows, 12 features); test.csv has none. NaNs are replaced with 0,
		# the convention of the Anomaly Transformer PSM loader
		# (np.nan_to_num). The single label column is repeated across the
		# 25 features (repo convention, as SWaT/SMAP/MSL). Provenance is
		# written to processed/PSM/preprocess_report.json.
		import hashlib, datetime
		dataset_folder = 'data/PSM'
		def md5(fn):
			with open(os.path.join(dataset_folder, fn), 'rb') as f: return hashlib.md5(f.read()).hexdigest()
		df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
		df_test = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
		df_labels = pd.read_csv(os.path.join(dataset_folder, 'test_label.csv'))
		assert (df_labels.iloc[:, 0].values == df_test.iloc[:, 0].values).all(), 'PSM label/test timestamps differ'
		n_nan_train, n_nan_test = int(df_train.isna().sum().sum()), int(df_test.isna().sum().sum())
		train = np.nan_to_num(df_train.values[:, 1:].astype('float64'), nan=0.0)
		test = np.nan_to_num(df_test.values[:, 1:].astype('float64'), nan=0.0)
		lab = df_labels['label'].values.astype('float64')
		labels = np.repeat(lab.reshape(-1, 1), test.shape[1], axis=1)
		assert train.shape[1] == test.shape[1] == labels.shape[1] == 25
		assert test.shape[0] == labels.shape[0]
		print('PSM:', train.shape, test.shape, labels.shape,
			  f'anomaly ratio {lab.mean():.4f}; train NaNs filled {n_nan_train}; test NaNs {n_nan_test}')
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
		report = {
			'dataset': 'PSM', 'created': datetime.datetime.now().isoformat(timespec='seconds'),
			'raw_source': {fn: {'path': os.path.join(dataset_folder, fn), 'md5': md5(fn)} for fn in ['train.csv', 'test.csv', 'test_label.csv']},
			'raw_shapes': {'train': list(df_train.shape), 'test': list(df_test.shape), 'labels': list(df_labels.shape)},
			'processed_shapes': {'train': list(train.shape), 'test': list(test.shape), 'labels': list(labels.shape)},
			'columns_dropped': [df_train.columns[0]],
			'normalization': 'none (publisher-scaled to [0, 1]; train min/max %.4f/%.4f, test min/max %.4f/%.4f)' % (train.min(), train.max(), test.min(), test.max()),
			'nan_handling': {'method': 'np.nan_to_num -> 0.0', 'train_cells_filled': n_nan_train, 'train_rows_affected': int(df_train.isna().any(axis=1).sum()),
							 'train_cells_per_feature': {k: int(v) for k, v in df_train.isna().sum().items() if v > 0}, 'test_cells': n_nan_test},
			'label_mapping': 'test_label.csv single 0/1 column repeated across the 25 features (np.repeat axis=1)',
			'label_stats': {'anomalous_rows': int(lab.sum()), 'rows_total': int(lab.shape[0]), 'anomaly_ratio': float(lab.mean()),
							'segments': int((np.diff(np.r_[0, lab, 0]) == 1).sum())},
		}
		with open(os.path.join(folder, 'preprocess_report.json'), 'w') as f:
			json.dump(report, f, indent=2)
	else:
		raise Exception(f'Not Implemented. Check one of {datasets}')

if __name__ == '__main__':
	commands = sys.argv[1:]
	load = []
	if len(commands) > 0:
		for d in commands:
			load_data(d)
	else:
		print("Usage: python preprocess.py <datasets>")
		print(f"where <datasets> is space separated list of {datasets}")



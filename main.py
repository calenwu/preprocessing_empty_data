import multiprocessing
import math
import time

import numpy as np
import pandas as pd

from scipy import stats

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import RidgeCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
# Logistic Regression
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


sample = pd.read_csv('../sample.csv', delimiter=',')
test_features = pd.read_csv('../test_features.csv', delimiter=',').sort_values(['pid'], ascending=[True])
train_features = pd.read_csv('../train_features.csv', delimiter=',').sort_values(['pid', 'Time'], ascending=[True, True])
train_labels = pd.read_csv('../train_labels.csv', delimiter=',').sort_values(['pid'], ascending=[True])
train_labels.set_index('pid', inplace=True)

classification_labels = [
	'LABEL_BaseExcess',
	'LABEL_Fibrinogen',
	'LABEL_AST',
	'LABEL_Alkalinephos',
	'LABEL_Bilirubin_total',
	'LABEL_Lactate',
	'LABEL_TroponinI',
	'LABEL_SaO2',
	'LABEL_Bilirubin_direct',
	'LABEL_EtCO2',
	'LABEL_Sepsis',
]
continuous_labels = [
	'LABEL_RRate',
	'LABEL_ABPm',
	'LABEL_SpO2',
	'LABEL_Heartrate'
]

features = train_features.columns[3:]
last_features = ['{}_11'.format(feat) for feat in features]
adjusted_features = ['pid', 'Age']
for feature in features:
	for i in range(12):
		adjusted_features.append('{}_{}'.format(feature, i))


def empty_feature_dict() -> dict[str, list[np.float64]]:
	empty_row_features = {}
	for feature in features:
		empty_row_features[feature] = []
	return empty_row_features


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
	new_df = []
	last_pid = -1
	for _, row in df.iterrows():
		if row['pid'] != last_pid:
			if last_pid != -1:
				print(len(new_df))
				new_df.append([*new_row, *new_row_from_features(new_row_features)])
			new_row_features = empty_feature_dict()
			new_row = [row['pid'], row['Age']]
		for feature in features:
			new_row_features[feature].append(row[feature])
		last_pid = row['pid']
	new_df.append([*new_row, *new_row_from_features(new_row_features)])
	return new_df


def parallelize_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
	cores = 16
	total_rows = df.shape[0]
	chunk_size = math.ceil(total_rows / 12 / cores)
	chunks = []
	for x in range(cores):
		start = (x * chunk_size * 12)
		end = (x + 1) * chunk_size * 12
		chunks.append(df.iloc[start:end, :])
	pool = multiprocessing.Pool(cores, maxtasksperchild=100)
	results = pool.map(preprocess, chunks)
	pool.close()
	pool.join()
	temp = []
	for x in results:
		temp.extend(x)
	return pd.DataFrame(temp, columns=adjusted_features)


def preprocess_train() -> None:
	t1 = time.time()
	df = parallelize_preprocessing(train_features)
	df.to_csv('preprocessed_train_0.csv', index=False)
	print(time.time() - t1)


def preprocess_test() -> None:
	t1 = time.time()
	df = parallelize_preprocessing(test_features)
	df.to_csv('preprocessed_test_0.csv', index=False)
	print(time.time() - t1)


def new_row_from_features(new_features: dict) -> list[np.float64]:
	new_row = []
	for feature in features:
		if np.isnan(new_features[feature]).all():
			# new_row.append(0)
			# new_row.append(train_features[feature].mean())
			new_row.extend([0 for x in range(12)])
		else:
			new_row.extend(pd.Series(new_features[feature], dtype='float64').interpolate(limit_direction='both').values)
			# new_row.append(pd.Series(new_features[feature], dtype='float64').mean())
	return new_row


def get_best_feature(X: pd.DataFrame, y: pd.DataFrame):
	X_new = SelectKBest(f_classif, k=2).fit_transform(X, y)
	print(X_new.shape)
	print(X_new.head())


def get_model(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> SVC:
	# model_svc = SVC(kernel='sigmoid', random_state=1, probability=False)
	t1 = time.time()
	# model_svc = SVC(kernel='sigmoid', random_state=1, probability=True)
	# model_svc = RandomForestClassifier(random_state=1)
	model_svc = LogisticRegression(random_state=1, max_iter=20000)
	model_svc.fit(x_train, y_train)

	# sc = StandardScaler()
	# sc.fit(x_train)
	# x_test = sc.transform(x_test)
	
	# y_pred = model_svc.predict(x_test)
	# print((y_test - y_pred).abs().sum())

	y_pred = model_svc.predict(x_test)
	print(accuracy_score(y_test, y_pred))
	print(classification_report(y_test, y_pred))
	print(confusion_matrix(y_test, y_pred))
	print(time.time() - t1)
	return model_svc


def get_model_for_all_class_labels(X: pd.DataFrame, y: pd.DataFrame) -> dict[str, SVC]:
	model_mapping = {}
	for label in classification_labels:
		t1 = time.time()
		x_train, x_test, y_train, y_test = train_test_split(X.values, y[label].values, test_size=0.2, random_state=1)
		model_mapping[label] = get_model(x_train, x_test, y_train, y_test)
		print(time.time() - t1)
	return model_mapping


def get_coeffs(X: np.ndarray, y: np.ndarray) -> np.ndarray:
	from scipy.linalg import lu_factor, lu_solve
	# clf = RidgeCV(alphas=[0.001, 0.01, 1, 10, 100, 200])
	# clf = LassoCV()

	# x = X
	# y = y
	# xtx = np.matmul(x.T, x)
	# lam_i = np.identity(xtx.shape[0]) * 0.1
	# coeffs = np.linalg.solve(xtx + lam_i, np.matmul(x.T, y))

	clf = RidgeCV(alphas=[0.001, 0.01, 1, 10, 100, 200], fit_intercept=False)
	clf.fit(X, y) 
	coeffs = clf.coef_



	print(np.average(y))
	print(np.average(np.matmul(X, coeffs)))
	# clf = LinearRegression()
	# clf.fit(X[0:36, :], y[0:36]) 
	# print(np.average(y[0:30]))
	# print('score: ' + str(clf.score(X, y)))
	# ret = clf.coef_
	# pred = np.matmul(X, ret)
	# print('actual avg:' + str(np.average(y)))
	# print('avg: ' + str(np.sum(pred)/pred.size))
	# diff = np.subtract(y, pred)
	# print(math.sqrt(np.matmul(diff.T, diff)))
	return coeffs


def get_coeffs_for_all_labels(X: pd.DataFrame, y: pd.DataFrame) -> dict[str, np.array]:
	mapping = {}
	for label in continuous_labels:
		mapping[label] = get_coeffs(X, y[label])
	return mapping


def train() -> tuple[dict[str, SVC], dict[str, np.ndarray]]:
	df = pd.read_csv('./preprocessed_avg_test.csv', delimiter=',')
	df.set_index('pid', inplace=True)
	return (get_model_for_all_class_labels(df, train_labels), get_coeffs_for_all_labels(df, train_labels))


def parallelize_train() -> tuple[dict[str, SVC], dict[str, np.ndarray]]:
	cores = 16
	df_class = pd.read_csv('./preprocessed_train_0.csv', delimiter=',')
	df_cont = pd.read_csv('./preprocessed_train_median.csv', delimiter=',')
	df_class.set_index('pid', inplace=True)
	df_cont.set_index('pid', inplace=True)
	X = df_class
	y = train_labels


	mapping_cont = {}
	xs = []
	ys = []
	for label in continuous_labels:
		xs.append(X.values)
		ys.append(y[label].values)
	pool = multiprocessing.Pool(cores, maxtasksperchild=100)
	results = pool.starmap(get_coeffs, zip(xs, ys))
	pool.close()
	pool.join()
	for x in range(len(results)):
		mapping_cont[continuous_labels[x]] = results[x]


	x_trains = []
	x_tests = []
	y_trains = []
	y_tests = []
	for label in classification_labels:
		x_train, x_test, y_train, y_test = train_test_split(X.values, y[label].values, test_size=0.2, random_state=0)
		x_trains.append(x_train)
		x_tests.append(x_test)
		y_trains.append(y_train)
		y_tests.append(y_test)
	pool = multiprocessing.Pool(cores, maxtasksperchild=100)
	results = pool.starmap(get_model, zip(x_trains, x_tests, y_trains, y_tests))
	pool.close()
	pool.join()
	mapping_class = {}
	for x in range(len(results)):
		mapping_class[classification_labels[x]] = results[x]
	return (mapping_class, mapping_cont)


def predict(mapping_class: dict[str, SVC], mapping_cont: dict[str, np.ndarray]) -> pd.DataFrame:
	res = pd.DataFrame()
	# df_train = pd.read_csv('./preprocessed_avg_train_0_avg.csv', delimiter=',')
	df_test_class = pd.read_csv('./preprocessed_test_0.csv', delimiter=',') 
	df_test_cont = pd.read_csv('./preprocessed_test_median.csv', delimiter=',')
	res['pid'] = df_test_class['pid']
	# df_train.set_index('pid', inplace=True)
	df_test_class.set_index('pid', inplace=True)
	df_test_cont.set_index('pid', inplace=True)

	# sc = StandardScaler()
	# sc.fit(df_train.values)
	# x_test = sc.transform(df_test.values)
	x_test = df_test_class.values

	for label in classification_labels:
		# res[label] = [x[1] for x in mapping_class[label].predict_proba(sc.transform(df.values))]
		res[label] = [x[1] for x in mapping_class[label].predict_proba(x_test)]
	for label in continuous_labels:
		res[label] = np.matmul(df_test_cont.values, mapping_cont[label])
	return res


def init() -> None:
	preprocess_train()
	preprocess_test()


if __name__ == '__main__':
	# init()
	mapping_class, mapping_cont = parallelize_train()
	res = predict(mapping_class, mapping_cont)
	res.to_csv('out.zip', index=False, float_format='%.3f', compression='zip')
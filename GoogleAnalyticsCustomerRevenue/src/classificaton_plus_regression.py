#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

#Parameters
scaleData = False
usePCA = None
useKPCA = None


print('Loading train/validate set.. ', end='', flush=True)
data = pd.read_csv('../data/trainAggregate.csv', dtype={'fullVisitorId': object})
data = data.drop(columns=['fullVisitorId'])
mask = np.random.rand(len(data)) < 0.7

#Extract 70% train set
train = data[mask]
y_revenue_train = train['totalTransactionRevenue']
y_class_train = train['nonZeroRevenue']
x_train = train.drop(columns=['totalTransactionRevenue', 'nonZeroRevenue'])
if scaleData:
	x_train = scale(x_train)

#Extract 30% validation set
val = data[~mask]
y_revenue_val = val['totalTransactionRevenue']	
y_class_val = val['nonZeroRevenue']
x_val = val.drop(columns=['totalTransactionRevenue', 'nonZeroRevenue'])
if scaleData:
	x_val = scale(x_val)
print('Done.')


print('Balancing dataset.. ', end='\n', flush=True)
train_pos = train[train['nonZeroRevenue'] == 1]
train_neg = train[train['nonZeroRevenue'] == 0]

train_pos = train_pos.sample(frac=11, replace=True)
mask = np.random.rand(len(train_neg)) < 0.24
train_neg = train_neg[mask]
print('Positive Samples: ' + str(len(train_pos)))
print('Negative Samples: ' + str(len(train_neg)))
class_train = shuffle(train_pos.append(train_neg))

y_class_train = class_train['nonZeroRevenue']
x_class_train = class_train.drop(columns=['totalTransactionRevenue', 'nonZeroRevenue'])
print('Done.')


print('Training classifier.. ', end='\n', flush=True)
classifier = GradientBoostingClassifier(
	learning_rate=0.1,
	n_estimators=60,
	max_features="auto",
	max_depth=40,
	min_samples_leaf=100,
	max_leaf_nodes=20,
	subsample=0.6,
	verbose=True)
classifier.fit(x_class_train, y_class_train)
print('Done.')


print('Performing classifier validation.. ', end='', flush=True)
train['class'] = classifier.predict(x_train)
val['class'] = classifier.predict(x_val)
print('Done.')
print('Classifier Scores: ')
confusion = confusion_matrix(y_class_val, val['class'])
print(confusion)
class1Accuracy = 1.0*(confusion[0,0] / (confusion[0,0] + confusion[0,1]))
class2Accuracy = 1.0*(confusion[1,1] / (confusion[1,0] + confusion[1,1]))
print("Class 1 Accuracy: " + str(class1Accuracy))
print("Class 2 Accuracy: " + str(class2Accuracy))


if class1Accuracy < 0.94 or class2Accuracy < 0.96:
	print('Classifier not strong enough, not proceeding.')
	sys.stdout.write('\a')
	sys.stdout.flush()
	sys.exit()

#Split dataset into predicted nonZeroRevenue and zeroRevenue
non_zero_revenue_train = train[train['class'] == 1]
non_zero_revenue_val = val[val['class'] == 1]
zero_revenue_train = train[train['class'] == 0]
zero_revenue_val = val[val['class'] == 0]
y_non_zero_revenue_train = non_zero_revenue_train['totalTransactionRevenue']
y_non_zero_revenue_val = non_zero_revenue_val['totalTransactionRevenue']
y_zero_revenue_train = zero_revenue_train['totalTransactionRevenue']
y_zero_revenue_val = zero_revenue_val['totalTransactionRevenue']
x_non_zero_revenue_train = non_zero_revenue_train.drop(columns=['class', 'totalTransactionRevenue', 'nonZeroRevenue'])
x_non_zero_revenue_val = non_zero_revenue_val.drop(columns=['class', 'totalTransactionRevenue', 'nonZeroRevenue'])
x_zero_revenue_train = zero_revenue_train.drop(columns=['class', 'totalTransactionRevenue', 'nonZeroRevenue'])
x_zero_revenue_val = zero_revenue_val.drop(columns=['class', 'totalTransactionRevenue', 'nonZeroRevenue'])

if scaleData:
	x_non_zero_revenue_train = scale(x_non_zero_revenue_train)
	x_non_zero_revenue_val = scale(x_non_zero_revenue_val)
	x_zero_revenue_train = scale(x_zero_revenue_train)
	x_zero_revenue_val = scale(x_zero_revenue_val)


print('Training non-zero regressor.. ', end='\n', flush=True)
non_zero_regressor = GradientBoostingRegressor(
	learning_rate=0.05,
	n_estimators=45,
	max_features="auto",
	max_depth=20,
	min_samples_leaf=30,
	max_leaf_nodes=40,
	subsample=0.8,
	verbose=True)
non_zero_regressor.fit(x_non_zero_revenue_train, y_non_zero_revenue_train)
print('Done.')
print('Training zero regressor.. ', end='\n', flush=True)
zero_regressor = GradientBoostingRegressor(
	learning_rate=0.05,
	n_estimators=5,
	max_features="auto",
	max_depth=20,
	min_samples_leaf=500,
	max_leaf_nodes=40,
	subsample=0.8,
	verbose=True)
zero_regressor.fit(x_zero_revenue_train, y_zero_revenue_train)
print('Done.')


print('Performing regressor validation.. ', end='', flush=True)
non_zero_predictions = non_zero_regressor.predict(x_non_zero_revenue_val)
non_zero_predictions[np.where(non_zero_predictions < 0)] = 0
non_zero_MSE = (((y_non_zero_revenue_val - non_zero_predictions)**2).sum()/y_non_zero_revenue_val.shape[0])
zero_predictions = zero_regressor.predict(x_zero_revenue_val)
zero_predictions[np.where(zero_predictions < 0)] = 0
zero_MSE = (((y_zero_revenue_val - zero_predictions)**2).sum()/y_zero_revenue_val.shape[0])
predictions = np.concatenate((non_zero_predictions, zero_predictions))
MSE = (((val['totalTransactionRevenue'] - predictions)**2).sum()/val.shape[0])
print('Done.')
print('Non-zero validation RMSE: ' + str(np.sqrt(non_zero_MSE)))
print('Zero validation RMSE: ' + str(np.sqrt(zero_MSE)))
print('RMSE: ' + str(np.sqrt(MSE))) 


if np.sqrt(MSE) <= 1.69:
	print('Loading test set.. ', end='', flush=True)
	test = pd.read_csv('../data/testAggregate.csv', dtype={'fullVisitorId': object})
	fullVisitorIds = test['fullVisitorId']
	x_test = test.drop(columns=['fullVisitorId', 'totalTransactionRevenue', 'nonZeroRevenue'])
	if scaleData:
		x_test = scale(x_test)
	print('Done.')

	# Splitting based on classificatiom
	test['class'] = classifier.predict(x_test)
	non_zero_revenue_test = test[test['class'] == 1]
	zero_revenue_test = test[test['class'] == 0]
	y_non_zero_revenue_test = non_zero_revenue_test['totalTransactionRevenue']
	y_zero_revenue_test = zero_revenue_test['totalTransactionRevenue']
	x_non_zero_revenue_test = non_zero_revenue_test.drop(columns=['fullVisitorId', 'class', 'totalTransactionRevenue', 'nonZeroRevenue'])
	x_zero_revenue_test = zero_revenue_test.drop(columns=['fullVisitorId', 'class', 'totalTransactionRevenue', 'nonZeroRevenue'])

	print('Saving results.. ', end='', flush=True)
	non_zero_predictions = non_zero_regressor.predict(x_non_zero_revenue_test)
	zero_predictions = zero_regressor.predict(x_zero_revenue_test)
	predictions = np.concatenate((non_zero_predictions, zero_predictions))
	predictions[np.where(predictions < 0)] = 0
	results = pd.concat([fullVisitorIds, pd.Series(predictions)], axis=1, keys=['fullVisitorId', 'PredictedLogRevenue'])
	results.to_csv('../predictions/results.csv', index=False)
	print('Done.')
else:
	print('Regressor not strong enough, not proceeding.')




# Sound the "done" bell!
sys.stdout.write('\a')
sys.stdout.flush()



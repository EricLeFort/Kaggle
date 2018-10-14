#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

#Parameters
scale_data = False
do_grid_search = True
use_PCA = None
use_KPCA = None


def perform_grid_search(estimator, param_grid, x_train, y_train, x_val, y_val):
	grid = GridSearchCV(
			estimator,
			param_grid,
			cv=5,
			scoring='r2',
			error_score=np.nan,
			n_jobs=-1
	)
	grid.fit(x_train, y_train)

	print("Best parameters set found on development set:", end="\n\n")
	print(grid.best_params_, end="\n\n")
	print("Grid scores on development set:", end="\n\n")
	means = grid.cv_results_['mean_test_score']
	stds = grid.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, grid.cv_results_['params']):
		print("%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params))

	return grid


print('Loading train/validate set.. ', end='', flush=True)
data = pd.read_csv('../data/trainAggregate.csv', dtype={'fullVisitorId': object})
data = data.drop(columns=['fullVisitorId'])
mask = np.random.rand(len(data)) < 0.7

#Extract 70% train set
train = data[mask]
y_train = train['totalTransactionRevenue']
x_train = train.drop(columns=['totalTransactionRevenue', 'nonZeroRevenue'])
if scale_data:
	x_train = scale(x_train)

#Extract 30% validation set
val = data[~mask]
y_val = val['totalTransactionRevenue']
x_val = val.drop(columns=['totalTransactionRevenue', 'nonZeroRevenue'])
if scale_data:
	x_val = scale(x_val)
print('Done.')

print('Training regressor.. ', end='\n', flush=True)
regressor = XGBRegressor(
	max_depth=10,
	max_leaves=10,
	learning_rate=0.2,
	n_estimators=40,
	booster='gbtree',
	n_jobs=8,
	min_split_loss=0.05,
	subsample=0.4,
	colsample_bytree=0.7,
	colsample_bylevel=0.4,
	reg_alpha=0,
	reg_lambda=1)
regressor = XGBRegressor(
	booster='gbtree',
	n_jobs=8,
	verbose=3,
	max_depth=15,
	n_estimators=50,
	learning_rate=0.05,
	colsample_bytree=0.4,
	colsample_bylevel=0.4,
	subsample=0.6)
if not do_grid_search:
	regressor.fit(x_train, y_train)
print('Done.')

if do_grid_search:
	param_grid = [
		{
			'max_depth': [5, 6, 7],					# 5 - 7
			'learning_rate': [0.07, 0.09, 0.11, 0.13, 0.15],	# 0.07 - 0.15
			#'n_estimators': [40, 50, 60],			# > 30
			#'subsample': [0.5, 0.75, 1],			# 0.20 - 1.00
			#'colsample_bytree': [0.75, 0.9, 1],	# 0.50 - 1.00
			#'colsample_bylevel': [0.4, 0.6, 1]		# 0.40 - 1.00
			'gamma': [0, 0.01, 0.05, 0.1, 0.5],
			#'reg_alpha': [0, 0.05, 0.5, 1],		# No significant dependence noted
			#'reg_lambda': [5, 25, 50]				# > 5
		}
	]
	regressor = XGBRegressor(booster='gbtree', n_jobs=8, verbose=3, n_estimators=50, reg_lambda=25)
	regressor = perform_grid_search(regressor, param_grid, x_train, y_train, x_val, y_val)


print('Performing regressor validation.. ', end='', flush=True)
if scale_data:
	x_val = scale(x_val)
predictions = regressor.predict(x_val)
predictions[np.where(predictions < 0)] = 0
MSE = (((y_val - predictions)**2).sum()/y_val.shape[0])
print('Done.')
print('Validation RMSE: ' + str(np.sqrt(MSE)))


if np.sqrt(MSE) <= 1.7:
	print('Loading test set.. ', end='', flush=True)
	data = pd.read_csv('../data/testAggregate.csv', dtype={'fullVisitorId': object})
	fullVisitorIds = data['fullVisitorId']
	x_test = data.drop(columns=['fullVisitorId', 'totalTransactionRevenue', 'nonZeroRevenue'])
	if scale_data:
		x_test = scale(x_test)
	print('Done.')

	print('Saving results.. ', end='', flush=True)
	predictions = regressor.predict(x_test)
	predictions[np.where(predictions < 0)] = 0
	results = pd.concat([fullVisitorIds, pd.Series(predictions)], axis=1, keys=['fullVisitorId', 'PredictedLogRevenue'])
	results.to_csv('../predictions/results.csv', index=False)
	print('Done.')
else:
	print('Regressor not strong enough, not proceeding.')


# Sound the "done" bell!
sys.stdout.write('\a')
sys.stdout.flush()



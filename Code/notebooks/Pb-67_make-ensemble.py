#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
import time
import os

# ## Get same X_train and X_test as in model-selection file 

with open('../../Data/WOD_Pb_dataset-cleanedSOPbconc.csv', mode='r', newline='') as csvfile:
    WOD_pb_dataset = pd.read_csv(csvfile)

pb_67_dataset = WOD_pb_dataset.dropna(subset = ['Pb_206_207'])
pb_67_dataset_ref = pb_67_dataset.copy()
pb_67_dataset.drop(['Ocean_basin'], axis=1, inplace=True)

idx_1 = pb_67_dataset[(pb_67_dataset['Cruise'] == 'GA02') & (pb_67_dataset['WOD_latitude [degrees_north]'] < 0) & (pb_67_dataset['WOD_latitude [degrees_north]'] > -27)].index
test_geo_NA = pb_67_dataset.loc[idx_1]
dataset = pb_67_dataset.drop(idx_1)

y_test_geo_NA = test_geo_NA['Pb_206_207']
X_test_geo_NA = test_geo_NA.drop(['Pb_CONC [pmol/kg]', 'Pb_206_207', 'Pb_208_207', 'Cruise', 'WOD_latitude [degrees_north]', 'WOD_longitude [degrees_east]'], axis=1)

y = dataset['Pb_206_207']
X = dataset.drop(['Pb_CONC [pmol/kg]', 'Pb_206_207', 'Pb_208_207', 'Cruise', 'WOD_latitude [degrees_north]', 'WOD_longitude [degrees_east]'], axis=1)

# Get the best model from the grid search
model_grid = pd.read_csv('Model_output/XGBoost-GridsearchCV_67pb_results_20240924-111640_with-test-performance.csv')
model_grid['combined_test_rmse'] = model_grid['rmse_test'] + model_grid['rmse_test_geo_NA']
model_grid.sort_values(by='rank_test_score', inplace=True)

#Open prediction dataset
with open('../../Data/Global-prediction_masked-df_no-coords.csv', mode='r', newline='') as csvfile:
    prediction_df_latlon = pd.read_csv(csvfile)
prediction_df = prediction_df_latlon.drop(['WOD_latitude [degrees_north]', 'WOD_longitude [degrees_east]', 'Ocean_basin'], axis=1)
prediction_df.rename(columns={'Depth': 'WOD_depth'}, inplace=True)

# make model ensemble predictions by changing random_state
rmse = []
r2 = []
mape = []
rmse_geo_NA = []
r2_geo_NA = []
mape_geo_NA = []

for i in range(0,100):
    print(f'Iteration {i}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=i)

    best_xgb_model = XGBRegressor(colsample_bytree=model_grid['param_colsample_bytree'].to_numpy()[0], 
                              learning_rate=model_grid['param_learning_rate'].to_numpy()[0], 
                              max_depth=model_grid['param_max_depth'].to_numpy()[0], 
                              min_child_weight=model_grid['param_min_child_weight'].to_numpy()[0], 
                              n_estimators=model_grid['param_n_estimators'].to_numpy()[0], 
                              reg_alpha=model_grid['param_reg_alpha'].to_numpy()[0],
                              reg_lambda=model_grid['param_reg_lambda'].to_numpy()[0],
                              early_stopping_rounds=50,
                              eval_metric = 'rmse',
                              seed=42)
    
    best_xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

    y_pred = best_xgb_model.predict(X_test, iteration_range=(0, best_xgb_model.best_iteration + 1))
    y_pred_geo_NA = best_xgb_model.predict(X_test_geo_NA, iteration_range=(0, best_xgb_model.best_iteration + 1))

    rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2.append(r2_score(y_test, y_pred))
    mape.append(mean_absolute_percentage_error(y_test, y_pred))
    rmse_geo_NA.append(np.sqrt(mean_squared_error(y_test_geo_NA, y_pred_geo_NA)))
    r2_geo_NA.append(r2_score(y_test_geo_NA, y_pred_geo_NA))
    mape_geo_NA.append(mean_absolute_percentage_error(y_test_geo_NA, y_pred_geo_NA))

    global_preds = best_xgb_model.predict(prediction_df)
    prediction_df_latlon[f'Pb_206_207_rs{i}'] = global_preds

# Save the predictions
timestr = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(f'Model_output/{timestr}_Pb67')
prediction_df_latlon.to_csv(f'Model_output/{timestr}_Pb67/XGBoost-67pb_ensemble_predictions.csv', index=False)

# Save the performance metrics
performance_metrics = pd.DataFrame({'rmse': rmse, 'r2': r2, 'mape': mape, 'rmse_geo_NA': rmse_geo_NA, 'r2_geo_NA': r2_geo_NA, 'mape_geo_NA': mape_geo_NA})
performance_metrics.to_csv(f'Model_output/{timestr}_Pb67/XGBoost-67pb_ensemble_performance.csv', index=False)

#Save copy of the script
os.system(f'cp Pb-67_make-ensemble.py Model_output/{timestr}_Pb67/Pb-67_make-ensemble.py')















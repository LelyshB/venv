# Задача - перебить эти параметры
# LightGBM - Средний RMSE: 0.4755, Средний MAE: 0.1960
# XGBoost - Средний RMSE: 0.5029, Средний MAE: 0.2207
# CatBoost - Средний RMSE: 0.4775, Средний MAE: 0.2021


# выбираем LightGBM (как лучшую модель)
# Показатели самой дефолтной модели
# Средний RMSE: 0.4755, Средний MAE: 0.1960


# Показатели Optuna для лучшей модели
# Наилучшие параметры: {'boosting_type': 'gbdt', 'num_leaves': 298, 
#                       'max_depth': 14, 'learning_rate': 0.013556172600367055, 
#                       'n_estimators': 966, 'min_child_samples': 16, 'subsample': 0.5589245959275299, 
#                       'subsample_freq': 1, 'colsample_bytree': 0.6938565563901126, 'reg_alpha': 0.020409788690954757, 
#                       'reg_lambda': 9.271074049708654e-07, 'min_split_gain': 0.18934219448386327, 'max_bin': 449}


import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np
import os

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_data(filepath):
    return pd.read_csv(filepath)

def objective(trial, X_train, y_train):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'max_bin': trial.suggest_int('max_bin', 100, 500),
        'verbose': -1

    }

    tscv = TimeSeriesSplit(n_splits=3)
    rmse_list = []
    
    for train_index, val_index in tscv.split(X_train):
        X_train_split, X_val_split = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_split, y_val_split = y_train.iloc[train_index], y_train.iloc[val_index]

        model = lgb.LGBMRegressor(**param)
        model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        # Предсказание на валидационной выборке
        y_val_pred = model.predict(X_val_split)
        rmse = calculate_rmse(y_val_split, y_val_pred)
        rmse_list.append(rmse)

    # Возвращаем среднее значение RMSE по всем разбиениям
    return np.mean(rmse_list)


if __name__ == "__main__":
    filepath = os.path.join('..', 'data', 'processed', 'final_dataset.csv')
    
    data = load_data(filepath)
    
    X = data.drop(columns=['item_cnt_day'])
    y = data['item_cnt_day']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=100)

    print("Наилучшие параметры:", study.best_params)

    # Далее мы обучим модель на лучших параметрах и посмотрим её показатели на тестовой выборке
    best_params = study.best_params
    best_model = lgb.LGBMRegressor(**best_params)
    best_model.fit(X_train, y_train)

    y_test_pred = best_model.predict(X_test)
    rmse_test = calculate_rmse(y_test, y_test_pred)
    print(f"RMSE на тестовой выборке: {rmse_test}")

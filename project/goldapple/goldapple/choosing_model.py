import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
import os



def load_data(filepath):
    return pd.read_csv(filepath)

def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rmse, mae

def plot_errors(y_true, y_pred, model_name):
    errors = y_true - y_pred
    plt.figure(figsize=(10, 5))
    sns.histplot(errors, kde=True, bins=30)
    plt.title(f"{model_name} - Распределение ошибок")
    plt.xlabel("Ошибка")
    plt.ylabel("Частота")
    plt.grid(True)
    plt.show()

def test_models_with_timeseries_split(X, y):
    results = {}

    # для временных рядов
    tscv = TimeSeriesSplit(n_splits=3)
    
    models = {
        "LightGBM": lgb.LGBMRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(random_state=42, eval_metric='rmse'),
        "CatBoost": cb.CatBoostRegressor(verbose=0, random_state=42)
    }

    for model_name, model in models.items():
        rmse_scores, mae_scores = [], []

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)

            rmse, mae = evaluate_model(y_val, y_val_pred, model_name)
            rmse_scores.append(rmse)
            mae_scores.append(mae)

        # Средние значения метрик по кросс-валидации
        results[model_name] = {
            'RMSE': np.mean(rmse_scores),
            'MAE': np.mean(mae_scores)
        }

    return results

def evaluate_on_test_set(model, X_test, y_test, model_name):
    y_test_pred = model.predict(X_test)
    rmse, mae = evaluate_model(y_test, y_test_pred, f"{model_name} (Test Set)")
    plot_errors(y_test, y_test_pred, f"{model_name} (Test Set)")
    return rmse, mae

if __name__ == "__main__":
    filepath = os.path.join('..', 'data', 'processed', 'final_dataset.csv')
    data = load_data(filepath)

    test_month = data['date_block_num'].max()
    train_data = data[data['date_block_num'] < test_month - 1]
    val_data = data[data['date_block_num'] == test_month - 1]
    test_data = data[data['date_block_num'] == test_month]

    X_train = train_data.drop(columns=['item_cnt_day'])
    y_train = train_data['item_cnt_day']
    X_val = val_data.drop(columns=['item_cnt_day'])
    y_val = val_data['item_cnt_day']
    X_test = test_data.drop(columns=['item_cnt_day'])
    y_test = test_data['item_cnt_day']

    results = test_models_with_timeseries_split(X_train, y_train)

    for model_name, metrics in results.items():
        print(f"{model_name} - Средний RMSE: {metrics['RMSE']:.4f}, Средний MAE: {metrics['MAE']:.4f}")

    best_model_name = min(results, key=lambda k: results[k]['RMSE'])
    print(f"Лучшая модель по RMSE: {best_model_name}")

    best_model = {
        "LightGBM": lgb.LGBMRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(random_state=42, eval_metric='rmse'),
        "CatBoost": cb.CatBoostRegressor(verbose=0, random_state=42)
    }[best_model_name]
    
    best_model.fit(X_train, y_train)
    evaluate_on_test_set(best_model, X_test, y_test, best_model_name)

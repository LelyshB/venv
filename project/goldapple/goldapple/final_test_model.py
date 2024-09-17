# Здесь мы обучаем модель на лучших параметрах от Optuna 
# и делаем из датасета тестовую выборку именно такой, какую будем предсказывать
# т.е. мы убираем из теста все фичи (как в test_submission), восстанавливаем их и предсказываем

# цель этого файла точно проверить, как модель описывает наши данные в реальной ситуации

import pandas as pd
import os
from sklearn.metrics import mean_squared_error
import numpy as np
import lightgbm as lgb
from feature_engineering import create_lag_features, create_rolling_features, add_shop_brand_features, plot_errors, plot_feature_importance, calculate_rmse

def load_data(filepath):
    return pd.read_csv(filepath)

def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rmse, mae

def create_date_features(df):
    df = df.copy()

    if 'date' in df.columns:
        # Если колонка 'date' существует, создаем временные признаки
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
    elif 'date_block_num' in df.columns:
        # Если колонки 'date' нет, используем 'date_block_num'
        df['month'] = (df['date_block_num'] % 12) + 1
        df['year'] = (df['date_block_num'] // 12) + 2018

    return df

if __name__ == "__main__":
    filepath = os.path.join('..', 'data', 'processed', 'final_dataset.csv')

    data = load_data(filepath)

    # Применяем временные признаки
    data = create_date_features(data)

    if 'date' in data.columns:
        data = data.drop(columns=['date'])

    train_data = data[data['date_block_num'] < 25]
    test_data = data[data['date_block_num'] == 25]

    # Обработка фичей на тренировочных данных
    train_data = create_lag_features(train_data, lags=[1, 2])
    train_data = create_rolling_features(train_data, window_sizes=[3])
    train_data, shop_items, brand_items, shop_sales, brand_sales, brand_avg_price, brand_avg_discount = add_shop_brand_features(train_data)

    X_train = train_data.drop(columns=['item_cnt_day'])
    y_train = train_data['item_cnt_day']

    # Применение фичей на тестовом датасете
    test_data = create_lag_features(test_data, lags=[1, 2])
    test_data = create_rolling_features(test_data, window_sizes=[3])
    test_data = test_data.merge(shop_items, on='shop_id', how='left').merge(brand_items, on='brand_id_x', how='left')
    test_data = test_data.merge(shop_sales, on='shop_id', how='left').merge(brand_sales, on='brand_id_x', how='left')
    test_data = test_data.merge(brand_avg_price, on='brand_id_x', how='left').merge(brand_avg_discount, on='brand_id_x', how='left')

    X_test = test_data.drop(columns=['item_cnt_day'])

    params = {'boosting_type': 'gbdt', 'num_leaves': 298, 
            'max_depth': 14, 'learning_rate': 0.013556172600367055, 
            'n_estimators': 966, 'min_child_samples': 16, 
            'subsample': 0.5589245959275299, 'subsample_freq': 1,
            'colsample_bytree': 0.6938565563901126, 'reg_alpha': 0.020409788690954757, 
            'reg_lambda': 9.271074049708654e-07, 'min_split_gain': 0.18934219448386327, 
            'max_bin': 449}

    model = lgb.LGBMRegressor(**params, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse, mae = evaluate_model(test_data['item_cnt_day'], y_pred, "LightGBM")

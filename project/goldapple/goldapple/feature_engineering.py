import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Вычисление RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Функция для создания временных признаков
def create_date_features(df):
    df = df.copy()
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.year
    return df

# Функция для создания лагов
def create_lag_features(df, lags):
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}_months'] = df.groupby('item_id')['item_cnt_day'].shift(lag)
    return df

# Функция для создания скользящих средних
def create_rolling_features(df, window_sizes):
    df = df.copy()
    for window in window_sizes:
        df[f'rolling_mean_{window}_months'] = df.groupby('item_id')['item_cnt_day'].shift(1).rolling(window=window).mean()
    return df

# Функция для удаления выбросов
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# Функция для добавления признаков для магазинов и брендов
def add_shop_brand_features(train):
    # Признаки для магазинов
    shop_items = train.groupby('shop_id')['item_id'].nunique().reset_index().rename(columns={'item_id': 'unique_items_per_shop'})
    shop_sales = train.groupby('shop_id')['item_cnt_day'].sum().reset_index().rename(columns={'item_cnt_day': 'total_sales_per_shop'})
    
    train = train.merge(shop_items, on='shop_id', how='left')
    train = train.merge(shop_sales, on='shop_id', how='left')

    # Признаки для брендов
    brand_items = train.groupby('brand_id_x')['item_id'].nunique().reset_index().rename(columns={'item_id': 'unique_items_per_brand'})
    brand_sales = train.groupby('brand_id_x')['item_cnt_day'].sum().reset_index().rename(columns={'item_cnt_day': 'total_sales_per_brand'})
    brand_avg_price = train.groupby('brand_id_x')['item_price'].mean().reset_index().rename(columns={'item_price': 'avg_price_per_brand'})
    brand_avg_discount = train.groupby('brand_id_x')['discamount'].mean().reset_index().rename(columns={'discamount': 'avg_discount_per_brand'})
    
    train = train.merge(brand_items, on='brand_id_x', how='left')
    train = train.merge(brand_sales, on='brand_id_x', how='left')
    train = train.merge(brand_avg_price, on='brand_id_x', how='left')
    train = train.merge(brand_avg_discount, on='brand_id_x', how='left')

    return train, shop_items, brand_items, shop_sales, brand_sales, brand_avg_price, brand_avg_discount

# Функция для визуализации ошибок
def plot_errors(y_test, y_test_pred, errors):
    # Создаем фигуру с двумя графиками
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(errors, bins=30, edgecolor='black')
    axs[0].set_title('Распределение ошибок модели')
    axs[0].set_xlabel('Ошибка')
    axs[0].set_ylabel('Частота')
    axs[0].grid(True)

    axs[1].scatter(y_test_pred, errors)
    axs[1].axhline(y=0, color='r', linestyle='--')
    axs[1].set_title('График остатков (Residuals)')
    axs[1].set_xlabel('Предсказанные значения')
    axs[1].set_ylabel('Ошибки')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# Важность признаков
def plot_feature_importance(model):
    lgb.plot_importance(model, importance_type="gain", figsize=(7, 6), title="LightGBM Feature Importance (Gain)")
    plt.show()






# Основной запуск
if __name__ == "__main__":
    filepath = os.path.join('..', 'data', 'interim', 'cleared_dataset.csv')

    # Загружаем и обрабатываем данные
    train = pd.read_csv(filepath)
    train['date'] = pd.to_datetime(train['date'], errors='coerce')

    # Добавляем признаки даты
    train = create_date_features(train)

    # Убираем колонку 'date'
    train = train.drop(columns=['date']).copy()

    # Создаем лаги и скользящие средние
    lags = [1, 2]
    window_sizes = [3]
    train = create_lag_features(train, lags)
    train = create_rolling_features(train, window_sizes)

    # Заполняем пропуски
    lags_columns = [f'lag_{lag}_months' for lag in lags]
    train[lags_columns] = train[lags_columns].fillna(0)
    rolling_means_columns = [f'rolling_mean_{window}_months' for window in window_sizes]
    train[rolling_means_columns] = train[rolling_means_columns].ffill()
    train['rolling_mean_3_months'] = train['rolling_mean_3_months'].fillna(0)
    train['days with 0 balance'] = train['days with 0 balance'].fillna(0)

    # Удаляем выбросы
    train = remove_outliers(train, 'item_price')
    train = remove_outliers(train, 'discamount')
    train = remove_outliers(train, 'promo_time')

    print(len(train))

    # Логарифмируем признаки
    train['log_item_price'] = np.log1p(train['item_price'].clip(lower=1e-6))

    # Убираем ненужные колонки
    train = train.drop(columns=['promo_time', 'year', 'week'])

    # Разделение данных на train/val/test
    train_data = train[train['date_block_num'] < 24]
    val_data = train[train['date_block_num'] == 24]
    test_data = train[train['date_block_num'] == 25]

    X_train = train_data.drop(columns=['item_cnt_day'])
    y_train = train_data['item_cnt_day']
    X_val = val_data.drop(columns=['item_cnt_day'])
    y_val = val_data['item_cnt_day']
    X_test = test_data.drop(columns=['item_cnt_day'])
    y_test = test_data['item_cnt_day']

    # Добавляем признаки магазинов и брендов
    train, shop_items, brand_items, shop_sales, brand_sales, brand_avg_price, brand_avg_discount = add_shop_brand_features(train)

    # Применяем те же преобразования к валидационным и тестовым данным
    X_train = X_train.merge(shop_items, on='shop_id', how='left').merge(brand_items, on='brand_id_x', how='left')
    X_val = X_val.merge(shop_items, on='shop_id', how='left').merge(brand_items, on='brand_id_x', how='left')
    X_test = X_test.merge(shop_items, on='shop_id', how='left').merge(brand_items, on='brand_id_x', how='left')

    X_train = X_train.merge(shop_sales, on='shop_id', how='left').merge(brand_sales, on='brand_id_x', how='left')
    X_val = X_val.merge(shop_sales, on='shop_id', how='left').merge(brand_sales, on='brand_id_x', how='left')
    X_test = X_test.merge(shop_sales, on='shop_id', how='left').merge(brand_sales, on='brand_id_x', how='left')

    X_train = X_train.merge(brand_avg_price, on='brand_id_x', how='left').merge(brand_avg_discount, on='brand_id_x', how='left')
    X_val = X_val.merge(brand_avg_price, on='brand_id_x', how='left').merge(brand_avg_discount, on='brand_id_x', how='left')
    X_test = X_test.merge(brand_avg_price, on='brand_id_x', how='left').merge(brand_avg_discount, on='brand_id_x', how='left')

    # Обучение модели на тренировочной выборке
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Предсказание на валидационной выборке
    y_val_pred = model.predict(X_val)

    # Оценка модели на валидационных данных
    rmse_val = calculate_rmse(y_val, y_val_pred)
    print(f'RMSE на валидационной выборке: {rmse_val}')

    # Предсказание на тестовой выборке
    y_test_pred = model.predict(X_test)

    # Оценка модели на тестовых данных
    rmse_test = calculate_rmse(y_test, y_test_pred)
    print(f'RMSE на тестовой выборке: {rmse_test}')

    # Визуализация ошибок
    errors = y_test - y_test_pred
    plot_errors(y_test, y_test_pred, errors)

    # Важность признаков
    plot_feature_importance(model)

    # Кросс-валидация
    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores_train = []
    rmse_scores_val = []

    for train_index, val_index in tscv.split(X_train):
        X_train_split, X_val_split = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_split, y_val_split = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(X_train_split, y_train_split)

        # Прогноз на тренировочной выборке
        y_train_pred = model.predict(X_train_split)
        rmse_train = calculate_rmse(y_train_split, y_train_pred)
        rmse_scores_train.append(rmse_train)

        # Прогноз на валидационной выборке
        y_val_pred = model.predict(X_val_split)
        rmse_val = calculate_rmse(y_val_split, y_val_pred)
        rmse_scores_val.append(rmse_val)

    # Средние RMSE по кросс-валидации
    mean_rmse_train = np.mean(rmse_scores_train)
    mean_rmse_val = np.mean(rmse_scores_val)
    print(f"Среднее RMSE на тренировочной выборке: {mean_rmse_train}")
    print(f"Среднее RMSE на валидационной выборке: {mean_rmse_val}")

    # Оценка переобучения
    if mean_rmse_train - mean_rmse_val < 0.05:
        print("Модель стабильна и не переобучена.")
    else:
        print("Переобучение")


    # Сохраняем финальный датасет
    final_dataset = pd.concat([train_data, val_data, test_data])
    final_dataset.to_csv('../data/processed/final_dataset.csv', index=False)
       
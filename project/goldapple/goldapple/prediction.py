# Я не знаю как это сделать, тк у нас просто нет никаких данных в test_submission
# и я не знаю как на нем предсказать
# если у нас банально на train

# month,date_block_num,shop_id,item_id,brand_id_x,item_cnt_day,item_price,discamount,promo ...

# а на test
# ID;date_block_num;shop_id;item_id;item_cnt_month
# То есть данных не хватает. Возможно я не понял сути. У себя на работе с таким не сталкивался

# Но я искренне постарался всё оформить как надо


import pandas as pd
import lightgbm as lgb
import os

train_file = os.path.join('..', 'data', 'processed', 'final_dataset.csv')
test_file = os.path.join('..', 'data', 'test', 'test_submission.csv')
items_file = os.path.join('..', 'data', 'raw', 'items.csv')
output_file = os.path.join('..', 'data', 'test', 'result.csv')

test_submission = pd.read_csv(test_file, delimiter=';')

print("Первые строки test_submission:")
print(test_submission.head())

# Добавляем недостающие признаки
# Эти признаки можно заменить более точными значениями, если есть данные.
test_submission['month'] = 3  # Март 2020
test_submission['brand_id_x'] = 0  # Используем 0, если данных нет
test_submission['item_price'] = 1  # Предположительное значение
test_submission['discamount'] = 0
test_submission['promo'] = 0
test_submission['size_disc'] = 0
test_submission['spec_promo'] = 0
test_submission['days with 0 balance'] = 0
test_submission['item_category'] = 0
test_submission['day'] = 15  # Средний день месяца
test_submission['day_of_week'] = 3  # Среда
test_submission['week_of_year'] = 12
test_submission['lag_1_months'] = 0
test_submission['lag_2_months'] = 0
test_submission['rolling_mean_3_months'] = 0
test_submission['log_item_price'] = 0

# Загрузка данных для обучения
train_data = pd.read_csv(train_file)
X_train = train_data.drop(columns=['item_cnt_day'])
y_train = train_data['item_cnt_day']

params = {
    'boosting_type': 'gbdt', 'num_leaves': 298, 'max_depth': 14, 
    'learning_rate': 0.0135, 'n_estimators': 966, 'min_child_samples': 16,
    'subsample': 0.5589, 'subsample_freq': 1, 'colsample_bytree': 0.6938,
    'reg_alpha': 0.02, 'reg_lambda': 9e-7, 'max_bin': 449
}

model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)

# Прогнозирование
test_submission['item_cnt_month'] = model.predict(test_submission.drop(columns=['item_cnt_month']))

# Ограничим прогнозы
test_submission['item_cnt_month'] = test_submission['item_cnt_month'].clip(0, 20)

test_submission[['ID', 'item_cnt_month']].to_csv(output_file, index=False)

print(f"Файл {output_file} успешно создан.")

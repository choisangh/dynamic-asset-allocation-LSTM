import pandas as pd
from src.data_preprocessing import data_preprocessing, divide_train_test_set
from src.config import FILE_NAME

print('Data load start')
df = pd.read_csv(f'./data/{FILE_NAME}')
df = df.set_index('date')
N_STOCKS = df.shape[1]
print('Data load completed')

print('Data Preprocessing start')
train_df, test_df = divide_train_test_set(df=df, test_ratio=0.7)
xc_train, xf_train, xc_test, xf_test, test_date = data_preprocessing(train_df, test_df)
print('Data Preprocessing completed')
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
import inspect
import logging
import torch

mylog = logging.getLogger(__name__)

def create_dataset(ticker:str='DB1.DE',
                   split:tuple[int]=(80, 15, 5),
                   seq_len:int=5
                   )->dict:
    if sum(split) != 100:
        mylog.error(f"Invalid split values (train, val, test) = {split}")
        raise ValueError('split must add up to 100')
    mylog.info(f"Called function: {inspect.stack()[0].function}")
    ticker = yf.Ticker(ticker)
    data_raw = ticker.history(period='max')
    data_raw = data_raw.dropna()
    mylog.info(f"Raw data shape after dropping NaN rows: {data_raw.shape}")
    stamps = data_raw.index
    mylog.info(f"Data covering {stamps[0]} to {stamps[-1]} with {len(stamps)} rows")
    # preprocessing data -------------------------------------------------------------
    # split vals for train and val --> test follows from them
    index_split = np.array([split[0], split[0]+split[1]])/100 * len(data_raw)

    # cutting down dataframes to five features wanted
    # columns stored in dataframe in  below order and from earliest [first entry]
    # to latest date [last entry]
    keys = ['Open', 'High', 'Low', 'Close', 'Volume']
    # scaling all entries to range [0,1]
    for key in keys:
        val = data_raw[key].values
        mylog.info(f"{key:6} in range [min, max] = [{val.min().__float__(), val.max().__float__()}]")
        data_raw[key] = MinMaxScaler(feature_range=(0,1)).fit_transform(data_raw[key].values.reshape(-1,1))
    targets = data_raw[keys[0]].values
    data_train = data_raw[keys][:int(index_split[0])].values
    data_vali  = data_raw[keys][int(index_split[0]):int(index_split[1])].values
    data_test  = data_raw[keys][int(index_split[1]):].values

    # creating sequence - target pairs, here sequence of all 5 features for seq_len days
    # concatenated after another. target is the open price of the succeeding day
    size_train = len(data_train)
    size_vali = len(data_vali)
    size_test = len(data_test)
    size_cols = data_train.shape[1] * seq_len
    x_train, y_train = np.zeros((size_train, size_cols), dtype=np.float64), np.zeros(size_train, dtype=np.float64)
    x_vali, y_vali   = np.zeros((size_vali, size_cols), dtype=np.float64), np.zeros(size_vali, dtype=np.float64)
    x_test, y_test   = np.zeros((size_test, size_cols), dtype=np.float64), np.zeros(size_test, dtype=np.float64)
    for i in range(len(data_train) - seq_len):
        x_train[i,:] = data_train[i:i+seq_len].flatten()
        y_train[i]   = targets[i+seq_len]
    for i in range(len(data_vali) - seq_len):
        x_vali[i,:] = data_vali[i:i+seq_len].flatten()
        y_vali[i]   = targets[i+seq_len]
    for i in range(len(data_test) - seq_len):
        x_test[i,:] = data_test[i:i+seq_len].flatten()
        y_test[i]   = targets[i+seq_len]

    # setting up self-explaining dict as return
    mydict = {'features-train': torch.from_numpy(x_train),
              'features-vali': torch.from_numpy(x_vali),
              'features-test': torch.from_numpy(x_test),
              'targets-train': torch.from_numpy(y_train),
              'targets-vali': torch.from_numpy(y_vali),
              'targets-test': torch.from_numpy(y_test),}

    return mydict
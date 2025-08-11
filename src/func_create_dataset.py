from email.encoders import encode_quopri
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
import inspect
import logging
import datetime

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
    targets = data_raw[keys[0]].values
    data_train = data_raw[keys][:int(index_split[0])].values
    data_vali = data_raw[keys][int(index_split[0]):int(index_split[1])].values
    data_test = data_raw[keys][int(index_split[1]):].values


    print(data_test)




    return 0
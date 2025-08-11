from src.func_create_dataset import create_dataset as CDS
import json
import pathlib as pl
import logging
import inspect
import datetime

#timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logging.basicConfig(level=logging.DEBUG,
                    filename=pl.Path.cwd().joinpath('logs', f"mainlog-{420}.log"),
                    encoding='utf-8',
                    format='%(asctime)s - PID %(process)d - %(name)s - %(levelname)s - %(message)s',
                    filemode='w')
mylog = logging.getLogger(__name__)

def main():
    mylog.info(f"Called function: {inspect.stack()[0].function}")
    # reading in configuration information from json file ----------------------------------
    path_config = pl.Path.cwd()/ 'input' / 'config.json'
    try:
        config = json.load(open(path_config))
        mylog.info(f"Succesfully loaded config from: {path_config}")
        mylog.info(f"The config file is: {config}")
    except FileNotFoundError:
        mylog.error(f"Config file not found: {path_config}")
        raise FileNotFoundError('config.json not found, check config path')

    # constructing pytorch dataset from tickers and other ----------------------------------
    # parameters passed to function
    mylog.info("Calling function for creating pytorch compatible dataset.")
    mydataset = CDS(ticker=config['ticker'],
                    split=config['pars-data']['data-split'],
                    seq_len=config['pars-data']['size-history'],)










    return 0

if __name__ == '__main__':
    main()
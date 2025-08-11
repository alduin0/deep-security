from src.func_create_dataset import create_dataset as CD
import json
import pathlib as pl
import logging
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logging.basicConfig(level=logging.DEBUG,
                    filename=pl.Path.cwd().joinpath('logs', f"mainlog-{timestamp}.log"),
                    encoding='utf-8',
                    format='%(asctime)s - PID %(process)d - %(name)s - %(levelname)s - %(message)s',)
mainlog = logging.getLogger(__name__)

def main():
    # reading in configuration information from json file ----------------------------------
    path_config = pl.Path.cwd()/ 'input' / 'config.json'
    try:
        config = json.load(open(path_config))
        mainlog.info(f"Succesfully loaded config from: {path_config}")
        mainlog.info(f"The config file is: {config}")
    except FileNotFoundError:
        mainlog.error(f"Config file not found: {path_config}")
        raise FileNotFoundError('config.json not found, check config path')












    return 0

if __name__ == '__main__':
    main()
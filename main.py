import numpy as np

from src.func_create_dataset import create_dataset as cds
from src.class_model_lstm import LSTMModel as lstm

import json
import pathlib as pl
import logging
import inspect
import torch
from tqdm import tqdm

#timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logging.basicConfig(level=logging.DEBUG,
                    filename=pl.Path.cwd().joinpath('logs', f"mainlog-{420}.log"),
                    encoding='utf-8',
                    format='%(asctime)s - PID %(process)d - %(name)s - %(levelname)s - %(message)s',
                    filemode='w')
handler = logging.FileHandler(pl.Path.cwd().joinpath('logs', f"mainlog-{420}.log"),
                              mode='a',
                              encoding='utf-8')
handler.setLevel(logging.DEBUG)

mylog = logging.getLogger(__name__)
mylog.addHandler(handler)

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
    mydataset = cds(ticker=config['ticker'],
                    split=config['pars-data']['data-split'],
                    seq_len=config['pars-data']['size-history'],
                    )
    # setting up and configuring model
    mylog.info(f"Setting up model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mylog.info(f"Using device: {device}")

    model = lstm(input_size  = len(mydataset['features-train'][0]),
                 hidden_size = config['pars-model']['size-hidden'],
                 num_layers  = config['pars-model']['num-layers'],
                 output_size = 1,
                 dropout     = config['pars-model']['dropout'],
                 ).to(device)
    mylog.info(f"Successfully mapped model to {device} device.")
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['pars-learning']['learning-rate'])

    # setting up pytorch DataLoaders
    dataset_train = torch.utils.data.TensorDataset(mydataset['features-train'],
                                                    mydataset['targets-train']
                                                  )
    dataset_vali = torch.utils.data.TensorDataset(mydataset['features-vali'],
                                                    mydataset['targets-vali']
                                                  )
    loader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=config['pars-learning']['size-batch'],
                                               shuffle=False,
                                               num_workers=config['pars-learning']['num-workers'],
                                               )
    loader_vali = torch.utils.data.DataLoader(dataset_vali,
                                               batch_size=config['pars-learning']['size-batch'],
                                               shuffle=False,
                                               num_workers=config['pars-learning']['num-workers'],
                                               )

    # training and evaluation loop
    num_epochs = config['pars-learning']['num-epochs']
    hist_train = np.zeros(num_epochs, dtype=np.float64)
    hist_vali  = np.zeros(num_epochs, dtype=np.float64)

    for epoch in tqdm(range(num_epochs)):
        loss_total = 0.0
        model.train()
        for batch_x, batch_y in loader_train:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = loss_fn(model(batch_x).reshape(len(batch_x)), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        hist_train[epoch] = loss_total / len(loader_train)

        model.eval()
        with torch.no_grad():
            loss_total_vali = 0.0
            for batch_x_vali, batch_y_vali in loader_vali:
                batch_x_vali = batch_x_vali.to(device)
                batch_y_vali = batch_y_vali.to(device)
                predictions_vali = model(batch_x_vali)
                loss_vali = loss_fn(predictions_vali.reshape(len(batch_x_vali)), batch_y_vali)
                loss_total_vali += loss_vali.item()
                hist_vali[epoch] = loss_total_vali / len(loader_vali)

        if (epoch + 1) % 10 == 0:
            infoline = f"Epoch [{epoch + 1:3}/{num_epochs:3}] - Training Loss: {hist_train[epoch]:.6f}, Validation Loss: {hist_vali[epoch]:.6f}"
            print(infoline)
            mylog.info(infoline)

    torch.save(model.state_dict(),
               pl.Path.cwd().joinpath("models", "lstm_model.pt")
               )
    return 0

if __name__ == '__main__':
    main()
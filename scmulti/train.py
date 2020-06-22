import argparse
import numpy as np
import time
import os
import json
from utils import create_optimizer, parse_config_file
from datasets import load_dataset
from models import create_model
import torch


def train(**config):
    # config torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.manual_seed(config['train']['seed'])

    # load train and validation datasets
    train_datasets, val_datasets = load_dataset(config['dataset'], device)
    
    # create the model to be trained
    model = create_model(config['model'], device)

    # load training configurations
    n_epochs = config['train']['n-epochs']
    early_stopping_limit = config['train'].get('early-stopping', None)
    optimizer = create_optimizer(model.parameters(), config['train']['optimizer'])
    print(model)
    print('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # do the training epochs
    best_model = None
    best_loss = np.inf
    early_stopping_count = 0

    for epoch in range(n_epochs):
        train_loss = 0
        epoch_time = time.time()

        # train
        model.train()
        for i_batch, datas in enumerate(zip(*train_datasets)):
            output, loss = model.forward(*datas)
            optimizer.zero_grad()
            model.backward()
            optimizer.step()

            train_loss += loss.item()
        
        epoch_time = time.time() - epoch_time
        early_stopping_count += 1

        # evaluate
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for _, datas in enumerate(zip(*val_datasets)):
                val_loss += model.test(*datas).item()
        
        print(f'epoch {epoch+1}/{n_epochs}: time={epoch_time:.2f}(s), loss={train_loss:.4f}, val_loss={val_loss:.4f}')

        # keep the best model
        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            early_stopping_count = 0
            print('new best.')
        
        # stop the training in case of early stopping
        if early_stopping_limit is not None and early_stopping_count > early_stopping_limit:                           
            print('early stopping.')
            break
    
    # save the best model and the experiment parameters
    output_dir = os.path.join(config['train']['output-dir'], config['experiment-name'])
    os.makedirs(output_dir, exist_ok=True)
    torch.save(best_model, os.path.join(output_dir, 'best-model.pt'))
    json.dump(config, open(os.path.join(output_dir, 'config.json'), 'w'), indent=2)
    
    return best_model


def parse_args():
    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('--config-file', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = parse_config_file(args.config_file)

    train(**config)
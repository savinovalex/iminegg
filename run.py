#!/usr/bin/env python3

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from im_in_egg_unet1 import ImineggNet1
from trainer import Iminegg
from my_iterable_dataset import MyIterableDataset

NAME = 'W128'
BS = 5
TRAIN_EPOCH_SIZE = 10
VAL_EPOCH_SIZE = 10

if __name__ == '__main__':
    print('Building net...')
    iminegg_net = ImineggNet1(2)
    iminegg = Iminegg(iminegg_net)

    print('Loading datasets...')
    ds_train = MyIterableDataset(TRAIN_EPOCH_SIZE, "./data/train")
    ds_validate = MyIterableDataset(VAL_EPOCH_SIZE, "./data/val")

    print('Creating training...')
    tb_logger = pl_loggers.TensorBoardLogger('logs/', name=NAME)

    print('Creating dataloaders...')
    train_set = DataLoader(ds_train, batch_size=BS)
    validate_set = DataLoader(ds_validate, batch_size=BS)

    print('Fitting...')
    trainer = pl.Trainer(max_epochs=300, gpus=[0, 1], logger=tb_logger, log_save_interval=10, track_grad_norm=2)
    trainer.fit(iminegg, train_set, validate_set)
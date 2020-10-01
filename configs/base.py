import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

from im_in_egg_unet1 import ImineggNet1
from my_iterable_dataset import MyIterableDataset
from trainer import Iminegg


class Config:
    W = 128
    BS = 20
    NAME = 'default'
    NPREF = ''
    NSUF = ''
    TRAIN_EPOCH_SIZE = 1000
    VAL_EPOCH_SIZE = 200

    GPUS = "0"

    @property
    def exp_name(self):
        exp_name = ''
        if self.NPREF:
            exp_name += f'{self.NPREF}-'

        exp_name += self.NAME
        if self.NSUF:
            exp_name += f'-{self.NSUF}-'

        exp_name += f'w{self.W}-bs{self.BS}'
        return exp_name

    def __init__(self):
        print('Building net...')
        iminegg_net = ImineggNet1(w=Config.W)
        self.model = Iminegg(iminegg_net)

        print('Loading datasets...')
        ds_train = MyIterableDataset(Config.TRAIN_EPOCH_SIZE, "./data/train")
        ds_validate = MyIterableDataset(Config.VAL_EPOCH_SIZE, "./data/val")

        print('Creating dataloaders...')
        self.dl_train = DataLoader(ds_train, batch_size=Config.BS)
        self.dl_validation = DataLoader(ds_validate, batch_size=Config.BS)

        self.tb_logger = pl_loggers.TensorBoardLogger('logs/', name=self.exp_name)

    def train(self):
        print(f'Fitting {self.exp_name} ...')
        trainer = pl.Trainer(max_epochs=10000, gpus=self.GPUS, logger=self.tb_logger, log_save_interval=10, track_grad_norm=2)
        trainer.fit(self.model, self.dl_train, self.dl_validation)
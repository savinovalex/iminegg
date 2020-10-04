import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

from im_in_egg_unet1_param import ImineggNet1
from my_iterable_dataset import MyIterableDataset
from trainer import Iminegg


class Config:
    W = 32
    DRP = 0.0
    BS = 40
    APPLY_BNM = False
    BLUR = False
    NAME = 'default'
    NPREF = ''
    NSUF = ''
    TRAIN_EPOCH_SIZE = 3000
    VAL_EPOCH_SIZE = 100
    DEV_RUN = False

    GPUS = "0"

    @property
    def exp_name(self):
        exp_name = ''
        if self.NPREF:
            exp_name += f'{self.NPREF}-'

        exp_name += self.NAME
        if self.NSUF:
            exp_name += f'-{self.NSUF}-'

        exp_name += f'w{self.W}-bs{self.BS}-drp{self.DRP}_relu0.2_add2l'
        if self.APPLY_BNM:
           exp_name += '-bnm'
        return exp_name

    def __init__(self, load_ckpt = None):
        self.load_ckpt = load_ckpt
    
        print('Building net...')
        iminegg_net = ImineggNet1(w=Config.W, blur=self.BLUR, apply_bnm=Config.APPLY_BNM, dropout_p=self.DRP)

        self.model = Iminegg(iminegg_net)

        print('Loading datasets...')
        ds_train = MyIterableDataset(self.TRAIN_EPOCH_SIZE, "./data/train")
        ds_validate = MyIterableDataset(self.VAL_EPOCH_SIZE, "./data/val")

        print('Creating dataloaders...')
        self.dl_train = DataLoader(ds_train, batch_size=self.BS)
        self.dl_validation = DataLoader(ds_validate, batch_size=self.BS)

        self.tb_logger = pl_loggers.TensorBoardLogger('logs/', name=self.exp_name)

    def train(self):
        print(f'Fitting {self.exp_name} ...')

        checkpoint_saver = pl.callbacks.ModelCheckpoint(
            save_top_k=2,
#            monitor='val/loss',
            verbose=True,
            mode='min'
        )

        trainer = pl.Trainer(max_epochs=3000, gpus=self.GPUS, logger=self.tb_logger, log_save_interval=10, track_grad_norm=2,
                             checkpoint_callback=checkpoint_saver, resume_from_checkpoint=self.load_ckpt,
                             fast_dev_run=self.DEV_RUN)
        trainer.fit(self.model, self.dl_train, self.dl_validation)
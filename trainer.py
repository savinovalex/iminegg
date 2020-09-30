import torch
from torch import nn
import pytorch_lightning as pl
from my_iterable_dataset import MyIterableDataset
import torch.nn.functional as F

class Iminegg(pl.LightningModule):
    
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inp):
        return self.net.calc(inp)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x = torch.unsqueeze(batch['input'], 1)
        y = torch.unsqueeze(batch['target'], 1)
        
        voice_noise_maskind = batch['type'] == MyIterableDataset.TYPE_VOICE_NOISE
        noise_maskind = batch['type'] == MyIterableDataset.TYPE_NOISE
        
        x_hat = self.net.calc(x)
        
        loss_voice_noise = F.mse_loss(x_hat[voice_noise_maskind], y[voice_noise_maskind])
        loss_noise = F.mse_loss(x_hat[noise_maskind], y[noise_maskind])
        
        loss = (loss_voice_noise + loss_noise) * 1000
        
        #loss = torch.norm(x_hat - y, p=2) #/ train_sample_length
        #print((x_hat-y).abs().max(), loss.item())
        
        logs = {
            "train_loss": loss,
            "train_voice_noise_psnr": -10 * loss_voice_noise.log(),
            "train_noise_psnr": -10 * loss_noise.log(),
        }

        res = pl.TrainResult(loss)
        res.log_dict(logs, on_step=True, on_epoch=True)

        return res
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        x = torch.unsqueeze(batch['input'], 1)
        y = torch.unsqueeze(batch['target'], 1)

        voice_noise_maskind = batch['type'] == MyIterableDataset.TYPE_VOICE_NOISE
        noise_maskind = batch['type'] == MyIterableDataset.TYPE_NOISE

        x_hat = self.net.calc(x)

        loss_voice_noise = F.mse_loss(x_hat[voice_noise_maskind], y[voice_noise_maskind])
        loss_noise = F.mse_loss(x_hat[noise_maskind], y[noise_maskind])

        loss = (loss_voice_noise + loss_noise) * 1000

        logs = {
            "loss": loss,
            "voice_noise_psnr": -10 * loss_voice_noise.log(),
            "noise_psnr": -10 * loss_noise.log(),
        }

        res = pl.EvalResult()
        res.log_dict(logs, on_step=False, on_epoch=True)

        return res

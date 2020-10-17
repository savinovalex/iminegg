import torch
import pytorch_lightning as pl
from my_iterable_dataset import MyIterableDataset
import torch.nn.functional as F

EPS = 1e-20


class Iminegg(pl.LightningModule):
    
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inp):
        return self.net.calc(inp)

    def step(self, batch):
        x = torch.unsqueeze(batch['input'], 1)
        y = torch.unsqueeze(batch['target'], 1)

        voice_noise_maskind = batch['type'] == MyIterableDataset.TYPE_VOICE_NOISE
        noise_maskind = batch['type'] == MyIterableDataset.TYPE_NOISE

        x_hat = self.net.calc(x)

        if voice_noise_maskind.sum() > 0:
            loss_voice_noise = F.mse_loss(x_hat[voice_noise_maskind], y[voice_noise_maskind])
        else:
            loss_voice_noise = torch.tensor(0).to(x)

        if noise_maskind.sum() > 0:
            loss_noise = F.mse_loss(x_hat[noise_maskind], y[noise_maskind])
        else:
            loss_noise = torch.tensor(0).to(x)

        loss = (loss_voice_noise + loss_noise) * 1000
        return loss, loss_voice_noise, loss_noise

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        loss, loss_voice_noise, loss_noise = self.step(batch)

        logs = {
            "train/loss": loss,
            "train/voice_noise_psnr": -10 * (EPS + loss_voice_noise).log10(),
            "train/noise_psnr": -10 * (EPS + loss_noise).log10(),
        }

        res = pl.TrainResult(loss)
        res.log_dict(logs, on_step=True, on_epoch=True)

        return res
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.994)
        return [optimizer], [scheduler]
    
    def validation_step(self, batch, batch_idx):
        loss, loss_voice_noise, loss_noise = self.step(batch)

        logs = {
            "val/loss": loss,
            "val/voice_noise_psnr": -10 * (EPS + loss_voice_noise).log10(),
            "val/noise_psnr": -10 * (EPS + loss_noise).log10(),
        }

        res = pl.EvalResult(checkpoint_on=loss)
        res.log_dict(logs, on_step=False, on_epoch=True)

        return res

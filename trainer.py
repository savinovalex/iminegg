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

    @staticmethod
    def masked_mse(pred, tgt, mask):
        if mask.sum() > 0:
            mse = F.mse_loss(pred[mask], tgt[mask])
        else:
            mse = 0
        return mse

    def step(self, batch):
        x = torch.unsqueeze(batch['input'], 1)
        y = torch.unsqueeze(batch['target'], 1)

        mask_vn = batch['type'] == MyIterableDataset.TYPE_VOICE_NOISE
        mask_n = batch['type'] == MyIterableDataset.TYPE_NOISE
        mask_v = batch['type'] == MyIterableDataset.TYPE_VOICE

        x_hat = self.net.calc(x)

        loss_vn = self.masked_mse(x_hat, y, mask_vn)
        loss_v  = self.masked_mse(x_hat, y, mask_v)
        loss_n  = self.masked_mse(x_hat, y, mask_n)

        loss = (loss_vn + loss_v + loss_n) * 1000

        logs = { "loss": loss }
        if loss_vn > 0:
            logs["voice_noise_psnr"] = -10 * loss_vn.log10()
            logs["noise_psnr"]: -10 * loss_n.log10()
            logs["voice_psnr"]: -10 * loss_v.log10()

        return loss, logs

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        loss, logs = self.step(batch)
        logs = {f"train/{k}": v for k, v in logs.items()}
        print(loss)
        res = pl.TrainResult(loss)
        res.log_dict(logs, on_step=False, on_epoch=True)

        return res
    
    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        logs = {f"val/{k}": v for k, v in logs.items()}

        res = pl.EvalResult(checkpoint_on=loss)
        res.log_dict(logs, on_step=False, on_epoch=True)

        return res

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


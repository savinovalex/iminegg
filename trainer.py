import torch
from torch import nn
import pytorch_lightning as pl
from my_iterable_dataset import MyIterableDataset
import torch.nn.functional as F

class Iminegg(pl.LightningModule):
    
    def __init__(self, net):
        super().__init__()
        self.net = net

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
            "voice_noise_psnr": -10 * loss_voice_noise.log(),
            "noise_psnr": -10 * loss_noise.log(),
        }
        batch_dictionary={
            #REQUIRED: It ie required for us to return "loss"
            "loss": loss,
            
            #optional for batch logging purposes
            "log": logs,
        }
        return batch_dictionary
    
    def training_epoch_end(self,outputs):
        #  the function is called after every epoch is completed

        # calculating average loss  
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # calculating correect and total predictions
        correct=1
        total=2

        # creating log dictionary
        tensorboard_logs = {'loss': avg_loss,"Accuracy": correct/total}

        epoch_dictionary={
            # required
            'loss': avg_loss,
            
            # for logging purposes
            'log': tensorboard_logs}

        return epoch_dictionary
    
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
        
        return {
            "val_loss": loss,
            "val_voice_noise_psnr": -10 * loss_voice_noise.log(),
            "val_noise_psnr": -10 * loss_noise.log(),
        }

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}
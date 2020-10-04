import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x


class BlurConv(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride, padding):
        assert stride == 2
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, ksize, 1, padding)
        self.register_buffer('blur', torch.tensor([[[1.0,4,6,4,1]]] * out_ch) / 16)

    def forward(self, inp):
        x = self.conv(inp)
        x = F.conv1d(x, weight=self.blur, groups=x.size(1), stride=2, padding=1)
        if x.size(2) % 2 == 1:
            x = torch.cat([x, x.new_zeros([x.size(0), x.size(1), 1])], dim=2)

        return x

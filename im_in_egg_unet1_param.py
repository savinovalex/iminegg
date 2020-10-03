import torch
from torch import nn
import torch.nn.functional as F
from pixel_shuffle import PixelShuffle1D

dbg_shape_print = False


class ImineggNet1(nn.Module):
    def _n_features(self, w):
        return [1, w, w*3, w*4, w*4, w*4, w*4, w*4, w*4, w*4]
    def _filter_sizes(self):
        return [65, 33, 17, 9, 9, 9, 9, 9]
    def _n_paddings(self, w):
        return [x // 2 for x in self._filter_sizes()]
    
    def __init__(self, w=128):
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv_down_list = nn.ModuleList()
        
        n_features = self._n_features(w) # len() = len(filter_sizes) + 1
        filter_sizes = self._filter_sizes()
        paddings = self._n_paddings(w)
        
        for i in range(len(filter_sizes)):
            self.conv_down_list.append( nn.Conv1d(n_features[i], n_features[i + 1], filter_sizes[i], 2, paddings[i]) )
        
        self.conv_btlneck = nn.Conv1d(w * 4, w * 4, 9, 2, 4)
        
        self.conv_up_list = nn.ModuleList()
        _first = True
        for i in range(len(filter_sizes) - 1, -1, -1):
            f_in = n_features[i + 2]
            if _first: _first = False 
            else: f_in *= 2
            f_out = n_features[i + 1] * 2
            self.conv_up_list.append( nn.Conv1d(f_in, f_out, filter_sizes[i], 1, paddings[i]) )
        
        self.conv_final = nn.Conv1d(w * 2, 1 * 2, 9, 1, 4)
        
        #self.conv4 = nn.Conv1d(512, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle2 = PixelShuffle1D(2)

       
    def downsampling(self, x):
        outs = []
        
        dbg_shape_print and print('DOWNSAMPLE', x.shape)
        
        for conv in self.conv_down_list:
            x = self.leaky_relu(conv(x))
            outs.append(x)
            dbg_shape_print and print(x.shape)

        return x, outs
    
    def bottle_neck(self, x):
        x = self.leaky_relu(self.dropout(self.conv_btlneck(x)))
        dbg_shape_print and print('BTLneck', x.shape)
        
        return x
        
    def upsampling(self, x, d_outs):
        dbg_shape_print and print('UPSAMPLE', x.shape)
        
        n = len(self.conv_up_list)
        for i, conv in enumerate(self.conv_up_list):
            x = conv(x)
            x = self.relu(self.dropout(x))
            dbg_shape_print and print('before pxl', x.shape)
            x = self.pixel_shuffle2(x)
            dbg_shape_print and print('pxl', x.shape)
            x = torch.cat((x, d_outs[n - i - 1]), dim=1)
            dbg_shape_print and print(x.shape)
        
        x = self.pixel_shuffle2(self.conv_final(x))
        dbg_shape_print and print(x.shape)
        #x = self.pixel_shuffle(x) # 32 left
        
        
        return x

    def calc(self, x):
        z = x
        x, d_outs = self.downsampling(x)
        x = self.bottle_neck(x)
        x = self.upsampling(x, d_outs)
        return x + z
import torch
from torch import nn
import torch.nn.functional as F
from pixel_shuffle import PixelShuffle1D

dbg_shape_print = True


class ImineggNet1(nn.Module):
    def __init__(self, w=128):
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv1d(1, w, 65, 2, 32)
        self.conv2 = nn.Conv1d(w, w*3, 33, 2, 16)
        self.conv3 = nn.Conv1d(w*3, w*4, 17, 2, 8)
        self.conv4 = nn.Conv1d(w*4, w*4, 9, 2, 4)
        self.conv5 = nn.Conv1d(w*4, w*4, 9, 2, 4)
        self.conv6 = nn.Conv1d(w*4, w*4, 9, 2, 4)
        self.conv7 = nn.Conv1d(w*4, w*4, 9, 2, 4)
        self.conv8 = nn.Conv1d(w*4, w*4, 9, 2, 4)
        
        self.conv_btlneck = nn.Conv1d(w*4, w*4, 9, 2, 4)
        
        self.conv8_up = nn.Conv1d(w*4, w*4 * 2, 9, 1, 4)
        self.conv7_up = nn.Conv1d(w*4 * 2, w*4 * 2, 9, 1, 4)
        self.conv6_up = nn.Conv1d(w*4 * 2, w*4 * 2, 9, 1, 4)
        self.conv5_up = nn.Conv1d(w*4 * 2, w*4 * 2, 9, 1, 4)
        self.conv4_up = nn.Conv1d(w*4 * 2, w*4 * 2, 9, 1, 4)
        self.conv3_up = nn.Conv1d(w*4 * 2, w*4 * 2, 17, 1, 8)
        self.conv2_up = nn.Conv1d(w*4 * 2, w*3 * 2, 33, 1, 16)
        self.conv1_up = nn.Conv1d(w*3 * 2, w * 2, 65, 1, 32)
        self.conv_final = nn.Conv1d(w * 2, 1 * 2, 9, 1, 4)
        
        #self.conv4 = nn.Conv1d(512, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = PixelShuffle1D(64)
        self.pixel_shuffle2 = PixelShuffle1D(2)

       
    def downsampling(self, x):
        outs = []
        
        dbg_shape_print and print('DOWNSAMPLE', x.shape)
        x = self.leaky_relu(self.conv1(x))
        outs.append(x)
        dbg_shape_print and print(x.shape)
        
        x = self.leaky_relu(self.conv2(x))
        outs.append(x)
        dbg_shape_print and print(x.shape)
        
        x = self.leaky_relu(self.conv3(x))
        outs.append(x)
        dbg_shape_print and print(x.shape)
        
        x = self.leaky_relu(self.conv4(x))
        outs.append(x)
        dbg_shape_print and print(x.shape)
        
        x = self.leaky_relu(self.conv5(x))
        outs.append(x)
        dbg_shape_print and print('conv5', x.shape)
        
        x = self.leaky_relu(self.conv6(x))
        outs.append(x)
        dbg_shape_print and print(x.shape)    
        
        x = self.leaky_relu(self.conv7(x))
        outs.append(x)
        dbg_shape_print and print(x.shape)
        
        x = self.leaky_relu(self.conv8(x))
        outs.append(x)
        dbg_shape_print and print(x.shape)

        return x, outs
    
    def bottle_neck(self, x):
        x = self.leaky_relu(self.dropout(self.conv_btlneck(x)))
        dbg_shape_print and print('BTLneck', x.shape)
        
        return x
        
    def upsampling(self, x, d_outs):
        dbg_shape_print and print('UPSAMPLE', x.shape)
        
        x = self.conv8_up(x)
        x = self.relu(self.dropout(x))
        dbg_shape_print and print('before pxl', x.shape)
        x = self.pixel_shuffle2(x)
        dbg_shape_print and print('pxl', x.shape)
        x = torch.cat((x, d_outs[8 - 1]), dim=1)
        dbg_shape_print and print(x.shape)    
        
        
        x = self.conv7_up(x)
        x = self.relu(self.dropout(x))
        dbg_shape_print and print('before pxl', x.shape)
        x = self.pixel_shuffle2(x)
        dbg_shape_print and print('pxl', x.shape)
        x = torch.cat((x, d_outs[7 - 1]), dim=1)
        dbg_shape_print and print(x.shape)    
        
        
        x = self.conv6_up(x)
        x = self.relu(self.dropout(x))
        dbg_shape_print and print('before pxl', x.shape)
        x = self.pixel_shuffle2(x)
        dbg_shape_print and print('pxl', x.shape)
        x = torch.cat((x, d_outs[6 - 1]), dim=1)
        dbg_shape_print and print(x.shape)    
        
        x = self.conv5_up(x)
        x = self.relu(self.dropout(x))
        dbg_shape_print and print('conv5_up before pxl', x.shape)
        x = self.pixel_shuffle2(x)
        dbg_shape_print and print('pxl', x.shape)
        x = torch.cat((x, d_outs[5 - 1]), dim=1)
        dbg_shape_print and print(x.shape)    
        
        x = self.conv4_up(x)
        x = self.relu(self.dropout(x))
        dbg_shape_print and print('before pxl', x.shape)
        x = self.pixel_shuffle2(x)
        dbg_shape_print and print('pxl', x.shape)
        x = torch.cat((x, d_outs[4 - 1]), dim=1)
        dbg_shape_print and print(x.shape)    
        
        x = self.conv3_up(x)
        x = self.relu(self.dropout(x))
        dbg_shape_print and print('before pxl', x.shape)
        x = self.pixel_shuffle2(x)
        dbg_shape_print and print('pxl', x.shape)
        x = torch.cat((x, d_outs[3 - 1]), dim=1)
        dbg_shape_print and print(x.shape)    
        
        x = self.conv2_up(x)
        x = self.relu(self.dropout(x))
        dbg_shape_print and print('before pxl', x.shape)
        x = self.pixel_shuffle2(x)
        dbg_shape_print and print('pxl', x.shape)
        x = torch.cat((x, d_outs[2 - 1]), dim=1)
        dbg_shape_print and print(x.shape)    
        
        x = self.conv1_up(x)
        x = self.relu(self.dropout(x))
        dbg_shape_print and print('before pxl', x.shape)
        x = self.pixel_shuffle2(x)
        dbg_shape_print and print('pxl', x.shape)
        x = torch.cat((x, d_outs[1 - 1]), dim=1)
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
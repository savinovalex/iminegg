import torch
from torch import nn
import torch.nn.functional as F
from modules import PixelShuffle1D, BlurConv
from im_in_egg_unet1_param import ImineggNet1

dbg_shape_print = False


class ImineggNet1_1(ImineggNet1):
    def _n_features(self, w):
        return [1, w, w*3, w*4, w*4, w*4, w*4, w*4, w*4, w*4, w*4, w*4, w*4]
    def _filter_sizes(self):
        return [33, 17, 9, 5, 5, 5, 5, 5, 5, 5]
    def _n_paddings(self, w):
        return [x // 2 for x in self._filter_sizes()]
    
	
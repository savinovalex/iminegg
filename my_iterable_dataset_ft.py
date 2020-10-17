import torch
import random
import os, glob
from scipy.io import wavfile
import re


class MyIterableDataset(torch.utils.data.IterableDataset):
    TYPE_VOICE = 1
    TYPE_VOICE_NOISE = 2
    TYPE_NOISE = 3
    
    def __init__(self, data_len, base_dir, train_sample_length = 22528):
        super().__init__()
        self.data_len = data_len
        
        print('loading tracks')
        self.tracks = []
        for file in glob.glob(os.path.join(base_dir, "*in_*.wav")):
            samplerate, data = wavfile.read(file)
            inp = torch.from_numpy(data).float() / 32768
            
            groups = re.match(r'(\d+)(.+?)_in_(\d+)\.wav', os.path.basename(file))
            if groups:
                target_file = os.path.join(base_dir, "{}{}_trgt.wav".format(groups[1], groups[2]))
                samplerate, data = wavfile.read(target_file)
                
                outp = torch.from_numpy(data).float() / 32768
                
                self.tracks.append({
                    'input': inp,
                    'target': outp,
                    'type': int(groups[3])
                })
        
        print('loaded')
        self.train_sample_length = train_sample_length

    def generate(self):
        return random.choice(self.tracks)
        
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.data_len
    
    def __next__(self):
        return self.generate()
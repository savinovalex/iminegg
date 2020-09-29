import torch
import random
import os, glob
from scipy.io import wavfile

voices = []
for file in glob.glob("./data/voice/*.wav"):
    samplerate, data = wavfile.read(file)
    voices.append(torch.from_numpy(data).float() / 32768)
noises = []
for file in glob.glob("./data/noise/*.wav"):
    _, data = wavfile.read(file)
    noises.append(torch.from_numpy(data).float() / 32768)
    
train_sample_length = 22528 #int(samplerate * 2)
silence = torch.zeros(train_sample_length)


class MyIterableDataset(torch.utils.data.IterableDataset):
    TYPE_VOICE = 1
    TYPE_VOICE_NOISE = 2
    TYPE_NOISE = 3
    
    def __init__(self, data_len, ):
        super().__init__()
        self.data_len = data_len
        

    def _voice_sample(self):
        s = voices[random.randint(0, len(voices) - 1)]
        offset = random.randint(0, len(s) - train_sample_length - 1)
        return s[offset:offset + train_sample_length]
    
    def _noise_sample(self):
        s = noises[random.randint(0, len(noises) - 1)]
        offset = random.randint(0, len(s) - train_sample_length - 1)
        return s[offset:offset + train_sample_length]
        
    def generate_with_noise(self):
        voice = self._voice_sample()
        noise = self._noise_sample()
        return {'input': voice + noise / random.uniform(6, 14), 'target': voice, 'type': self.TYPE_VOICE_NOISE}
    
    def generate_voice(self):
        voice = self._voice_sample()
        return {'input': voice, 'target': voice, 'type': self.TYPE_VOICE}
    
    def generate_noise(self):
        noise = self._noise_sample()
        return {'input': noise / random.uniform(6, 14), 'target': silence, 'type': self.TYPE_NOISE}
    
    def generate(self):
        r = random.uniform(0, 1)
        if r < 0.5:
            return self.generate_with_noise()
        if r < 1.0:
            return self.generate_noise()
        return self.generate_noise()
        
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.data_len
    
    def __next__(self):
        return self.generate()
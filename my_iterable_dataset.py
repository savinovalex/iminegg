import torch
import random
import os, glob
from scipy.io import wavfile


class MyIterableDataset(torch.utils.data.IterableDataset):
    TYPE_VOICE = 1
    TYPE_VOICE_NOISE = 2
    TYPE_NOISE = 3
    
    def __init__(self, data_len, base_dir, train_sample_length = 22528):
        super().__init__()
        self.data_len = data_len
        
        self.voices = []
        for file in glob.glob(os.path.join(base_dir, "voice", "*.wav")):
            samplerate, data = wavfile.read(file)
            self.voices.append(torch.from_numpy(data).float() / 32768)
        self.noises = []
        for file in glob.glob(os.path.join(base_dir, "noise", "*.wav")):
            _, data = wavfile.read(file)
            self.noises.append(torch.from_numpy(data).float() / 32768)

        self.train_sample_length = train_sample_length
        self.silence = torch.zeros(train_sample_length)

    def _voice_sample(self):
        s = self.voices[random.randint(0, len(self.voices) - 1)]
        offset = random.randint(0, len(s) - self.train_sample_length - 1)
        return s[offset:offset + self.train_sample_length]
    
    def _noise_sample(self):
        s = self.noises[random.randint(0, len(self.noises) - 1)]
        offset = random.randint(0, len(s) - self.train_sample_length - 1)
        return s[offset:offset + self.train_sample_length]
        
    def generate_with_noise(self):
        voice = self._voice_sample()
        noise = self._noise_sample()
        return {'input': voice + noise / random.uniform(1, 3), 'target': voice, 'type': self.TYPE_VOICE_NOISE}
    
    def generate_voice(self):
        voice = self._voice_sample()
        return {'input': voice, 'target': voice, 'type': self.TYPE_VOICE}
    
    def generate_noise(self):
        noise = self._noise_sample()
        return {'input': noise / random.uniform(1, 3), 'target': self.silence, 'type': self.TYPE_NOISE}
    
    def generate(self):
        r = random.uniform(0, 1)
        if r < 0.4:
            return self.generate_with_noise()
        if r < 0.8:
            return self.generate_noise()
        return self.generate_voice()
        
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.data_len
    
    def __next__(self):
        return self.generate()
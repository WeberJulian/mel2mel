# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from mozilla_voice_tts.utils.audio import AudioProcessor
from mozilla_voice_tts.utils.io import load_config
from mozilla_voice_tts.tts.utils.visual import plot_spectrogram
import librosa
import glob
import IPython.display as ipd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import json
from multiprocessing import Pool



# %%
class Mel2MelDataset(Dataset):
    def __init__(self, path, INPUT_SR, TARGET_SR, WINDOW_LENGTH):
        self.INPUT_SR = INPUT_SR
        self.TARGET_SR = TARGET_SR
        self.WINDOW_LENGTH = WINDOW_LENGTH
        self.CONFIG = load_config('config_fr.json')
        self.CONFIG['audio']['sample_rate'] = self.INPUT_SR
        self.AP_INPUT = AudioProcessor(**self.CONFIG['audio'])
        self.CONFIG['audio']['sample_rate'] = self.TARGET_SR
        self.AP_TARGET = AudioProcessor(**self.CONFIG['audio'])
        self.files = glob.glob(path+'/**/*.wav',recursive=True)
        #If you change your dataset, delete cache.json
        if os.path.isfile('./cache.json'):
            with open('./cache.json', "r") as json_file:
                self.pre_repertoir = json.load(json_file)
        else:
            print("> Computing wave files length...")
            self.pre_repertoir = [librosa.get_duration(filename=file) for file in tqdm(self.files)]
            with open('./cache.json', mode="w") as json_file:
                json.dump(self.pre_repertoir, json_file)
        self.repertoir = [int(item / WINDOW_LENGTH) for item in self.pre_repertoir]
        self.length = self.get_len()
        
    def __len__(self):
        return self.length

    def __getitem__(self, id):
        ref = self.get_reference(id)
        input_wav, _ = librosa.load(self.files[ref[0]], offset=self.WINDOW_LENGTH*ref[1], duration=self.WINDOW_LENGTH)
        target_wav = librosa.resample(input_wav, self.INPUT_SR, self.TARGET_SR)
        input = torch.tensor(self.AP_INPUT.melspectrogram(input_wav))
        target = torch.tensor(self.AP_TARGET.melspectrogram(target_wav))
        scale_factor=(target.shape[0]/input.shape[0],target.shape[1]/input.shape[1])
        input = torch.nn.functional.interpolate(input.unsqueeze(0).unsqueeze(0), scale_factor=scale_factor, mode='bilinear').reshape(target.shape)
        #return (self.normalize(input), self.normalize(target))
        return {
            'image': self.normalize(input).unsqueeze(0).type(torch.FloatTensor),
            'mask': self.normalize(target).unsqueeze(0).type(torch.FloatTensor)
        }
    
    def get_reference(self, id):
        i = 0
        sum = 0
        while True:
            if(sum > id):
                return (i - 1,  id - sum + self.repertoir[i-1])
            else:
                sum += self.repertoir[i]
                i+=1
    
    def get_len(self):
        sum = 0
        for num in self.repertoir:
            sum += num
        return sum
    
    def normalize(self, tensor):
        return tensor / 8 + 0.5
    
    def denormalize(self, tensor):
        return tensor - 0.5 * 8


# %%
INPUT_SR = 16000
TARGET_SR = 22050
WINDOW_LENGTH = 1.07412

dataset = Mel2MelDataset('/home/ubuntu/mailabs', INPUT_SR, TARGET_SR, WINDOW_LENGTH)
#dataset = Mel2MelDataset('/Users/julian/workspace/ML/mailabs', INPUT_SR, TARGET_SR, WINDOW_LENGTH)


# %%
os.makedirs('images/')
os.makedirs('masks/')

def f(i):
    item = dataset.__getitem__(i)
    torch.save(item['image'], 'images/'+str(i)+'.pt')
    torch.save(item['mask'], 'masks/'+str(i)+'.pt')

if __name__ == '__main__':
    with Pool() as p:
        p.map(f,list(range(dataset.length)))


# %%




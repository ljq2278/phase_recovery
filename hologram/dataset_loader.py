import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np


class MyDataset(data.Dataset):
    def __init__(self, dir_input, dir_label):
        super().__init__()
        self.dir_input_amp = dir_input + "/amp"
        self.dir_input_phase = dir_input + "/phase"
        self.dir_label_amp = dir_label + "/amp"
        self.dir_label_phase = dir_label + "/phase"
        self.input_amp_names, self.input_phase_names, self.label_amp_names, self.label_phase_names = self.make_dataset()

    def __getitem__(self, index):
        input_amp = np.array(Image.open(os.path.join(self.dir_input_amp, self.input_amp_names[index])).convert('L')).astype(float)
        input_phase = np.array(Image.open(os.path.join(self.dir_input_phase, self.input_phase_names[index])).convert('L')).astype(float)
        label_amp = np.array(Image.open(os.path.join(self.dir_label_amp, self.label_amp_names[index])).convert('L')).astype(float)
        label_phase = np.array(Image.open(os.path.join(self.dir_label_phase, self.label_phase_names[index])).convert('L')).astype(float)
        return (input_amp, input_phase), (label_amp, label_phase)

    def __len__(self):
        return len(self.input_amp_names)

    def make_dataset(self):
        return sorted(os.listdir(self.dir_input_amp)), sorted(os.listdir(self.dir_input_phase)), sorted(os.listdir(self.dir_label_amp)), sorted(os.listdir(self.dir_label_phase))


if __name__ == '__main__':
    dataset = MyDataset(r"D:\PycharmProjects\phase_recovery\hologram\dataset\train\input", r"D:\PycharmProjects\phase_recovery\hologram\dataset\train\label")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    for images, labels in dataloader:
        # Forward pass through your network
        tt = 1

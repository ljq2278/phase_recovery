""" Full assembly of the parts to form the complete network """
import numpy as np
import torch
import torch.nn as nn
from modules import Conv, UpsampleBlock, ResidualBlock
from dataset_loader import MyDataset
from torch.utils.data import DataLoader

class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()
        self.init_convs = [Conv(2, 16), Conv(), Conv(), Conv()]
        self.down_samples = [nn.MaxPool2d(1, stride=1), nn.MaxPool2d(2, stride=2), nn.MaxPool2d(4, stride=4), nn.MaxPool2d(8, stride=8)]
        self.residual_blocks = [[ResidualBlock(), ResidualBlock(), ResidualBlock(), ResidualBlock()],
                                [ResidualBlock(), ResidualBlock(), ResidualBlock(), ResidualBlock()],
                                [ResidualBlock(), ResidualBlock(), ResidualBlock(), ResidualBlock()],
                                [ResidualBlock(), ResidualBlock(), ResidualBlock(), ResidualBlock()]]
        self.upsample_blocks = [[],
                                [UpsampleBlock()],
                                [UpsampleBlock(), UpsampleBlock()],
                                [UpsampleBlock(), UpsampleBlock(), UpsampleBlock()]]
        self.final_convs = [Conv(), Conv(), Conv(), Conv()]
        self.end_conv = Conv(64, 2)

    def forward(self, x):
        outputs = []

        ################################### i: path index, j: operation_block index ####################
        # init convs
        for i in range(0, 4):
            outputs.append(self.init_convs[i](x))
            x = outputs[-1]

        # down sample
        for i in range(0, 4):
            outputs[i] = self.down_samples[i](outputs[i])

        # residual ops.
        for i in range(0, 4):
            for j in range(0, 4):
                outputs[i] = self.residual_blocks[i][j](outputs[i])

        # up sampling blocks.
        for i in range(0, 4):
            for j in range(0, i):
                outputs[i] = self.upsample_blocks[i][j](outputs[i])

        # final convs of every path
        for i in range(0, 4):
            outputs[i] = self.final_convs[i](outputs[i])

        # merge and last conv
        output = torch.concatenate(outputs, dim=1)
        output = self.end_conv(output)

        return output


if __name__ == '__main__':
    H = 256
    W = 256
    bz = 3
    lr = 1e-4
    loss_accept_thresh = 0.1
    resUnet = ResUNet()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam([
        {'params': resUnet.parameters(), 'lr': lr},
    ])
    flg = "img_test"
    ######################################################## train ##############################################################
    if flg == "simple_test":
        torch_inputs = torch.zeros([bz, 2, H, W], dtype=torch.float)  # load your data
        torch_labels = torch.ones([bz, 2, H, W], dtype=torch.float)
        loss = 1000
        while loss > loss_accept_thresh:
            torch_outputs = resUnet(torch_inputs)
            loss = loss_func(torch_outputs, torch_labels)
            # take gradient step
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            print(loss)
    elif flg == "img_test":
        input_dir = r"D:\PycharmProjects\phase_recovery\hologram\dataset\train\input"
        label_dir = r"D:\PycharmProjects\phase_recovery\hologram\dataset\train\label"
        mydataset = MyDataset(input_dir, label_dir)
        dataloader = DataLoader(mydataset, batch_size=bz, shuffle=True)
        for inputs, labels in dataloader:
            input_amp, input_phase = inputs
            label_amp, label_phase = labels
            torch_inputs = torch.tensor(np.stack([input_amp, input_phase], axis=1), dtype=torch.float)
            torch_labels = torch.tensor(np.stack([label_amp, label_phase], axis=1), dtype=torch.float)
            torch_outputs = resUnet(torch_inputs)
            loss = loss_func(torch_outputs, torch_labels)
            # take gradient step
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            print(loss)
            tt = 1
            # get the data, according to your memory size. for example:

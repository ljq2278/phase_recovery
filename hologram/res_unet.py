""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
from modules import Conv, UpsampleBlock, ResidualBlock


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
    bz = 10
    lr = 1e-4
    loss_accept_thresh = 0.1
    resUnet = ResUNet()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam([
        {'params': resUnet.parameters(), 'lr': lr},
    ])

    ################################################### simple test #########################################
    test_input = torch.zeros([bz, 2, H, W], dtype=torch.float)  # load your data
    test_label = torch.ones([bz, 2, H, W], dtype=torch.float)
    loss = 1000
    while loss > loss_accept_thresh:
        test_output = resUnet(test_input)
        loss = loss_func(test_output, test_label)
        # take gradient step
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        print(loss)


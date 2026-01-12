import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class AlphaZeroResNet(nn.Module):
    def __init__(self, board_size=8, num_res_blocks=4, num_channels=64, input_channels=4):
        super(AlphaZeroResNet, self).__init__()
        self.board_size = board_size
        
        # Input: Dynamic channels
        self.start_conv = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.start_bn = nn.BatchNorm2d(num_channels)
        
        # Residual Tower
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels, num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, stride=1, padding=0)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, 2 * board_size * board_size) # 128 actions
        
        # Value Head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, stride=1, padding=0)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch, 4, 8, 8)
        x = F.relu(self.start_bn(self.start_conv(x)))
        
        for res_block in self.res_blocks:
            x = res_block(x)
            
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        # Note: We return Logits here. Softmax will be applied in loss or MCTS.
        
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)) # Value between -1 and 1
        
        return p, v

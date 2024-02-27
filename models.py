import torch
import torch.nn as nn

class BasicConvolution(nn.Module):
    def __init__(self, input_ch : int, output_ch : int) -> None:
        super().__init__()

        self.activation    = nn.LeakyReLU()
        self.convolution   = nn.Conv2d(input_ch, output_ch, kernel_size = 5, padding = 'same')
        self.normalization = nn.BatchNorm2d(output_ch)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x

# option 1 
class ResidualConv(nn.Module):
    def __init__(self, channels : int) -> None:
        super().__init__()

        self.conv = BasicConvolution(channels, channels)
    
    def forward(self, x : torch.Tensor):
        y = self.conv(x)
        y = x + y
        return y 


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # 3, 128, 128
        self.conv_1 = BasicConvolution(1, 16)
        self.maxp_1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # 16, 64, 64
        self.conv_2 = BasicConvolution(16, 32)
        self.maxp_2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # 32, 32, 32
        self.avgp_3 = nn.AdaptiveAvgPool2d(1)

        # 32, 1, 1
        self.conv_3 = nn.Conv2d(32, 3, kernel_size = 1)
        self.finact = nn.Softmax(dim = 1)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.maxp_1(x)

        x = self.conv_2(x)
        x = self.maxp_2(x)

        x = self.avgp_3(x)
        x = self.conv_3(x)
        x = self.finact(x)

        # reshape to N, 3
        x = x.view(x.size(0), 3)
        return x
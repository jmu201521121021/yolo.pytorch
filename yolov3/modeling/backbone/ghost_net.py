
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov3.layers import get_activate, get_norm

class DWConv(nn.Module):
    def __init__(self, input_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1):
        super(DWConv,self).__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, kernel_size, stride, padding, groups=input_channels, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.Conv2d(input_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.layers(x)

class SELayer(nn.Module):
    def __init__(self, input_channels, reduction):
        super(SELayer, self).__init__()
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.f_sq = nn.Linear(input_channels, input_channels // reduction, bias=False)
        self.f_ex = nn.Linear(input_channels // reduction, input_channels, bias=False)

    def forward(self, x):
        squeeze = self.global_avg(x)
        batch, channel, _, _ = squeeze.size()
        squeeze = squeeze.view(batch, channel)
        excitation =  self.f_sq(squeeze)
        excitation = F.relu(excitation, inplace=True)
        excitation = self.f_ex(excitation)
        excitation = torch.sigmoid(excitation)

        excitation = excitation.view(batch, channel, 1, 1)
        out = x * excitation
        return out


class GhostModule(nn.Module):
    def __init__(self, input_channels,
                      output_channels,
                      ratio,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      use_bias=True,
                      linear_kernel=3,
                      activation="ReLU",
                      alpha=0,):
        super(GhostModule, self).__init__()
        assert output_channels % ratio == 0, "not support format with param s"
        self.middle_channels = int(output_channels // ratio)
        self.conv = nn.Conv2d(input_channels, self.middle_channels, kernel_size, stride, padding, bias=use_bias)
        self.linear_transfer = nn.Conv2d(self.middle_channels, (output_channels - self.middle_channels), linear_kernel, 1,
                                         padding=int(linear_kernel/2), groups=self.middle_channels, bias=False)
        if activation is not  None:
            self.activation = get_activate(activation, alpha)
        else:
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not  None:
            x = self.activation(x)
        linear_out = self.linear_transfer(x)
        if self.activation is not  None:
            linear_out = self.activation(linear_out)

        out = torch.cat((x, linear_out), 1)
        return out

class Bottlenecks(nn.Module):
    def __init__(self, input_channels,
                        expand_ratio,
                        ratio,
                        linear_size=3,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        use_bias=False,
                        norm = "BN",
                        activation="ReLU",
                        alpha=0,
                        use_se=False,
                        ):
        super(Bottlenecks, self).__init__()

        middle_channels = input_channels * expand_ratio
        self.ghost_module0 = nn.Sequential(GhostModule(input_channels, middle_channels, ratio, kernel_size,1, padding, linear_kernel=linear_size, use_bias=use_bias),
                                           get_activate(activation, alpha),
                                           get_norm(norm, middle_channels),
                                           )
        self.ghost_module1 = nn.Sequential(GhostModule(middle_channels, input_channels, ratio, kernel_size, 1, padding, linear_kernel=linear_size, use_bias=use_bias),
                                            get_norm(norm, input_channels),
                                           )
        self.dw_stride2x2 = nn.Conv2d(middle_channels, middle_channels, kernel_size, stride, padding, groups=middle_channels) if stride == 2 else None
        self.stride2x2_res = DWConv(input_channels, input_channels, kernel_size, stride, padding) if stride == 2 else None
        self.se_layer = SELayer(middle_channels, 4) if use_se else None

    def forward(self, x):
        shortcut = x
        # ghost_module1
        out = self.ghost_module0(x)
        # stride = 2
        if self.dw_stride2x2 is not  None:
            shortcut = self.stride2x2_res(shortcut)
            out = self.dw_stride2x2(out)
        #SE
        if self.se_layer is not None:
            out = self.se_layer(out)
        # ghost_module2
        out = self.ghost_module1(out)
        # add
        out = out + shortcut
        return  out

if __name__ == "__main__":
    se_layer = SELayer(32, 16)
    ghost_module = GhostModule(3, 32, 4)
    print(ghost_module)
    x = torch.randn(1, 3, 256, 256)
    out = ghost_module(x)
    se_out = se_layer(out)
    x_bls = torch.randn(1, 16, 256, 256)
    bottlenecks = Bottlenecks(16, 2, 2, stride=2, use_se=True)
    print(bottlenecks)
    out_bls = bottlenecks(x_bls)

    print(out_bls.size())
    print(out.size(), se_out.size())

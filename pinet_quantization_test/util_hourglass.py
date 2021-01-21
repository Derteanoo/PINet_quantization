#########################################################################
##
## Some utility for training, data processing, and network.
##
#########################################################################
import torch
import torch.nn as nn
from parameters import Parameters
from UNet_2Plus import UNet_2Plus
p = Parameters()

######################################################################
##
## Convolution layer modules
##
######################################################################
class Conv2D_BatchNorm_Relu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=True, acti=True, dilation=1):
        super(Conv2D_BatchNorm_Relu, self).__init__()

        if acti:
            self.cbr_unit = nn.Sequential(nn.Conv2d(in_channels, n_filters, k_size, 
                                                    padding=padding, stride=stride, bias=bias, dilation=dilation),
                                    nn.BatchNorm2d(n_filters),
                                    nn.ReLU())
        else:
            self.cbr_unit = nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class bottleneck_dilation(nn.Module):
    def __init__(self, in_channels, out_channels,residual=False):
        super(bottleneck_dilation, self).__init__()
        self.residual = residual
        temp_channels = in_channels//4
        self.conv1 = Conv2D_BatchNorm_Relu(in_channels, temp_channels, 1, 0, 1)
        self.conv2 = Conv2D_BatchNorm_Relu(temp_channels, temp_channels, 3, 1, 1, dilation=1)
        self.conv3 = nn.Conv2d(temp_channels, out_channels, 1, padding=0, stride=1, bias=True)
        self.ConvRelu = nn.Sequential(nn.Conv2d(temp_channels, out_channels, 1, padding=0, stride=1, bias=True),
                                      nn.ReLU())

    def forward(self, x, ):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.residual:
            out =  self.conv3(out)
        else:
            out = self.ConvRelu(out)
        return out

class Output(nn.Module):
    def __init__(self, in_size, out_size):
        super(Output, self).__init__()
        self.conv1 = Conv2D_BatchNorm_Relu(in_size, in_size//2, 3, 1, 1, dilation=1)
        self.conv2 = Conv2D_BatchNorm_Relu(in_size//2, in_size//4, 3, 1, 1, dilation=1)
        self.conv3 = Conv2D_BatchNorm_Relu(in_size//4, out_size, 1, 0, 1, acti = False)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs

class resize_layer(nn.Module):
    def __init__(self, in_channels, out_channels, acti = True):
        super(resize_layer, self).__init__()
        self.conv1 = Conv2D_BatchNorm_Relu(in_channels, out_channels // 4, 3, 1, 2, dilation=1, acti=True)
        self.conv2 = Conv2D_BatchNorm_Relu(out_channels // 4, out_channels // 2, 3, 1, 2, dilation=1, acti=True)
        self.conv3 = Conv2D_BatchNorm_Relu(out_channels // 2, out_channels // 1, 3, 1, 2, dilation=1, acti=False)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs   

class hourglass_block(nn.Module):
    def __init__(self, in_channels, out_channels, quantization=True):
        super(hourglass_block, self).__init__()
        self.layer1 = UNet_2Plus()
        self.add = Conv2D_BatchNorm_Relu(in_channels, in_channels, 1, 0, 1)  # new_add
        self.cbr = Conv2D_BatchNorm_Relu(out_channels, out_channels, 1, 0, 1)
        self.re1 = bottleneck_dilation(out_channels, out_channels)
        self.re2 = nn.Conv2d(out_channels, out_channels, 1, padding=0, stride=1, bias=True, dilation=1)
        self.re3 = nn.Conv2d(1, out_channels, 1, padding=0, stride=1, bias=True, dilation=1)

        self.out_confidence = Output(out_channels, 1)     
        self.out_offset = Output(out_channels, 2)      
        self.out_instance = Output(out_channels, p.feature_size)

        self.bnrelu = Conv2D_BatchNorm_Relu(1, 1, 1, 0, 1)
        self.skip_add = nn.quantized.FloatFunctional()
        self.quantization = quantization

        
    def forward(self, inputs):
        inputs_a = self.add(inputs)
        outputs, feature = self.layer1(inputs_a)
        outputs_a = self.cbr(outputs)
        outputs_a = self.re1(outputs_a)

        outputs = self.re2(outputs_a)

        out_confidence = self.out_confidence(outputs_a)
        out_offset = self.out_offset(outputs_a)
        out_instance = self.out_instance(outputs_a)

        out = self.bnrelu(out_confidence)
        out = self.re3(out)

        if self.quantization:
            outputs = self.skip_add.add(self.re2(self.skip_add.add(outputs, out)), inputs)
        else:
            outputs = self.re2(outputs + out) + inputs

        return [out_confidence, out_offset, out_instance], outputs, feature
